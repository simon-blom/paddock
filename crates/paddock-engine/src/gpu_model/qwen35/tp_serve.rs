//! Rank-0-authoritative TP=2 serving over the distributed control channel.
//! The production scheduler owns rank 0's dense slot plan; rank 1 replays the
//! ordered live rows and mirrored KV operations. Each row uses eager kernels.
use std::{
    collections::VecDeque,
    io::Read,
    net::TcpStream,
    path::Path,
    sync::Arc,
    time::Duration,
};

use cudarc::driver::CudaEvent;
use paddock_dist::{
    config::Resolved,
    protocol::{
        ControlMessage, TpSpanFinisherPlan, receive_nccl_id, send_nccl_id,
    },
    worker::shutdown_worker,
};
use paddock_models::mapped::MappedGguf;

use super::{
    tp_kv::{Event, MirroredKv, Operation, Snapshot},
    tp_model::Qwen35TpRank,
};
use crate::{
    generator::{GenError, Generator, RowSample, SampledStep},
    gpu::{
        GpuExecutor, KvDtype,
        distributed::{NcclCommunicator, create_unique_id},
    },
};

const PINNED_SHA256: &str = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482";
const READY_TIMEOUT: Duration = Duration::from_secs(600);
const STEP_TIMEOUT: Duration = Duration::from_secs(120);

/// The wire spelling of a KV dtype, shared by both ranks (Phase 12). Rank 0
/// resolves the runner's dtype gate and sends the resolved value; rank 1
/// parses the same string. Only the two dtypes the TP GQA path implements
/// exist - anything else fails closed at both ends before any allocation.
pub(super) fn kv_dtype_wire(dtype: KvDtype) -> &'static str {
    match dtype {
        KvDtype::Fp16 => "fp16",
        KvDtype::Fp8E4m3 => "fp8_e4m3",
    }
}

pub(super) fn kv_dtype_parse(wire: &str) -> Result<KvDtype, String> {
    match wire {
        "fp16" => Ok(KvDtype::Fp16),
        "fp8_e4m3" => Ok(KvDtype::Fp8E4m3),
        other => Err(format!(
            "TP kv_dtype {other:?} not recognized (expected fp16 or fp8_e4m3)"
        )),
    }
}

/// The KV dtype this serve runs. Rank 0 resolves it against the device's
/// compute capability (the same sm_89 rule the runner's `apply_kv_dtype`
/// applies for TP=1) and SENDS the resolved value; rank 1 parses what
/// arrives. The ranks can therefore never disagree, which is what the old
/// hardcoded `KvDtype::Fp16` pair silently guaranteed - and what an fp8 KV
/// serve must keep guaranteeing before either rank allocates.
fn kv_dtype_serve(cc: (u32, u32)) -> KvDtype {
    kv_dtype_from_env(
        std::env::var("PADDOCK_KV_CACHE_DTYPE").ok().as_deref(),
        cc,
    )
}

/// Pure core of [`kv_dtype_serve`] (host-testable): `PADDOCK_KV_CACHE_DTYPE`
/// value in, dtype out. `fp8_e4m3` is honored wherever
/// `gpu_support::fp8_kv_blocked` says the die can store fp8 KV - which is
/// every die this build serves today (fp8 STORAGE is software-emulated and
/// byte-exact; the arch allowlist has already refused the rest). If the
/// seam ever answers blocked again, demote LOUDLY to f16 (the fp8 ask halves
/// KV bytes; getting f16 doubles the pool and must stay attributable) - the
/// same rule the TP=1 runner applies, with the threshold living beside the
/// support table, not duplicated here.
fn kv_dtype_from_env(env: Option<&str>, cc: (u32, u32)) -> KvDtype {
    match env {
        Some("fp8_e4m3") => {
            if let Some(why) = paddock_models::gpu_support::fp8_kv_blocked(cc) {
                tracing::error!(
                    "kv cache: TP=2 asked for fp8_e4m3, but {why} (this GPU is sm_{:#02}{:#02}). \
                     Serving f16 instead. The KV pool is twice the size it would have been - \
                     lower max_ctx if the server no longer fits.",
                    cc.0,
                    cc.1
                );
                KvDtype::Fp16
            } else {
                tracing::info!("kv cache: fp8-e4m3 (--kv-cache-dtype; halves per-rank KV bytes)");
                KvDtype::Fp8E4m3
            }
        }
        _ => KvDtype::Fp16,
    }
}

fn hashes(model: &Path, pack: &Path) -> Result<(String, String), String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(model).map_err(|e| format!("model open: {e}"))?;
    let mut sha = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("model read: {e}"))?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    let checkpoint: String = sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if checkpoint != PINNED_SHA256 {
        return Err(format!(
            "TP=2 serving requires the pinned Qwen3.8 GGUF with SHA-256 {PINNED_SHA256}, got {checkpoint}"
        ));
    }
    let mut file = std::fs::File::open(pack).map_err(|e| format!("CUDA pack open: {e}"))?;
    let mut hasher = blake3::Hasher::new();
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("CUDA pack read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok((checkpoint, hasher.finalize().to_hex().to_string()))
}

fn ready(stream: &mut TcpStream, sequence: u64) -> Result<(), String> {
    match ControlMessage::from_stream(stream).map_err(|e| e.to_string())? {
        ControlMessage::TpReady { sequence: got } if got == sequence => Ok(()),
        ControlMessage::TpError { reason } | ControlMessage::Reject { reason } => Err(reason),
        other => Err(format!(
            "rank 1 reply out of order (expected {sequence}): {other:?}"
        )),
    }
}

fn prepared(stream: &mut TcpStream, sequence: u64) -> Result<(), String> {
    match ControlMessage::from_stream(stream).map_err(|e| e.to_string())? {
        ControlMessage::TpPrepared { sequence: got } if got == sequence => Ok(()),
        ControlMessage::TpError { reason } | ControlMessage::Reject { reason } => Err(reason),
        other => Err(format!("rank 1 did not prepare step {sequence}: {other:?}")),
    }
}

fn validate_multistep_positions(
    rows: &[(usize, u32, usize)],
    positions: &[usize],
    max_ctx: usize,
) -> Result<(), String> {
    let mut cursor = positions.to_vec();
    for &(slot, _, position) in rows {
        if slot >= cursor.len() || position >= max_ctx || position != cursor[slot] {
            return Err("TP multirow position diverged".into());
        }
        cursor[slot] += 1;
    }
    Ok(())
}

// The scheduler sends dense rows through its high-water slot; position 0 is
// a hole, including a newly admitted slot that has not finished prefill.
fn active_rows(
    tokens: &[u32],
    positions: &[u32],
    slots: usize,
) -> Result<Vec<(usize, u32, usize)>, String> {
    if tokens.len() != positions.len() || tokens.is_empty() || tokens.len() > slots {
        return Err("TP batch width mismatch".into());
    }
    Ok(positions
        .iter()
        .enumerate()
        .filter_map(|(slot, &position)| {
            (position != 0).then_some((slot, tokens[slot], position as usize))
        })
        .collect())
}

fn argmax_logits(logits: &[f32]) -> Result<u32, String> {
    let first = logits
        .first()
        .ok_or("TP speculative verify returned no logits")?;
    if !first.is_finite() {
        return Err("TP speculative verify returned non-finite logits".into());
    }
    let mut best = 0usize;
    for (i, &value) in logits.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err("TP speculative verify returned non-finite logits".into());
        }
        if value > logits[best] {
            best = i;
        }
    }
    u32::try_from(best).map_err(|_| "TP vocabulary exceeds u32 token IDs".into())
}

fn validate_sampled_rows(
    rows: &[(usize, u32, usize)],
    plans: &[RowSample],
    width: usize,
) -> Result<(), String> {
    if plans.len() != width || rows.iter().any(|&(slot, _, _)| slot >= width) {
        return Err("TP sampling plan width mismatch".into());
    }
    let mut active = vec![false; width];
    for &(slot, _, _) in rows {
        active[slot] = true;
    }
    for (slot, plan) in plans.iter().enumerate() {
        match (active[slot], plan) {
            (false, RowSample::Hole) | (true, RowSample::Host) => {}
            (true, RowSample::Device(plan)) => {
                super::tp_model::tp_sample_params(*plan).map_err(|e| e.to_string())?;
            }
            _ => return Err("TP sampling plan disagrees with live slots".into()),
        }
    }
    Ok(())
}

fn logical(max_ctx: usize, slots: usize) -> Result<MirroredKv, String> {
    let blocks = u32::try_from(
        max_ctx
            .div_ceil(crate::kv_pool::BLOCK_TOKENS)
            .checked_mul(slots)
            .ok_or("KV block count overflow")?,
    )
    .map_err(|_| "KV block count overflow")?;
    MirroredKv::new(blocks, slots, max_ctx).map_err(str::to_owned)
}

struct PipeFlight {
    /// Member slots in WIRE order (TpPipeBegin/Next rows must ascend by
    /// slot; identity rows for plain pipes).
    slots: Vec<usize>,
    width: usize,
    plane: usize,
    events: Vec<(usize, CudaEvent)>,
    /// Wire position -> scheduler row index. Identity for plain pipes; for
    /// slot-mapped pipes the scheduler's row i drives slots[i] in ITS order,
    /// which the wire sort permuted.
    row_of: Vec<usize>,
    /// Slot-mapped pipe (overlap path): readback ids are indexed by ROW;
    /// plain pipes keep the dense contract (ids indexed by slot).
    mapped: bool,
}

/// Runs on the production engine thread; owns the control stream until drop.
pub struct TpCoordinator {
    stream: TcpStream,
    group: NcclCommunicator,
    model: Qwen35TpRank,
    logical: MirroredKv,
    positions: Vec<usize>,
    occupied: Vec<bool>,
    sequence: u64,
    pipe: Option<PipeFlight>,
    /// Queued prompt chunks (FIFO) the scheduler advances with mixed ticks.
    chunks: VecDeque<PrefillChunk>,
    /// In-flight prefill span: its finishing chunks plus the rank-0 events
    /// marking each finisher's GPU completion (chunk order).
    span: Option<SpanFlight>,
    /// Shared span-done flag for the send-only proxy's `unified_span_done`
    /// (`&self`, no request round-trip). Published after every command.
    span_done: Option<Arc<std::sync::atomic::AtomicBool>>,
    poisoned: Option<String>,
}

/// One queued prompt chunk: the remaining ordered rows (slot, token,
/// position) plus the finishing chunk's device plan, if its first token
/// samples on device (`None` = full-logit host readback).
struct PrefillChunk {
    slot: usize,
    rows: Vec<(usize, u32, usize)>,
    fin_plan: Option<crate::sampler::DevicePlan>,
}

/// A finishing chunk's summary, kept from launch until the span drains so
/// `span_finish` can read each finisher's result in chunk order.
struct SpanFinisher {
    slot: usize,
    plan: Option<crate::sampler::DevicePlan>,
    rows: usize,
}

struct SpanFlight {
    /// The span's finishing chunks, in chunk order (outer = oldest).
    finishing: Vec<SpanFinisher>,
    /// Rank-0 events recorded after each finisher's GPU work was enqueued,
    /// in chunk order (front = oldest = first readback).
    events: VecDeque<CudaEvent>,
    /// Distinct slots with rows inside this span, row order. A prefill
    /// abort must not drop a slot whose rows are in flight here.
    slots: Vec<usize>,
}

/// Parse the wire KV-mirror end state on the worker side. Fails closed on
/// any malformed value: an unparseable snapshot is a protocol mismatch, not
/// a best-effort mirror.
fn wire_kv_state(value: serde_json::Value) -> Result<Snapshot, String> {
    serde_json::from_value(value).map_err(|e| format!("TP wire KV state malformed: {e}"))
}

/// Engine `DevicePlan` -> wire plan. Unsupported plans fail closed before
/// any KV authorization or worker message. The reverse conversion does not
/// exist: rank 1 never samples (the finisher sampler has no collectives and
/// runs only on rank 0), it only reads the finisher slot ids to promote
/// lane-local state.
fn wire_finisher_plan(
    plan: crate::sampler::DevicePlan,
) -> Result<TpSpanFinisherPlan, String> {
    match plan {
        crate::sampler::DevicePlan::Greedy => Ok(TpSpanFinisherPlan::Greedy),
        crate::sampler::DevicePlan::Categorical { inv_t, u } => {
            Ok(TpSpanFinisherPlan::Categorical { inv_t, u })
        }
        other => Err(format!("TP span finisher plan unsupported: {other:?}")),
    }
}

impl TpCoordinator {
    pub fn load(
        mut stream: TcpStream,
        resolved: &Resolved,
        model: &Path,
        pack: &Path,
        gpu: usize,
        max_ctx: usize,
        slots: usize,
        span_done: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, String> {
        if !resolved.is_coordinator() || max_ctx == 0 || !(1..=2).contains(&slots) {
            return Err(
                "TP=2 serving requires the rank-0 coordinator and a nonzero context".into(),
            );
        }
        stream
            .set_read_timeout(Some(READY_TIMEOUT))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(STEP_TIMEOUT))
            .map_err(|e| e.to_string())?;
        // Phase 12 fp8 KV gate, rank-0-authoritative like everything else
        // here: the executor exists before any wire traffic, so ask THIS
        // device the same question the TP=1 apply_kv_dtype path asks. A
        // below-sm_89 card demotes LOUDLY to f16 (the fp8 ask doubles the KV
        // pool; the serve must stay attributable), and the demoted value is
        // what TpInit carries - rank 1 never has to know the runner's env.
        let exec = Arc::new(GpuExecutor::new(gpu, pack).map_err(|e| e.to_string())?);
        let kv_dtype = kv_dtype_serve(exec.compute_capability());
        // Graph mode is rank-0-authoritative too (upstream-readiness I4):
        // the resolved value rides TpInit, so a hand-started remote worker
        // cannot disagree with the coordinator's env and mispair the
        // graphed/eager collective sequencing.
        let use_graphs = Qwen35TpRank::tp_graph_enabled_for_serve();
        let (checkpoint_sha256, pack_blake3) = hashes(model, pack)?;
        ControlMessage::TpInit {
            checkpoint_sha256,
            pack_blake3,
            max_ctx,
            slots,
            kv_dtype: kv_dtype_wire(kv_dtype).to_owned(),
            use_graphs,
        }
        .to_stream(&mut stream)
        .map_err(|e| e.to_string())?;
        // Rank 1 checks identity before NCCL is initialized.
        ready(&mut stream, 0)?;
        let id = create_unique_id().map_err(|e| e.to_string())?;
        send_nccl_id(&mut stream, &id).map_err(|e| e.to_string())?;
        let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)
            .map_err(|e| e.to_string())?
            .ok_or("TP group absent")?;
        let map = MappedGguf::open(model).map_err(|e| e.to_string())?;
        let mut model =
            Qwen35TpRank::load_slots(exec, &map, &group, max_ctx, kv_dtype, slots)
                .map_err(|e| e.to_string())?;
        // The lane's weights re-upload from the same mapped file, so the map
        // must outlive load.
        model
            .enable_prefill_lane(&group, &map)
            .map_err(|e| e.to_string())?;
        drop(map);
        // The worker enables graphs from the TpInit value (never its own env),
        // so both ranks run the identical graphed/eager sequencing.
        if use_graphs {
            model
                .enable_tp_graphs(&group)
                .map_err(|e| e.to_string())?;
        }
        ready(&mut stream, 1)?;
        stream
            .set_read_timeout(Some(STEP_TIMEOUT))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            group,
            model,
            logical: logical(max_ctx, slots)?,
            positions: vec![0; slots],
            occupied: vec![false; slots],
            sequence: 1,
            pipe: None,
            chunks: VecDeque::new(),
            span: None,
            span_done: Some(span_done),
            poisoned: None,
        })
    }

    /// Publish the prefill lane's GPU event status after every command. The
    /// span may still be in flight on the host while its queued GPU work has
    /// finished; the scheduler uses this flag to drain the decode pipe.
    fn publish_span_done(&mut self) {
        if let Some(flag) = self.span_done.as_ref() {
            flag.store(
                self.span.is_none() || self.model.prefill_lane_done(),
                std::sync::atomic::Ordering::Release,
            );
        }
    }

    fn exchange(&mut self, msg: ControlMessage) -> Result<(), String> {
        msg.to_stream(&mut self.stream).map_err(|e| e.to_string())?;
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)
    }

    fn reset_both(&mut self) -> Result<(), String> {
        let event = self
            .logical
            .authorize(Operation::Flush)
            .map_err(str::to_owned)?;
        let kv_event = serde_json::to_value(&event).map_err(|e| e.to_string())?;
        self.exchange(ControlMessage::TpReset {
            sequence: self.sequence + 1,
            kv_event,
        })?;
        self.model.reset().map_err(|e| e.to_string())?;
        // The prefill lane mirrors the flush: its KV/DeltaNet state must not
        // outlive the logical tables it was built from.
        self.model.reset_lane().map_err(|e| e.to_string())?;
        self.chunks.clear();
        self.span = None;
        self.positions.fill(0);
        self.occupied.fill(false);
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.run_rows(&[(0, token, self.positions[0])], 0)
    }

    fn release(&mut self, occupied: &[bool]) -> Result<(), String> {
        if occupied.len() != self.occupied.len() {
            return Err("TP occupancy width mismatch".into());
        }
        let slots: Vec<usize> = self
            .occupied
            .iter()
            .enumerate()
            .filter_map(|(i, was)| (*was && !occupied[i]).then_some(i))
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        // Fail closed before touching slot state: a release while the decode
        // pipe or a prefill span still owns the slot would race the in-flight
        // kernels against the reset below.
        if self.pipe.is_some() || self.span.is_some() || !self.model.prefill_lane_done() {
            return Err("TP release before GPU flight drained".into());
        }
        self.model.prefill_lane_join().map_err(|e| e.to_string())?;
        self.group
            .stream()
            .synchronize()
            .map_err(|e| e.to_string())?;
        self.model.synchronize().map_err(|e| e.to_string())?;
        let ops: Vec<Operation> = slots.iter().map(|&slot| Operation::Release { slot }).collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        self.exchange(ControlMessage::TpRelease {
            sequence: self.sequence + 1,
            slots: slots.clone(),
            kv_state,
        })?;
        for slot in slots {
            self.model.reset_slot(slot).map_err(|e| e.to_string())?;
            self.model
                .reset_lane_slot(slot)
                .map_err(|e| e.to_string())?;
            self.positions[slot] = 0;
            self.occupied[slot] = false;
        }
        Ok(())
    }

    fn batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, String> {
        let rows = active_rows(tokens, positions, self.positions.len())?;
        if rows.is_empty() {
            return Ok(vec![0.0; tokens.len() * self.model.vocab()]);
        }
        self.run_rows(&rows, tokens.len())
    }

    /// Greedy speculative verification using the scheduler-owned draft.
    /// Execute the proposed rows until the first rejection. Both TP ranks
    /// consume exactly the same accepted-prefix target sequence. The path is
    /// KV-dtype-agnostic (f16 / fp8_e4m3): rows flow through the same
    /// `forward_token_*` pipeline as decode, and every element size and slab
    /// stride derives from the load-time dtype carried in `TpInit`. Rejected
    /// draft rows are never executed, so no KV or position state advances
    /// past the verified prefix.
    fn spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<u32>, String> {
        if reqs.is_empty()
            || reqs.iter().any(|(slot, pos, chunk)| {
                *slot >= self.positions.len()
                    || *pos != self.positions[*slot]
                    || chunk.is_empty()
            })
        {
            return Err("TP speculative request shape or position invalid".into());
        }
        let mut picks = Vec::new();
        for &(slot, start, ref chunk) in reqs {
            let mut position = start;
            for (i, &token) in chunk.iter().enumerate() {
                let logits = self.run_rows(&[(slot, token, position)], 0)?;
                let next = argmax_logits(&logits)?;
                picks.push(next);
                position += 1;
                if i + 1 < chunk.len() && next != chunk[i + 1] {
                    picks.extend(std::iter::repeat_n(0, chunk.len() - i - 1));
                    break;
                }
            }
        }
        Ok(picks)
    }

    fn spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Vec<u32>, String> {
        let rows: Vec<(usize, u32, usize)> = reqs
            .iter()
            .flat_map(|&(slot, start, ref chunk)| {
                chunk
                    .iter()
                    .enumerate()
                    .map(move |(i, &token)| (slot, token, start + i))
            })
            .collect();
        if rows.is_empty()
            || rows.len() != plans.len()
            || rows.windows(2).any(|w| w[0].0 > w[1].0)
        {
            return Err("TP speculative sampled rows or plans invalid".into());
        }
        let mut expected_positions = self.positions.clone();
        for &(slot, _, position) in &rows {
            if slot >= expected_positions.len() || position != expected_positions[slot] {
                return Err("TP speculative sampled position disagrees with rank-0 plan".into());
            }
            expected_positions[slot] += 1;
        }
        let mut picks = Vec::with_capacity(rows.len());
        // One row is live per verify step, so one scratch plan vector reused
        // across the loop replaces a fresh dense allocation per row.
        let mut dense_plans = vec![crate::generator::RowSample::Hole; self.positions.len()];
        let mut plan_idx = 0;
        for (slot, start, chunk) in reqs.iter().cloned() {
            for (i, token) in chunk.iter().copied().enumerate() {
                dense_plans.fill(crate::generator::RowSample::Hole);
                dense_plans[slot] = crate::generator::RowSample::Device(plans[plan_idx]);
                let step = self.run_rows_impl(
                    &[(slot, token, start + i)],
                    self.positions.len(),
                    Some(&dense_plans),
                )?;
                let pick = step.ids[slot];
                plan_idx += 1;
                picks.push(pick);
                if i + 1 < chunk.len() && pick != chunk[i + 1] {
                    picks.extend(std::iter::repeat_n(0, chunk.len() - i - 1));
                    plan_idx += chunk.len() - i - 1;
                    break;
                }
            }
        }
        Ok(picks)
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if slot >= self.positions.len()
            || tokens.is_empty()
            || self.occupied[slot]
            || tokens.len() >= self.model.max_ctx()
        {
            return Err("TP prefill slot or prompt invalid".into());
        }
        let mut last = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            last = self.run_rows(&[(slot, token, position)], 0)?;
        }
        Ok(last)
    }

    // Validate the entire tick before authorizing any KV mutation or sending
    // control. Active pipe members keep executing as dummies after completion;
    // the scheduler ignores their Hole results until the segment drains.
    fn pipe_plans(
        slots: &[usize],
        width: usize,
        plans: &[RowSample],
    ) -> Result<Vec<crate::sampler::DevicePlan>, String> {
        if plans.len() != width || slots.is_empty() {
            return Err("TP pipe plan width mismatch".into());
        }
        let mut selected = Vec::with_capacity(slots.len());
        for (i, plan) in plans.iter().enumerate() {
            if slots.contains(&i) {
                let device = match plan {
                    RowSample::Device(p) => *p,
                    RowSample::Hole => crate::sampler::DevicePlan::Greedy,
                    RowSample::Host => return Err("TP pipe cannot read host logits".into()),
                };
                super::tp_model::tp_sample_params(device).map_err(|e| e.to_string())?;
                selected.push(device);
            } else if !matches!(plan, RowSample::Hole) {
                return Err("TP pipe plan disagrees with holes".into());
            }
        }
        Ok(selected)
    }

    fn pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), String> {
        if self.pipe.is_some() || !self.model.supports_device_sampling() {
            return Err("TP pipe unavailable or already in flight".into());
        }
        let rows = active_rows(tokens, positions, self.positions.len())?;
        let slots: Vec<_> = rows.iter().map(|row| row.0).collect();
        // Begin cannot contain a dead member; next ticks can carry dummy holes.
        validate_sampled_rows(&rows, plans, tokens.len())?;
        let devices = Self::pipe_plans(&slots, tokens.len(), plans)?;
        if rows.is_empty()
            || rows
                .iter()
                .any(|&(slot, _, pos)| pos != self.positions[slot] || pos >= self.model.max_ctx())
        {
            return Err("TP pipe begin position invalid".into());
        }
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeBegin {
            sequence: seq,
            rows: rows.clone(),
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        let mut events = Vec::with_capacity(rows.len());
        for (&(slot, token, position), plan) in rows.iter().zip(devices) {
            let event = self
                .model
                .forward_host_to_feedback(
                    &self.group,
                    &self.logical,
                    token,
                    position,
                    slot,
                    0,
                    plan,
                )
                .map_err(|e| e.to_string())?;
            events.push((slot, event));
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        self.sequence = seq;
        let n = slots.len();
        self.pipe = Some(PipeFlight {
            slots,
            width: tokens.len(),
            plane: 0,
            events,
            row_of: (0..n).collect(),
            mapped: false,
        });
        tracing::info!(sequence = seq, "TP decode pipe began");
        Ok(())
    }
    /// Slot-mapped pipe begin (overlap path): scheduler row i drives
    /// `slots[i]` - the churn-phase decode set, never contiguous. The wire
    /// stays canonical (TpPipeBegin rows ascending, the worker validates and
    /// stores them), so rows are sorted by slot and the flight keeps the
    /// wire->scheduler-row map for readback. Dead members ride as greedy
    /// dummies (Hole) with their ids discarded.
    fn pipe_begin_slots(
        &mut self,
        slots_v: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), String> {
        if self.pipe.is_some() || !self.model.supports_device_sampling() {
            return Err("TP pipe unavailable or already in flight".into());
        }
        if slots_v.is_empty()
            || slots_v.len() != tokens.len()
            || slots_v.len() != positions.len()
            || plans.len() != slots_v.len()
        {
            return Err("TP slot-mapped pipe width mismatch".into());
        }
        let slots: Vec<usize> = slots_v.iter().map(|&s| s as usize).collect();
        if slots.iter().any(|&s| s >= self.positions.len())
            || slots.windows(2).any(|w| w[0] == w[1])
        {
            return Err("TP slot-mapped pipe membership invalid".into());
        }
        let mut devices = Vec::with_capacity(slots.len());
        for plan in plans {
            match plan {
                RowSample::Device(p) => {
                    super::tp_model::tp_sample_params(*p).map_err(|e| e.to_string())?;
                    devices.push(*p);
                }
                RowSample::Hole => devices.push(crate::sampler::DevicePlan::Greedy),
                RowSample::Host => return Err("TP pipe cannot read host logits".into()),
            }
        }
        for (i, &slot) in slots.iter().enumerate() {
            let pos = positions[i] as usize;
            if pos != self.positions[slot] || pos >= self.model.max_ctx() {
                return Err("TP slot-mapped pipe position invalid".into());
            }
        }
        // Wire order: ascending by slot; remember each wire row's scheduler row.
        let mut order: Vec<usize> = (0..slots.len()).collect();
        order.sort_by_key(|&i| slots[i]);
        let wire_slots: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
        let wire_rows: Vec<(usize, u32, usize)> = order
            .iter()
            .map(|&i| (slots[i], tokens[i], positions[i] as usize))
            .collect();
        let wire_plans: Vec<crate::sampler::DevicePlan> =
            order.iter().map(|&i| devices[i]).collect();
        let ops: Vec<Operation> = wire_rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeBegin {
            sequence: seq,
            rows: wire_rows.clone(),
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        let mut events = Vec::with_capacity(wire_rows.len());
        for (&(slot, token, position), plan) in wire_rows.iter().zip(wire_plans) {
            let event = self
                .model
                .forward_host_to_feedback(&self.group, &self.logical, token, position, slot, 0, plan)
                .map_err(|e| e.to_string())?;
            events.push((slot, event));
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        self.sequence = seq;
        self.pipe = Some(PipeFlight {
            slots: wire_slots,
            width: slots.len(),
            plane: 0,
            events,
            row_of: order,
            mapped: true,
        });
        tracing::info!(sequence = seq, rows = slots.len(), "TP slot-mapped pipe began");
        Ok(())
    }

    fn pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, String> {
        let flight = self.pipe.as_ref().ok_or("TP pipe next without begin")?;
        let devices = if flight.mapped {
            // Slot-mapped rows: plan per SCHEDULER row, executed per WIRE row
            // via row_of; dead members ride as greedy dummies (their ids are
            // discarded by the scheduler).
            flight
                .row_of
                .iter()
                .map(|&row| match plans[row] {
                    RowSample::Device(plan) => {
                        super::tp_model::tp_sample_params(plan).map_err(|e| e.to_string())?;
                        Ok(plan)
                    }
                    RowSample::Hole => Ok(crate::sampler::DevicePlan::Greedy),
                    RowSample::Host => Err("TP pipe cannot read host logits".to_string()),
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            Self::pipe_plans(&flight.slots, flight.width, plans)?
        };
        let rows = flight
            .slots
            .iter()
            .map(|&slot| (slot, self.positions[slot]))
            .collect::<Vec<_>>();
        if rows.iter().any(|&(_, pos)| pos >= self.model.max_ctx()) {
            return Err("TP pipe reached context limit; drain before next tick".into());
        }
        let source_plane = flight.plane;
        let next_plane = source_plane ^ 1;
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeNext {
            sequence: seq,
            rows: rows.clone(),
            source_plane,
            next_plane,
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        let mut events = Vec::with_capacity(rows.len());
        for (&(slot, position), plan) in rows.iter().zip(devices) {
            let event = self
                .model
                .forward_feedback_to_feedback(
                    &self.group,
                    &self.logical,
                    slot,
                    position,
                    source_plane,
                    next_plane,
                    plan,
                )
                .map_err(|e| e.to_string())?
                .ok_or("rank 0 missing feedback event")?;
            events.push((slot, event));
            self.positions[slot] += 1;
        }
        // Rank 1 has enqueued the next tick before acknowledging the old one.
        ready(&mut self.stream, self.sequence)?;
        let ids = {
            let old = self.pipe.as_ref().ok_or("TP pipe disappeared")?;
            self.pipe_readback(old, source_plane)?
        };
        let old = self.pipe.as_mut().ok_or("TP pipe disappeared")?;
        old.events = events;
        old.plane = next_plane;
        self.sequence = seq;
        Ok(ids)
    }

    fn pipe_drain(&mut self) -> Result<Vec<u32>, String> {
        let flight = self.pipe.as_ref().ok_or("TP pipe drain without begin")?;
        let seq = self.sequence + 1;
        ControlMessage::TpPipeDrain { sequence: seq }
            .to_stream(&mut self.stream)
            .map_err(|e| e.to_string())?;
        ready(&mut self.stream, self.sequence)?;
        ready(&mut self.stream, seq)?;
        let ids = self.pipe_readback(flight, flight.plane)?;
        self.group
            .stream()
            .synchronize()
            .map_err(|e| e.to_string())?;
        self.model.synchronize().map_err(|e| e.to_string())?;
        self.sequence = seq;
        self.pipe = None;
        tracing::info!(sequence = seq, "TP decode pipe drained");
        Ok(ids)
    }

    /// Collect the in-flight tick's ids. Plain pipes keep the dense contract
    /// (`ids[slot]`, identity rows); slot-mapped pipes (overlap path, never
    /// contiguous) index by SCHEDULER row - the scheduler reads ids[i]
    /// against its row plan. Per-row plan gates which entries are meaningful.
    fn pipe_readback(&self, flight: &PipeFlight, plane: usize) -> Result<Vec<u32>, String> {
        let mut ids = vec![0; flight.width];
        for (wire, (slot, event)) in flight.events.iter().enumerate() {
            let id = self
                .model
                .feedback_id_after(event, *slot, plane)
                .map_err(|e| e.to_string())?;
            if flight.mapped {
                ids[flight.row_of[wire]] = id;
            } else {
                ids[*slot] = id;
            }
        }
        Ok(ids)
    }

    /// Abandon `slot`'s queued chunked prefill (client hung up). False when
    /// the slot's rows are inside the in-flight span ("not now") or its
    /// chunk is already popped for launch (in flight by definition); the
    /// scheduler retries next tick and drops the chunk at finish.
    fn chunk_abort(&mut self, slot: usize) -> bool {
        if self
            .span
            .as_ref()
            .is_some_and(|span| span.slots.contains(&slot))
        {
            return false;
        }
        let before = self.chunks.len();
        self.chunks.retain(|chunk| chunk.slot != slot);
        self.chunks.len() != before
    }

    /// One mixed tick: decode rows + the next chunk budget's rows in one
    /// weight pass. The two row groups' slots are disjoint by scheduler
    /// construction (chunking slots never ride `dec`), so the tick executes
    /// in SLOT order: both row sets merge into one ascending-slot sequence
    /// (the worker's `TpMixed` rule), each slot's rows keeping their relative
    /// order. Plans are ROW-position-indexed per service.rs:4606 and
    /// hole-free (dec is built compactly); a decode row samples on device
    /// (its id lands in `step.ids[dec_row]`) or returns host logits. A
    /// finishing chunk's LAST row is its finisher: with a supported
    /// `Device` plan it samples that row on device (`FinishSample::Sampled`,
    /// no readback); otherwise the row's full logits are returned
    /// (`FinishSample::Logits`, peeked uniforms stay uncommitted per the
    /// scheduler's rule).
    #[allow(clippy::type_complexity)]
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(
        crate::generator::SampledStep,
        Vec<(usize, crate::generator::FinishSample, usize)>,
    ), String> {
        // A span owns the lane; the scheduler only pumps decode-pipe ticks
        // over an in-flight span (or finishes it). Mixed vs span launch are
        // exclusive by scheduler design; guard it anyway.
        if self.span.is_some() {
            return Err("TP span in flight; finish or drain it first".into());
        }
        self.chunk_plans(fin_plans);
        let (chunk_rows, mut finishers) = self.chunk_take(budget);
        let dec_n = decodes.len();
        if plans.len() != dec_n {
            return Err("TP mixed plan width mismatch".into());
        }
        // service builds dec slot-ascending; enforce it so the slot-sorted
        // execution order matches the host_rows push order the scheduler
        // pops in dec-iteration order.
        if decodes.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err("TP mixed decode rows must be strictly ascending by slot".into());
        }
        if dec_n + !chunk_rows.is_empty() as usize == 0 {
            return Ok((
                crate::generator::SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                },
                Vec::new(),
            ));
        }
        for plan in plans {
            if matches!(plan, RowSample::Hole) {
                return Err("TP mixed tick cannot carry holes".into());
            }
            if let RowSample::Device(p) = plan {
                super::tp_model::tp_sample_params(*p).map_err(|e| e.to_string())?;
            }
        }
        // Validate the whole tick before any KV authorization.
        for &(slot, _, pos) in decodes {
            if slot >= self.positions.len()
                || pos as usize != self.positions[slot]
                || pos as usize >= self.model.max_ctx()
            {
                return Err("TP mixed decode position invalid".into());
            }
        }
        let mut cursor: Vec<usize> = self.positions.clone();
        for &(s, _, pos) in &chunk_rows {
            if s >= cursor.len() || pos != cursor[s] || pos >= self.model.max_ctx() {
                return Err("TP mixed chunk position invalid".into());
            }
            cursor[s] += 1;
        }
        if chunk_rows
            .iter()
            .any(|&(s, _, _)| decodes.iter().any(|&(d, _, _)| d == s))
        {
            return Err("TP mixed tick slot overlap between decode and chunk rows".into());
        }
        // Finisher plans: supported Device -> device-sampled first token;
        // Host or an unsupported plan -> full-logit readback.
        for fin in &mut finishers {
            fin.plan = fin_plans.iter().find_map(|(s, p)| match p {
                RowSample::Device(p)
                    if *s == fin.slot && super::tp_model::tp_sample_params(*p).is_ok() =>
                {
                    Some(*p)
                }
                _ => None,
            });
        }
        // Merge into one ascending-slot row sequence (stable: same-slot rows
        // keep their relative order; groups are slot-disjoint anyway).
        let mut rows: Vec<(usize, u32, usize)> =
            decodes.iter().map(|&(s, t, p)| (s, t, p as usize)).collect();
        rows.extend(chunk_rows.iter().copied());
        rows.sort_by_key(|&(slot, _, _)| slot);
        // Each finisher's last row (its chunk's final prefill row) produces
        // the finisher result; row index -> (finisher index, plan).
        let fin_last_row: std::collections::HashMap<usize, (usize, Option<crate::sampler::DevicePlan>)> =
            finishers
                .iter()
                .enumerate()
                .filter_map(|(fi, fin)| {
                    rows.iter()
                        .rposition(|&(s, _, _)| s == fin.slot)
                        .map(|ri| (ri, (fi, fin.plan)))
                })
                .collect();
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        ControlMessage::TpMixed {
            sequence: self.sequence + 1,
            rows: rows.clone(),
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, self.sequence + 1)?;
        let mut step = crate::generator::SampledStep {
            ids: vec![0; dec_n],
            host_rows: Vec::new(),
        };
        let mut fin_ids = vec![0u32; finishers.len()];
        let mut fin_logits: Vec<Option<Vec<f32>>> = vec![None; finishers.len()];
        for (ri, &(slot, token, position)) in rows.iter().enumerate() {
            if let Some(di) = decodes.iter().position(|&(s, _, _)| s == slot) {
                // Decode row: route by the ROW-position plan.
                match plans[di] {
                    RowSample::Device(plan) => {
                        step.ids[di] = self
                            .model
                            .forward_token_sampled_slot(
                                &self.group, &self.logical, token, position, slot, plan,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        let logits = self
                            .model
                            .forward_token_slot(&self.group, &self.logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        step.host_rows.push((slot, logits));
                    }
                }
            } else if let Some(&(fi, plan)) = fin_last_row.get(&ri) {
                // Finisher row: the chunk's final prefill forward.
                match plan {
                    Some(plan) => {
                        fin_ids[fi] = self
                            .model
                            .forward_token_sampled_slot(
                                &self.group, &self.logical, token, position, slot, plan,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    None => {
                        let logits = self
                            .model
                            .forward_token_slot(&self.group, &self.logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        fin_logits[fi] = Some(logits);
                    }
                }
            } else {
                // Interior chunk row: no readback, no sample.
                self.model
                    .forward_token_enqueue(&self.group, &self.logical, token, position, slot)
                    .map_err(|e| e.to_string())?;
            }
        }
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)?;
        for &(slot, _, _) in decodes {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        for &(slot, _, _) in &chunk_rows {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        let mut results = Vec::with_capacity(finishers.len());
        for (fi, fin) in finishers.iter().enumerate() {
            let sample = match fin.plan {
                Some(_) => crate::generator::FinishSample::Sampled(fin_ids[fi]),
                None => crate::generator::FinishSample::Logits(
                    fin_logits[fi].take().unwrap_or_default(),
                ),
            };
            results.push((fin.slot, sample, fin.rows));
        }
        Ok((step, results))
    }

    /// Queue a prompt for chunked prefill (scheduler-side admission). No
    /// worker message and no KV change: rows flow out through mixed ticks'
    /// `chunk_take` and span ticks' `span_take`.
    fn chunk_enqueue(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), String> {
        if slot >= self.positions.len() || tokens.is_empty() {
            return Err("TP chunked prefill slot or prompt invalid".into());
        }
        if self
            .chunks
            .iter()
            .map(|c| c.slot)
            .chain(
                self.span
                    .as_ref()
                    .map(|s| &s.finishing)
                    .into_iter()
                    .flatten()
                    .map(|f| f.slot),
            )
            .any(|s| s == slot)
        {
            return Err("TP slot already has a chunked prefill in flight".into());
        }
        if tokens.len() >= self.model.max_ctx() {
            return Err("TP prompt exceeds the context window".into());
        }
        self.chunks.push_back(PrefillChunk {
            slot,
            rows: tokens
                .into_iter()
                .enumerate()
                .map(|(pos, token)| (slot, token, pos))
                .collect(),
            fin_plan: None,
        });
        Ok(())
    }

    /// Fill `fin_plan` for queued chunks from the scheduler's finisher plans.
    fn chunk_plans(&mut self, fin_plans: &[(usize, RowSample)]) {
        for chunk in &mut self.chunks {
            if chunk.fin_plan.is_some() {
                continue;
            }
            for &(slot, plan) in fin_plans {
                if slot == chunk.slot {
                    if let RowSample::Device(p) = plan {
                        chunk.fin_plan = Some(p);
                    }
                }
            }
        }
    }

    /// Pop up to `budget` ordered rows from the queue front, recording every
    /// finishing chunk (its plan if any; `None` = host-logit readback). A
    /// chunk finishes when its last row is popped.
    fn chunk_take(
        &mut self,
        budget: usize,
    ) -> (Vec<(usize, u32, usize)>, Vec<SpanFinisher>) {
        let mut rows = Vec::new();
        let mut finishers = Vec::new();
        while rows.len() < budget {
            // Continue only within the chunk already open this tick.
            let go = match self.chunks.front() {
                Some(chunk) => {
                    rows.is_empty() || rows.last().is_some_and(|&(s, _, _)| s == chunk.slot)
                }
                None => false,
            };
            if !go {
                break;
            }
            let (slot, take, finishing, plan) = {
                let chunk = self
                    .chunks
                    .front()
                    .expect("chunk front is Some: the go guard just matched it");
                let take = chunk.rows.len().min(budget - rows.len());
                (chunk.slot, take, chunk.rows.len() == take, chunk.fin_plan)
            };
            let drained: Vec<_> = {
                let chunk = self
                    .chunks
                    .front_mut()
                    .expect("chunk front is Some: the go guard just matched it");
                chunk.rows.drain(..take).collect()
            };
            rows.extend(drained);
            if finishing {
                self.chunks.pop_front();
                finishers.push(SpanFinisher {
                    slot,
                    plan,
                    rows: take,
                });
            }
        }
        (rows, finishers)
    }

    /// Authorize and enqueue one span: validate the whole row plan, mirror
    /// each KV Ensure, send the launch with the finishers, wait for the
    /// worker's Prepared, then enqueue the rows and every finisher on the
    /// prefill lane without host-synchronizing. The caller pumps decode-pipe
    /// ticks while the lane's kernels run.
    fn span_begin(
        &mut self,
        rows: Vec<(usize, u32, usize)>,
        finishers: Vec<SpanFinisher>,
    ) -> Result<(), String> {
        if self.span.is_some() {
            return Err("TP span already in flight".into());
        }
        if self.pipe.is_some() {
            return Err("TP pipe must drain before a span launch".into());
        }
        if !self.model.has_prefill_lane() {
            return Err("TP prefill lane unavailable".into());
        }
        if rows.is_empty() {
            return Err("TP span launch has no rows".into());
        }
        // Chunks of one slot are contiguous in the queue, so the slice must
        // be too: equal slot => adjacent.
        if rows
            .windows(2)
            .any(|w| w[0].0 != w[1].0 && w[1].0 == rows[0].0)
        {
            return Err("TP span chunk rows must be contiguous per slot".into());
        }
        // Finishers: supported Device -> device-sampled first token; Host or
        // an unsupported plan -> full-logit readback. The wire carries EVERY
        // finishing chunk (plan or None): rank 1 promotes exactly these
        // slots' lane-local state at the span finish.
        let wire_finishers = finishers
            .iter()
            .map(|f| f.plan.map(wire_finisher_plan).transpose().map(|w| (f.slot, w)))
            .collect::<Result<Vec<_>, String>>()?;
        // Validate the entire tick BEFORE authorizing any KV mutation. Within
        // a chunk the position advances per enqueued row; across chunks the
        // next chunk of the same slot continues exactly at the running pos.
        let mut cursor: Vec<usize> = self.positions.clone();
        for &(s, _, pos) in &rows {
            if s >= cursor.len() || pos != cursor[s] || pos >= self.model.max_ctx() {
                return Err("TP span position disagrees with rank-0 plan".into());
            }
            cursor[s] += 1;
        }
        // Authorize the whole span's KV mutation first: failed validation must
        // not leave the ranks' logical tables diverged mid-span. The wire
        // carries the ordered rows plus ONE end-of-tick snapshot (v2 protocol,
        // B2) instead of one full-snapshot event per row, so long prefills
        // stay far below the 1 MiB frame cap.
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        ControlMessage::TpSpanLaunch {
            sequence: seq,
            rows: rows.clone(),
            finishers: wire_finishers,
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, seq)?;
        // Both ranks now hold the authorized plan. Enqueue the whole span on
        // the prefill lane, then each finishing chunk's finisher in chunk
        // order. `positions` advance as each row is enqueued so a later tick
        // sees the new state.
        for &(slot, token, pos) in &rows {
            self.model
                .prefill_lane_step(&self.group, &self.logical, slot, token, pos)
                .map_err(|e| e.to_string())?;
            self.positions[slot] += 1;
        }
        let mut events = VecDeque::with_capacity(finishers.len());
        for fin in &finishers {
            let event = self
                .model
                .prefill_lane_finisher(fin.slot, fin.plan)
                .map_err(|e| e.to_string())?;
            events.push_back(event);
        }
        // The worker sends exactly one tail Ready after enqueuing its rows.
        // Consume it before another command (pipe begin or span finish) can
        // expect its own ACK. This is an enqueue ACK, not a GPU fence.
        ready(&mut self.stream, seq)?;
        self.sequence = seq;
        tracing::info!(
            sequence = seq,
            rows = rows.len(),
            finishers = finishers.len(),
            "TP prefill span launched"
        );
        let mut span_slots: Vec<usize> = Vec::new();
        for &(s, _, _) in &rows {
            if !span_slots.contains(&s) {
                span_slots.push(s);
            }
        }
        self.span = Some(SpanFlight {
            finishing: finishers,
            events,
            slots: span_slots,
        });
        Ok(())
    }

    /// Pull the next span's rows from the chunk queue and launch it. Returns
    /// false when the queue is empty (nothing launched).
    fn span_take(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, String> {
        if self.chunks.is_empty() || self.span.is_some() || self.pipe.is_some() {
            return Ok(false);
        }
        self.chunk_plans(fin_plans);
        let (rows, finishers) = self.chunk_take(budget);
        if rows.is_empty() {
            return Ok(false);
        }
        self.span_begin(rows, finishers)?;
        Ok(true)
    }

    /// Fence the in-flight span: join the lane on both ranks, then read each
    /// finisher's result on rank 0, chunk order. Returns the scheduler's
    /// `(slot, FinishSample, rows)` contract directly.
    fn span_finish(
        &mut self,
    ) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, String> {
        let seq = self.sequence + 1;
        ControlMessage::TpSpanFinish { sequence: seq }
            .to_stream(&mut self.stream)
            .map_err(|e| e.to_string())?;
        ready(&mut self.stream, seq)?;
        let flight = self.span.take().ok_or("TP span finish without launch")?;
        self.group
            .stream()
            .synchronize()
            .map_err(|e| e.to_string())?;
        self.model.prefill_lane_join().map_err(|e| e.to_string())?;
        self.sequence = seq;
        tracing::info!(
            sequence = seq,
            finishers = flight.finishing.len(),
            "TP prefill span finished"
        );
        // Promote each finished slot's lane-local state lane->decode (locked
        // decision 1): the lane's KV slab and DeltaNet slot state are private
        // allocations, so a finished prompt's context would otherwise be
        // stranded. One lane-stream mark covers all slots (the join already
        // guarantees it fires); each promotion waits it device-side on the
        // decode stream. ~1-3 ms per finished prompt; the TTFT win survives.
        let mark = self.model.prefill_lane_mark().map_err(|e| e.to_string())?;
        for fin in &flight.finishing {
            let live = self
                .logical
                .slot_blocks(fin.slot)
                .ok_or("TP span finish slot out of range")?
                .to_vec();
            self.model
                .promote_lane_slot(&mark, fin.slot, &live)
                .map_err(|e| e.to_string())?;
        }
        let mut results = Vec::with_capacity(flight.finishing.len());
        for (fin, ev) in flight.finishing.into_iter().zip(flight.events) {
            match fin.plan {
                Some(_) => {
                    let id = self
                        .model
                        .prefill_lane_sampled_id_after(&ev, fin.slot)
                        .map_err(|e| e.to_string())?;
                    results.push((fin.slot, crate::generator::FinishSample::Sampled(id), fin.rows));
                }
                None => {
                    let logits = self
                        .model
                        .prefill_lane_logits_after(&ev)
                        .map_err(|e| e.to_string())?;
                    results.push((fin.slot, crate::generator::FinishSample::Logits(logits), fin.rows));
                }
            }
        }
        Ok(results)
    }

    fn sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, String> {
        if !self.model.supports_device_sampling() {
            return Err("TP device sampler unavailable".into());
        }
        let rows = active_rows(tokens, positions, self.positions.len())?;
        validate_sampled_rows(&rows, plans, tokens.len())?;
        if rows.is_empty() {
            return Ok(SampledStep {
                ids: vec![0; tokens.len()],
                host_rows: Vec::new(),
            });
        }
        self.run_rows_impl(&rows, tokens.len(), Some(plans))
    }

    fn run_rows(&mut self, rows: &[(usize, u32, usize)], width: usize) -> Result<Vec<f32>, String> {
        let result = self.run_rows_impl(rows, width, None)?;
        if width == 0 {
            return Ok(result
                .host_rows
                .into_iter()
                .next()
                .ok_or("TP missing serial logits")?
                .1);
        }
        let mut logits = vec![0.0; width * self.model.vocab()];
        for (slot, row) in result.host_rows {
            logits[slot * self.model.vocab()..(slot + 1) * self.model.vocab()]
                .copy_from_slice(&row);
        }
        Ok(logits)
    }

    fn run_rows_impl(
        &mut self,
        rows: &[(usize, u32, usize)],
        width: usize,
        plans: Option<&[RowSample]>,
    ) -> Result<SampledStep, String> {
        if rows.is_empty()
            || rows.len() > self.positions.len()
            || rows.windows(2).any(|w| w[0].0 >= w[1].0)
        {
            return Err("TP batch has no rows or unordered slots".into());
        }
        for &(slot, _, position) in rows {
            if slot >= self.positions.len()
                || position != self.positions[slot]
                || position >= self.model.max_ctx()
            {
                return Err("TP slot position disagrees with rank-0 plan".into());
            }
        }
        if let Some(plans) = plans {
            validate_sampled_rows(rows, plans, width)?;
        }
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        ControlMessage::TpBatch {
            sequence: self.sequence + 1,
            rows: rows.to_vec(),
            kv_state,
        }
        .to_stream(&mut self.stream)
        .map_err(|e| e.to_string())?;
        prepared(&mut self.stream, self.sequence + 1)?;
        let mut step = SampledStep {
            ids: vec![0; width],
            host_rows: Vec::new(),
        };
        for &(slot, token, position) in rows {
            match plans.map(|p| p[slot]).unwrap_or(RowSample::Host) {
                RowSample::Host => {
                    let row = self
                        .model
                        .forward_token_slot(&self.group, &self.logical, token, position, slot)
                        .map_err(|e| e.to_string())?;
                    step.host_rows.push((slot, row));
                }
                RowSample::Device(plan) => {
                    step.ids[slot] = self
                        .model
                        .forward_token_sampled_slot(
                            &self.group,
                            &self.logical,
                            token,
                            position,
                            slot,
                            plan,
                        )
                        .map_err(|e| e.to_string())?;
                }
                RowSample::Hole => return Err("TP live slot was planned as a hole".into()),
            }
        }
        self.sequence += 1;
        ready(&mut self.stream, self.sequence)?;
        for &(slot, _, _) in rows {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        Ok(step)
    }
}

enum Command {
    Reset,
    Step(u32),
    Prefill(usize, Vec<u32>),
    Batch(Vec<u32>, Vec<u32>),
    BatchSampled(Vec<u32>, Vec<u32>, Vec<RowSample>),
    Spec(Vec<(usize, usize, Vec<u32>)>),
    SpecPlans(Vec<(usize, usize, Vec<u32>)>, Vec<crate::sampler::DevicePlan>),
    PipeBegin(Vec<u32>, Vec<u32>, Vec<RowSample>),
    PipeNext(Vec<RowSample>),
    PipeDrain,
    Release(Vec<bool>),
    /// Chunked prefill admission: queue the prompt on the coordinator.
    ChunkBegin(usize, Vec<u32>),
    /// Abandon a slot's queued chunked prefill (client hangup).
    PrefillAbort(usize),
    /// Mixed tick: decode rows + chunked-prefill budget.
    Mixed(
        Vec<(usize, u32, u32)>,
        usize,
        Vec<RowSample>,
        Vec<(usize, RowSample)>,
    ),
    /// Span ticks (overlap path): take from the same queue as mixed ticks.
    SpanLaunch(usize, Vec<(usize, RowSample)>),
    SpanFinish,
    /// Slot-mapped decode pipe (overlap path).
    PipeBeginSlots(Vec<u32>, Vec<u32>, Vec<u32>, Vec<RowSample>),
}

enum Response {
    Logits(Vec<f32>),
    Sampled(SampledStep),
    Ids(Vec<u32>),
    Launched(bool),
    Aborted(bool),
    Mixed(
        SampledStep,
        Vec<(usize, crate::generator::FinishSample, usize)>,
    ),
    SpanFinished(Vec<(usize, crate::generator::FinishSample, usize)>),
}

/// Send-only proxy on the engine scheduler thread. The NCCL communicator and
/// CUDA context never leave their owning GPU thread (cudarc Comm is !Send).
pub struct TpGenerator {
    commands: std::sync::mpsc::Sender<(Command, std::sync::mpsc::Sender<Result<Response, String>>)>,
    vocab: usize,
    max_ctx: usize,
    slots: usize,
    device_sampling: bool,
    overlap: bool,
    /// Rank-local accounting captured at load (Phase 12): exact per-rank
    /// context bytes (GQA slabs + DeltaNet slot state), and this rank
    /// process's device-pool measurement at the same point.
    context_bytes: u64,
    process_bytes: Option<u64>,
    /// Shared with the coordinator thread: false while a prefill span is in
    /// flight (published after every command). `unified_span_done` polls it.
    span_done: Arc<std::sync::atomic::AtomicBool>,
    poisoned: Option<String>,
}

impl TpGenerator {
    pub fn load(
        stream: TcpStream,
        resolved: Resolved,
        path: &Path,
        pack: &Path,
        gpu: usize,
        max_ctx: usize,
        slots: usize,
    ) -> Result<Self, String> {
        let (commands, rx) = std::sync::mpsc::channel::<(
            Command,
            std::sync::mpsc::Sender<Result<Response, String>>,
        )>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let path = path.to_owned();
        let pack = pack.to_owned();
        let span_done = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let lane_flag = Arc::clone(&span_done);
        std::thread::Builder::new()
            .name("qwen35-tp-rank0".into())
            .spawn(move || {
                let mut coordinator = match TpCoordinator::load(
                    stream,
                    &resolved,
                    &path,
                    &pack,
                    gpu,
                    max_ctx,
                    slots,
                    lane_flag,
                ) {
                    Ok(coordinator) => coordinator,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if ready_tx
                    .send(Ok((
                        coordinator.model.vocab(),
                        coordinator.model.max_ctx(),
                        coordinator.model.supports_device_sampling(),
                        // Phase 12 rank-local accounting: measured at load.
                        // Context bytes are exact per-rank geometry (GQA
                        // slabs + DeltaNet slot state); weights bytes come
                        // from the executor's process-pool measurement with
                        // the context planes allocated but idle.
                        coordinator.model.context_mem_bytes(),
                        coordinator.model.process_mem_used_bytes(),
                    )))
                    .is_err()
                {
                    return;
                }
                for (cmd, reply) in rx {
                    let result = if coordinator.pipe.is_some()
                        && !matches!(cmd, Command::PipeNext(_) | Command::PipeDrain)
                    {
                        Err("TP pipe must drain before another command".into())
                    } else if coordinator.span.is_some()
                        && !matches!(
                            cmd,
                            Command::PipeBegin(..)
                                | Command::PipeBeginSlots(..)
                                | Command::PipeNext(_)
                                | Command::PipeDrain
                                | Command::SpanFinish
                                | Command::PrefillAbort(_)
                        )
                    {
                        // The span owns the lane: the scheduler may begin/
                        // pump the decode pipe over it (the overlap pump
                        // starts after the launch), finish it, or abort a
                        // QUEUED chunk (client hangup; in-flight span slots
                        // refuse inside chunk_abort). SpanFinish requires
                        // the pipe drained first - the worker refuses
                        // TpSpanFinish mid-pipe.
                        Err("TP span in flight; only pipe ticks or the span finish may run".into())
                    } else {
                        match cmd {
                            Command::Reset => coordinator
                                .reset_both()
                                .map(|_| Response::Logits(Vec::new())),
                            Command::Step(token) => coordinator.step(token).map(Response::Logits),
                            Command::Prefill(slot, tokens) => {
                                coordinator.prefill(slot, &tokens).map(Response::Logits)
                            }
                            Command::Batch(tokens, positions) => {
                                coordinator.batch(&tokens, &positions).map(Response::Logits)
                            }
                            Command::BatchSampled(tokens, positions, plans) => coordinator
                                .sampled(&tokens, &positions, &plans)
                                .map(Response::Sampled),
                            Command::Spec(reqs) => coordinator.spec_batch(&reqs).map(Response::Ids),
                            Command::SpecPlans(reqs, plans) => coordinator
                                .spec_batch_plans(&reqs, &plans)
                                .map(Response::Ids),
                            Command::PipeBegin(tokens, positions, plans) => coordinator
                                .pipe_begin(&tokens, &positions, &plans)
                                .map(|_| Response::Logits(Vec::new())),
                            Command::PipeBeginSlots(slots, tokens, positions, plans) => coordinator
                                .pipe_begin_slots(&slots, &tokens, &positions, &plans)
                                .map(|_| Response::Logits(Vec::new())),
                            Command::PipeNext(plans) => {
                                coordinator.pipe_next(&plans).map(Response::Ids)
                            }
                            Command::PipeDrain => coordinator.pipe_drain().map(Response::Ids),
                            Command::Release(occupied) => coordinator
                                .release(&occupied)
                                .map(|_| Response::Logits(Vec::new())),
                            Command::ChunkBegin(slot, tokens) => coordinator
                                .chunk_enqueue(slot, tokens)
                                .map(|_| Response::Logits(Vec::new())),
                            Command::PrefillAbort(slot) => {
                                Ok(Response::Aborted(coordinator.chunk_abort(slot)))
                            }
                            Command::Mixed(decodes, budget, plans, fin_plans) => coordinator
                                .forward_mixed(&decodes, budget, &plans, &fin_plans)
                                .map(|(step, finished)| Response::Mixed(step, finished)),
                            Command::SpanLaunch(budget, fin_plans) => coordinator
                                .span_take(budget, &fin_plans)
                                .map(Response::Launched),
                            Command::SpanFinish => {
                                coordinator.span_finish().map(Response::SpanFinished)
                            }
                        }
                    };
                    if let Err(e) = &result {
                        coordinator.poisoned = Some(e.clone());
                    }
                    coordinator.publish_span_done();
                    let failed = result.is_err();
                    let _ = reply.send(result);
                    if failed {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let (vocab, max_ctx, device_sampling, context_bytes, process_bytes) =
            ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            commands,
            vocab,
            max_ctx,
            slots,
            device_sampling,
            // The mapped decode pipe requires resident device sampling.
            overlap: device_sampling,
            context_bytes,
            process_bytes,
            span_done,
            poisoned: None,
        })
    }

    fn request(&mut self, command: Command) -> Result<Response, String> {
        if let Some(e) = &self.poisoned {
            return Err(e.clone());
        }
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let result = self
            .commands
            .send((command, reply_tx))
            .map_err(|e| e.to_string())
            .and_then(|_| reply_rx.recv().map_err(|e| e.to_string()))
            .and_then(|result| result);
        if let Err(ref e) = result {
            self.poisoned = Some(e.clone());
        }
        result
    }

    fn request_logits(&mut self, command: Command) -> Result<Vec<f32>, String> {
        match self.request(command)? {
            Response::Logits(logits) => Ok(logits),
            _ => Err("TP reply kind mismatch".into()),
        }
    }
}

impl Generator for TpGenerator {
    fn serial_only(&self) -> bool {
        self.slots == 1
    }
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        if max_batch != self.slots {
            return Err(GenError::Config(
                "TP batch width changed after rank bootstrap".into(),
            ));
        }
        Ok(self.slots)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        // The Generator trait has no error return here. Keep the proxy poisoned
        // and surface the failure; subsequent forwards must fail rather than
        // reuse a slot whose worker-side release was not acknowledged.
        if let Err(e) = self.request(Command::Release(occupied.to_vec())) {
            tracing::error!(error = %e, "TP slot release failed; rank pair poisoned");
        }
    }
    fn supports_device_sampling(&self) -> bool {
        self.device_sampling
    }
    fn spec_capable(&self) -> bool {
        true
    }
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        match self
            .request(Command::Spec(reqs.to_vec()))
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(Some(ids)),
            _ => Err(GenError::Backend("TP spec reply kind mismatch".into())),
        }
    }
    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        match self
            .request(Command::SpecPlans(reqs.to_vec(), plans.to_vec()))
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(Some(ids)),
            _ => Err(GenError::Backend("TP sampled spec reply kind mismatch".into())),
        }
    }
    fn supports_decode_pipe(&self) -> bool {
        self.device_sampling
    }
    fn decode_pipe_context_limit(&self) -> Option<usize> {
        Some(self.max_ctx)
    }
    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        self.request_logits(Command::PipeBegin(
            tokens.to_vec(),
            positions.to_vec(),
            plans.to_vec(),
        ))
        .map(|_| ())
        .map_err(GenError::Backend)
    }
    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        match self
            .request(Command::PipeNext(plans.to_vec()))
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(ids),
            _ => Err(GenError::Backend("TP pipe reply kind mismatch".into())),
        }
    }
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        match self
            .request(Command::PipeDrain)
            .map_err(GenError::Backend)?
        {
            Response::Ids(ids) => Ok(ids),
            _ => Err(GenError::Backend("TP pipe reply kind mismatch".into())),
        }
    }
    fn supports_overlap(&self) -> bool {
        self.overlap
    }
    fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        self.request_logits(Command::PipeBeginSlots(
            slots.to_vec(),
            tokens.to_vec(),
            positions.to_vec(),
            plans.to_vec(),
        ))
        .map(|_| ())
        .map_err(GenError::Backend)
    }
    fn unified_span_launch(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, GenError> {
        match self
            .request(Command::SpanLaunch(budget, fin_plans.to_vec()))
            .map_err(GenError::Backend)?
        {
            Response::Launched(launched) => Ok(launched),
            _ => Err(GenError::Backend("TP span reply kind mismatch".into())),
        }
    }
    fn unified_span_done(&self) -> bool {
        self.span_done
            .load(std::sync::atomic::Ordering::Acquire)
    }
    fn unified_span_finish(&mut self) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, GenError> {
        match self
            .request(Command::SpanFinish)
            .map_err(GenError::Backend)?
        {
            Response::SpanFinished(finished) => Ok(finished),
            _ => Err(GenError::Backend("TP span reply kind mismatch".into())),
        }
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.request_logits(Command::ChunkBegin(slot, tokens))
            .map(|_| ())
            .map_err(GenError::Backend)
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        match self.request(Command::PrefillAbort(slot)) {
            Ok(Response::Aborted(aborted)) => aborted,
            _ => {
                // A failed abort poisons the pair (request marks it); report
                // "not now" so the scheduler keeps the chunk and retries -
                // but the pair is gone, so make it a hard false.
                false
            }
        }
    }
    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, crate::generator::FinishSample, usize)>), GenError> {
        match self
            .request(Command::Mixed(
                decodes.to_vec(),
                budget,
                plans.to_vec(),
                fin_plans.to_vec(),
            ))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))?
        {
            Response::Mixed(step, finished) => Ok((step, finished)),
            _ => Err(GenError::Backend("TP mixed reply kind mismatch".into())),
        }
    }
    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Prefill(slot, tokens.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Batch(tokens.to_vec(), positions.to_vec()))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        match self
            .request(Command::BatchSampled(
                tokens.to_vec(),
                positions.to_vec(),
                plans.to_vec(),
            ))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))?
        {
            Response::Sampled(step) => Ok(step),
            _ => Err(GenError::Backend("TP sampled reply kind mismatch".into())),
        }
    }
    fn reset(&mut self) {
        if let Err(e) = self.request(Command::Reset) {
            tracing::error!(error = %e, "TP reset failed; rank pair poisoned");
        }
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.request_logits(Command::Step(token))
            .map_err(|e| GenError::Backend(format!("TP rank lost: {e}")))
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_context(&self) -> usize {
        self.max_ctx
    }
    /// Rank-local context bytes, exact (Phase 12): the GQA K/V slab pairs and
    /// DeltaNet recurrent/conv slot state THIS rank holds. Both ranks build
    /// identical geometry, so the number reads as "per rank" honestly.
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.context_bytes)
    }
    /// This rank's device-pool measurement at load, minus the exact context
    /// bytes - an upper bound on resident weights + scratch (the pool may
    /// hold transient frees; the TP=1 family reports the same way).
    fn weights_mem_bytes(&self) -> Option<u64> {
        self.process_bytes.map(|p| p.saturating_sub(self.context_bytes))
    }
    fn device_mem_used(&self) -> Option<u64> {
        self.process_bytes
    }
}

impl Drop for TpCoordinator {
    fn drop(&mut self) {
        let _ = shutdown_worker(&mut self.stream, self.poisoned.is_none());
    }
}

/// Runs in the rank-1 process, with no HTTP listener, tokenizer or sampler.
pub fn run_worker(
    mut stream: TcpStream,
    resolved: &Resolved,
    model_path: &Path,
    pack: &Path,
    // The worker's local GPU ordinal. The runner's worker paths always pass 0:
    // each rank process owns one GPU on its own node, and the coordinator's
    // --gpu selection applies to rank 0 only.
    gpu: usize,
) -> Result<(), String> {
    if !resolved.is_worker() {
        return Err("TP worker must have rank 1".into());
    }
    stream
        .set_read_timeout(Some(READY_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(STEP_TIMEOUT))
        .map_err(|e| e.to_string())?;
    let run = (|| -> Result<(), String> {
        let (max_ctx, slots, kv_dtype, use_graphs) =
            match ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())? {
                ControlMessage::TpInit {
                    checkpoint_sha256,
                    pack_blake3,
                    max_ctx,
                    slots,
                    kv_dtype,
                    use_graphs,
                } => {
                    let (own_checkpoint, own_pack) = hashes(model_path, pack)?;
                    // The dtype parses BEFORE the hash compare so an unknown
                    // wire value reads as what it is - a protocol mismatch -
                    // not as an identity failure.
                    let kv_dtype = kv_dtype_parse(&kv_dtype)?;
                    if own_checkpoint != checkpoint_sha256
                        || own_pack != pack_blake3
                        || max_ctx == 0
                        || !(1..=2).contains(&slots)
                    {
                        return Err(
                            "rank-1 checkpoint, CUDA pack or context disagrees with rank 0".into(),
                        );
                    }
                    (max_ctx, slots, kv_dtype, use_graphs)
                }
                ControlMessage::Shutdown { graceful: true } => return Ok(()),
                other => {
                    return Err(format!(
                        "TP worker expected the rank-0 TpInit handshake, got: {other:?}"
                    ));
                }
            };
        ControlMessage::TpReady { sequence: 0 }
            .to_stream(&mut stream)
            .map_err(|e| e.to_string())?;
        let id = receive_nccl_id(&mut stream).map_err(|e| e.to_string())?;
        let exec = Arc::new(GpuExecutor::new(gpu, pack).map_err(|e| e.to_string())?);
        let group = NcclCommunicator::from_resolved(Some(resolved), exec.stream.context(), id)
            .map_err(|e| e.to_string())?
            .ok_or("TP group absent")?;
        let map = MappedGguf::open(model_path).map_err(|e| e.to_string())?;
        let mut model =
            Qwen35TpRank::load_slots(exec.clone(), &map, &group, max_ctx, kv_dtype, slots)
                .map_err(|e| e.to_string())?;
        // The lane's weights re-upload from the same mapped file (same
        // map-lifetime reorder as the coordinator's load).
        model
            .enable_prefill_lane(&group, &map)
            .map_err(|e| e.to_string())?;
        drop(map);
        // Graph mode is rank-0-authoritative via TpInit (protocol v2): a
        // manually started worker must not resolve its own env value, or the
        // ranks could pair graphed and eager sequencing.
        if use_graphs {
            model
                .enable_tp_graphs(&group)
                .map_err(|e| e.to_string())?;
        }
        let mut logical = logical(max_ctx, slots)?;
        let mut positions = vec![0; slots];
        let mut pipe_slots: Option<Vec<usize>> = None;
        let mut span_in_flight = false;
        // Finishing chunks of the in-flight span (wire shape, chunk order):
        // rank 1 promotes exactly these slots at the span finish.
        let mut span_finishers: Vec<(usize, Option<TpSpanFinisherPlan>)> = Vec::new();
        let mut pipe_plane = 0usize;
        let mut pending_pipe: Option<u64> = None;
        let mut sequence = 1;
        ControlMessage::TpReady { sequence }
            .to_stream(&mut stream)
            .map_err(|e| e.to_string())?;
        // Idle serving may last indefinitely; only rank 0's per-step ACK
        // reads have a timeout. A closed control socket still ends this loop.
        stream.set_read_timeout(None).map_err(|e| e.to_string())?;
        loop {
            let msg = ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())?;
            let got = match &msg {
                ControlMessage::TpReset { sequence, .. }
                | ControlMessage::TpBatch { sequence, .. }
                | ControlMessage::TpPipeBegin { sequence, .. }
                | ControlMessage::TpPipeNext { sequence, .. }
                | ControlMessage::TpPipeDrain { sequence }
                | ControlMessage::TpRelease { sequence, .. }
                | ControlMessage::TpMixed { sequence, .. }
                | ControlMessage::TpSpanLaunch { sequence, .. }
                | ControlMessage::TpSpanFinish { sequence } => *sequence,
                ControlMessage::Shutdown { graceful: true }
                    if pipe_slots.is_none() && !span_in_flight =>
                {
                    return Ok(());
                }
                ControlMessage::Shutdown { graceful: true } => {
                    return Err("TP pipe or span must drain before shutdown".into());
                }
                ControlMessage::Shutdown { graceful: false } => return Err("rank 0 aborted".into()),
                other => return Err(format!("unexpected TP command: {other:?}")),
            };
            if got != sequence + 1 {
                return Err(format!(
                    "TP command sequence {got}, expected {}",
                    sequence + 1
                ));
            }
            if pipe_slots.is_some()
                && !matches!(
                    msg,
                    ControlMessage::TpPipeNext { .. } | ControlMessage::TpPipeDrain { .. }
                )
            {
                return Err("TP pipe must drain before another command".into());
            }
            if span_in_flight
                && !matches!(
                    msg,
                    ControlMessage::TpPipeBegin { .. }
                        | ControlMessage::TpPipeNext { .. }
                        | ControlMessage::TpPipeDrain { .. }
                        | ControlMessage::TpSpanFinish { .. }
                )
            {
                return Err(
                    "TP span in flight: only pipe ticks or the span finish may run".into(),
                );
            }
            match msg {
                ControlMessage::TpPipeBegin {
                    rows, kv_state, ..
                } => {
                    if rows.is_empty()
                        || rows.len() > slots
                        || rows.windows(2).any(|w| w[0].0 >= w[1].0)
                    {
                        return Err("invalid TP pipe begin membership".into());
                    }
                    for &(slot, _, position) in &rows {
                        if slot >= slots || position >= max_ctx || position != positions[slot] {
                            return Err("TP pipe begin position mismatch".into());
                        }
                    }
                    let ops: Vec<Operation> = rows
                        .iter()
                        .map(|&(slot, _, position)| Operation::Ensure { slot, position })
                        .collect();
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for &(slot, token, position) in &rows {
                        model
                            .forward_token_worker_slot(&group, &logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                    pipe_slots = Some(rows.iter().map(|r| r.0).collect());
                    pipe_plane = 0;
                    pending_pipe = Some(got);
                }
                ControlMessage::TpPipeNext {
                    rows,
                    source_plane,
                    next_plane,
                    kv_state,
                    ..
                } => {
                    let members = pipe_slots.as_ref().ok_or("TP pipe next without begin")?;
                    if rows.len() != members.len()
                        || source_plane != pipe_plane
                        || next_plane != (source_plane ^ 1)
                    {
                        return Err("TP pipe next shape or plane mismatch".into());
                    }
                    for (&(slot, position), &member) in rows.iter().zip(members) {
                        if slot != member || position >= max_ctx || position != positions[slot] {
                            return Err("TP pipe next position or membership mismatch".into());
                        }
                    }
                    let ops: Vec<Operation> = rows
                        .iter()
                        .map(|&(slot, position)| Operation::Ensure { slot, position })
                        .collect();
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, position) in rows {
                        model
                            .forward_feedback_to_feedback(
                                &group,
                                &logical,
                                slot,
                                position,
                                source_plane,
                                next_plane,
                                crate::sampler::DevicePlan::Greedy,
                            )
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                    let previous = pending_pipe
                        .replace(got)
                        .ok_or("TP pipe missing pending tick")?;
                    ControlMessage::TpReady { sequence: previous }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    pipe_plane = next_plane;
                }
                ControlMessage::TpPipeDrain { .. } => {
                    if pipe_slots.is_none() {
                        return Err("TP pipe drain without begin".into());
                    }
                    exec.synchronize().map_err(|e| e.to_string())?;
                    group.stream().synchronize().map_err(|e| e.to_string())?;
                    pipe_slots = None;
                    let previous = pending_pipe.take().ok_or("TP pipe missing pending tick")?;
                    ControlMessage::TpReady { sequence: previous }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                }
                ControlMessage::TpReset { kv_event, .. } => {
                    let event: Event =
                        serde_json::from_value(kv_event).map_err(|e| e.to_string())?;
                    if event.operation != Operation::Flush {
                        return Err("TP reset is not a flush".into());
                    }
                    logical.mirror(&event).map_err(str::to_owned)?;
                    model.reset().map_err(|e| e.to_string())?;
                    // The prefill lane mirrors the flush: its KV/DeltaNet
                    // state must not outlive the logical tables.
                    model.reset_lane().map_err(|e| e.to_string())?;
                    positions.fill(0);
                }
                ControlMessage::TpRelease {
                    slots: freed,
                    kv_state,
                    ..
                } => {
                    if freed.is_empty()
                        || freed.windows(2).any(|w| w[0] >= w[1])
                        || freed.iter().any(|&slot| slot >= slots)
                    {
                        return Err("invalid TP release membership".into());
                    }
                    // Fail closed before touching slot state, mirroring rank 0:
                    // a release while the decode pipe or a prefill span still
                    // owns the slot would race the in-flight kernels against
                    // the resets below.
                    if pipe_slots.is_some() || span_in_flight || !model.prefill_lane_done() {
                        return Err("TP worker release before GPU flight drained".into());
                    }
                    model.prefill_lane_join().map_err(|e| e.to_string())?;
                    group.stream().synchronize().map_err(|e| e.to_string())?;
                    model.synchronize().map_err(|e| e.to_string())?;
                    let ops: Vec<Operation> = freed
                        .iter()
                        .map(|&slot| Operation::Release { slot })
                        .collect();
                    logical.mirror_tick(&ops, &wire_kv_state(kv_state)?).map_err(str::to_owned)?;
                    for slot in freed {
                        model.reset_slot(slot).map_err(|e| e.to_string())?;
                        model.reset_lane_slot(slot).map_err(|e| e.to_string())?;
                        positions[slot] = 0;
                    }
                }
                ControlMessage::TpBatch {
                    rows, kv_state, ..
                } => {
                    if rows.is_empty()
                        || rows.len() > slots
                        || rows.windows(2).any(|w| w[0].0 >= w[1].0)
                    {
                        return Err("TP batch membership invalid".into());
                    }
                    for &(slot, _, position) in &rows {
                        if slot >= slots || position >= max_ctx || position != positions[slot] {
                            return Err("TP worker position diverged".into());
                        }
                    }
                    let ops: Vec<Operation> = rows
                        .iter()
                        .map(|&(slot, _, position)| Operation::Ensure { slot, position })
                        .collect();
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for &(slot, token, position) in &rows {
                        model
                            .forward_token_worker_slot(&group, &logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                }
                ControlMessage::TpMixed {
                    rows, kv_state, ..
                } => {
                    // Mixed tick (decode + chunk rows in one pass): rows may
                    // exceed the slot count - a chunk advances multiple rows
                    // of one slot per tick. Same execution contract as
                    // TpBatch: mirror KV, Prepared, then one eager forward
                    // per row in wire order (no logits, no sampling - rank 0
                    // owns both).
                    if rows.is_empty() {
                        return Err("TP mixed tick membership invalid".into());
                    }
                    validate_multistep_positions(&rows, &positions, max_ctx)?;
                    let ops: Vec<Operation> = rows
                        .iter()
                        .map(|&(slot, _, position)| Operation::Ensure { slot, position })
                        .collect();
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, token, position) in rows {
                        model
                            .forward_token_worker_slot(&group, &logical, token, position, slot)
                            .map_err(|e| e.to_string())?;
                        positions[slot] += 1;
                    }
                }
                ControlMessage::TpSpanLaunch {
                    rows,
                    finishers,
                    kv_state,
                    ..
                } => {
                    // The whole prompt span enqueues on rank 1's prefill lane:
                    // the lane steps enter the same collectives in the same
                    // order as rank 0's lane. No finisher enqueue here - the
                    // finisher sampler has no collectives and runs only on
                    // rank 0; the finisher list exists so this rank promotes
                    // the same slots at the span finish.
                    if rows.is_empty() {
                        return Err("TP span launch membership invalid".into());
                    }
                    validate_multistep_positions(&rows, &positions, max_ctx)?;
                    let ops: Vec<Operation> = rows
                        .iter()
                        .map(|&(slot, _, position)| Operation::Ensure { slot, position })
                        .collect();
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, token, position) in &rows {
                        model
                            .prefill_lane_step(&group, &logical, *slot, *token, *position)
                            .map_err(|e| e.to_string())?;
                        positions[*slot] += 1;
                    }
                    // Rank 1 does not sample; its lane finisher marks span
                    // completion for the promotion at the span finish.
                    model
                        .prefill_lane_finisher(
                            rows.last()
                                .expect("span launch rows are non-empty: checked above")
                                .0,
                            None,
                        )
                        .map_err(|e| e.to_string())?;
                    span_in_flight = true;
                    span_finishers = finishers;
                }
                ControlMessage::TpSpanFinish { .. } => {
                    if !span_in_flight {
                        return Err("TP span finish without launch".into());
                    }
                    if pipe_slots.is_some() {
                        return Err("TP span finish with pipe in flight".into());
                    }
                    // Join both streams (collectives + lane), then promote
                    // each finished slot's lane-local state onto this rank's
                    // decode executor - same chunk order as rank 0. The lane
                    // slab and DeltaNet slot state are private allocations on
                    // every rank, so rank 1's decode executor would otherwise
                    // never see the prompt's context.
                    exec.synchronize().map_err(|e| e.to_string())?;
                    group.stream().synchronize().map_err(|e| e.to_string())?;
                    model.prefill_lane_join().map_err(|e| e.to_string())?;
                    let mark = model.prefill_lane_mark().map_err(|e| e.to_string())?;
                    for (slot, _) in &span_finishers {
                        let live = logical
                            .slot_blocks(*slot)
                            .ok_or("TP span finish slot out of range")?
                            .to_vec();
                        model
                            .promote_lane_slot(&mark, *slot, &live)
                            .map_err(|e| e.to_string())?;
                    }
                    span_in_flight = false;
                    span_finishers.clear();
                }
                _ => unreachable!("message shape checked above"),
            }
            sequence = got;
            if pending_pipe.is_none() {
                ControlMessage::TpReady { sequence }
                    .to_stream(&mut stream)
                    .map_err(|e| e.to_string())?;
            }
        }
    })();
    if let Err(ref e) = run {
        let _ = ControlMessage::TpError { reason: e.clone() }.to_stream(&mut stream);
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn sockets() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let worker = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (head, _) = listener.accept().unwrap();
        (head, worker)
    }

    #[test]
    fn kv_dtype_from_env_defaults_to_f16_and_demotes_below_sm89_loudly() {
        // Unset/auto/f16 serve f16 regardless of device.
        for env in [None, Some("auto"), Some("f16"), Some("")] {
            assert_eq!(
                kv_dtype_from_env(env, (12, 0)),
                KvDtype::Fp16,
                "env {env:?} must serve f16"
            );
        }
        // Spark sm_121 and consumer sm_89 both honor the fp8 ask: today
        // gpu_support::fp8_kv answers yes on every die this build serves
        // (fp8 storage is software-emulated and byte-exact), so the device
        // gate never fires. The seam stays for a future broken-e4m3 die.
        assert_eq!(
            kv_dtype_from_env(Some("fp8_e4m3"), (12, 1)),
            KvDtype::Fp8E4m3
        );
        assert_eq!(
            kv_dtype_from_env(Some("fp8_e4m3"), (8, 9)),
            KvDtype::Fp8E4m3
        );
        // Unknown values never silently pick a dtype.
        assert_eq!(kv_dtype_from_env(Some("int8"), (12, 1)), KvDtype::Fp16);
    }

    #[test]
    fn kv_dtype_wire_roundtrip_covers_both_dtypes() {
        for dtype in [KvDtype::Fp16, KvDtype::Fp8E4m3] {
            let wire = kv_dtype_wire(dtype);
            assert_eq!(
                kv_dtype_parse(wire).unwrap(),
                dtype,
                "wire {wire:?} must round-trip {dtype:?}"
            );
        }
        assert!(kv_dtype_parse("int8").is_err());
        assert!(kv_dtype_parse("").is_err());
    }

    #[test]
    fn dense_scheduler_rows_preserve_slot_ids_and_skip_holes() {
        assert_eq!(active_rows(&[9, 8], &[0, 12], 2).unwrap(), vec![(1, 8, 12)]);
        assert_eq!(
            active_rows(&[9, 8], &[3, 12], 2).unwrap(),
            vec![(0, 9, 3), (1, 8, 12)]
        );
        assert!(active_rows(&[0, 0], &[0, 0], 2).unwrap().is_empty());
        assert!(active_rows(&[1], &[1, 2], 2).is_err());
        assert!(active_rows(&[1, 2, 3], &[1, 2, 3], 2).is_err());
    }

    #[test]
    fn sampled_dense_rows_keep_holes_and_host_fallback() {
        use crate::sampler::DevicePlan;
        let rows = active_rows(&[11, 22], &[0, 9], 2).unwrap();
        assert!(
            validate_sampled_rows(
                &rows,
                &[RowSample::Hole, RowSample::Device(DevicePlan::Greedy)],
                2
            )
            .is_ok()
        );
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole, RowSample::Host], 2).is_ok());
        assert!(validate_sampled_rows(&rows, &[RowSample::Host, RowSample::Host], 2).is_err());
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole, RowSample::Hole], 2).is_err());
        assert!(validate_sampled_rows(&rows, &[RowSample::Hole], 2).is_err());
        assert!(
            validate_sampled_rows(
                &[(0, 11, 3), (1, 22, 9)],
                &[
                    RowSample::Host,
                    RowSample::Device(DevicePlan::Categorical { inv_t: 2.0, u: 0.5 })
                ],
                2
            )
            .is_ok()
        );
        assert!(
            validate_sampled_rows(
                &rows,
                &[
                    RowSample::Hole,
                    RowSample::Device(DevicePlan::TruncCat {
                        inv_t: 1.0,
                        u: 0.5,
                        k: 10,
                        top_p: 0.9,
                        min_p: 0.0
                    })
                ],
                2
            )
            .is_err()
        );
    }

    #[test]
    fn pipe_plan_preflight_handles_holes_and_rejects_host() {
        use crate::sampler::DevicePlan;
        let d = RowSample::Device(DevicePlan::Greedy);
        assert_eq!(
            TpCoordinator::pipe_plans(&[1], 2, &[RowSample::Hole, d])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            TpCoordinator::pipe_plans(&[0, 1], 2, &[RowSample::Hole, d])
                .unwrap()
                .len(),
            2
        );
        assert!(TpCoordinator::pipe_plans(&[1], 2, &[d, d]).is_err());
        assert!(TpCoordinator::pipe_plans(&[0], 2, &[RowSample::Host, RowSample::Hole]).is_err());
        assert!(TpCoordinator::pipe_plans(&[0], 2, &[d]).is_err());
        assert!(TpCoordinator::pipe_plans(&[], 2, &[RowSample::Hole, RowSample::Hole]).is_err());
    }

    #[test]
    fn pipe_acks_defer_old_ready_until_next_is_prepared() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpPrepared { sequence: 2 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 2).unwrap();
        ControlMessage::TpPrepared { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ControlMessage::TpReady { sequence: 2 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 3).unwrap();
        ready(&mut head, 2).unwrap();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ControlMessage::TpReady { sequence: 4 }
            .to_stream(&mut worker)
            .unwrap();
        ready(&mut head, 3).unwrap();
        ready(&mut head, 4).unwrap();
    }

    #[test]
    fn mirrored_two_slot_release_and_reuse_keeps_survivor() {
        let mut head = logical(32, 2).unwrap();
        let mut worker = logical(32, 2).unwrap();
        for (slot, position) in [(0, 0), (1, 0), (0, 16), (1, 16)] {
            let event = head
                .authorize(Operation::Ensure { slot, position })
                .unwrap();
            worker.mirror(&event).unwrap();
        }
        let survivor = head.snapshot().tables[1].clone();
        let event = head.authorize(Operation::Release { slot: 0 }).unwrap();
        worker.mirror(&event).unwrap();
        let event = head
            .authorize(Operation::Ensure {
                slot: 0,
                position: 0,
            })
            .unwrap();
        worker.mirror(&event).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert_eq!(head.snapshot().tables[1], survivor);
        let event = head.authorize(Operation::Flush).unwrap();
        worker.mirror(&event).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert!(head.snapshot().tables.iter().all(Vec::is_empty));
    }

    #[test]
    fn multirow_positions_validate_before_prepared() {
        let start = [0, 7];
        assert!(validate_multistep_positions(&[(1, 10, 7), (1, 11, 8)], &start, 32).is_ok());
        assert!(validate_multistep_positions(&[(1, 10, 7), (1, 11, 7)], &start, 32).is_err());
        assert!(validate_multistep_positions(&[(1, 10, 7), (1, 11, 9)], &start, 32).is_err());
        assert!(validate_multistep_positions(&[(2, 10, 0)], &start, 32).is_err());
        assert!(validate_multistep_positions(&[(1, 10, 32)], &start, 32).is_err());
    }

    #[test]
    fn step_requires_prepared_before_completed_ack() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        assert!(
            prepared(&mut head, 3)
                .unwrap_err()
                .contains("did not prepare")
        );
        ControlMessage::TpPrepared { sequence: 4 }
            .to_stream(&mut worker)
            .unwrap();
        assert!(
            prepared(&mut head, 3)
                .unwrap_err()
                .contains("did not prepare")
        );
        ControlMessage::TpPrepared { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        prepared(&mut head, 3).unwrap();
        ControlMessage::TpReady { sequence: 3 }
            .to_stream(&mut worker)
            .unwrap();
        ready(&mut head, 3).unwrap();
    }

    #[test]
    fn worker_error_and_closed_connection_fail_closed() {
        let (mut head, mut worker) = sockets();
        ControlMessage::TpError {
            reason: "bad KV event".into(),
        }
        .to_stream(&mut worker)
        .unwrap();
        assert!(prepared(&mut head, 7).unwrap_err().contains("bad KV event"));
        drop(worker);
        assert!(ready(&mut head, 7).is_err());
    }
}
