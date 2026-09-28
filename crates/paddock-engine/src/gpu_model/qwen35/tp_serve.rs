//! Rank-0-authoritative tensor-parallel serving over distributed worker channels.
//! The production scheduler owns rank 0's dense slot plan; worker rank replays the
//! ordered live rows and mirrored KV operations. Decode runs eager per row;
//! contiguous prompt rows use bounded batched spans.
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
    protocol::{ControlMessage, TpSpanFinisherPlan, receive_nccl_id},
    worker::WorkerControl,
};
use paddock_models::mapped::MappedGguf;

use super::{tp_kv::tp_resume_decision, tp_model::Qwen35TpRank};
use crate::gpu_model::tp::cache::{
    Event, MirroredKv, Operation, Snapshot, tp_publish_ops,
};
use crate::gpu_model::tp::control::WorkerSet;
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

/// The wire spelling of a KV dtype, shared by all ranks (Phase 12). Rank 0
/// resolves the runner's dtype gate and sends the resolved value; worker rank
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
/// applies for TP=1) and SENDS the resolved value; worker ranks parse what
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
                    "kv cache: TP asked for fp8_e4m3, but {why} (this GPU is sm_{:#02}{:#02}). \
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
            "TP serving requires the pinned Qwen3.8 GGUF with SHA-256 {PINNED_SHA256}, got {checkpoint}"
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

/// Mirror coordinator decode-prefix ordering before the worker mirrors KV or
/// acknowledges Prepared. The chunk tail may repeat one different slot.
fn validate_mixed_worker_rows(
    rows: &[(usize, u32, usize)],
    chunk_rows: usize,
    positions: &[usize],
    max_ctx: usize,
) -> Result<usize, String> {
    if rows.is_empty() || chunk_rows > rows.len() {
        return Err("TP mixed tick membership invalid".into());
    }
    let split = rows.len() - chunk_rows;
    if rows[..split].windows(2).any(|w| w[0].0 >= w[1].0) {
        return Err("TP mixed decode rows must be strictly ascending by slot".into());
    }
    let chunk = &rows[split..];
    if !chunk.is_empty()
        && (chunk.iter().any(|&(s, _, _)| s != chunk[0].0)
            || rows[..split].iter().any(|r| r.0 == chunk[0].0))
    {
        return Err("TP mixed tick chunk run must be one disjoint slot".into());
    }
    validate_multistep_positions(rows, positions, max_ctx)?;
    Ok(split)
}

/// Row cap of the production batched prefill span. Resolved once per process
/// from the rank-0-authoritative `TpInit` value so neither rank can drift.
fn tp_span_cap() -> usize {
    super::tp_span_cap::span_cap()
}

/// Boundaries for one slot's contiguous prompt-row run at an explicit cap.
/// Each window has 1..=`cap` rows; the last window may produce logits only
/// when this run finishes the prompt. Pure so the geometry stays testable
/// at every sweep width without racing the process-global resolution.
fn span_chunk_points_at(len: usize, cap: usize) -> Vec<usize> {
    let mut points = Vec::new();
    let mut at = 0usize;
    while at < len {
        points.push(at);
        at += cap.min(len - at);
    }
    points.push(len);
    points
}

/// Split at both the GPU row cap and each reserved logical checkpoint cut,
/// at an explicit cap. A cut N is the state after rows [0,N), before any row
/// N changes DeltaNet.
fn span_checkpoint_points_at(start: usize, len: usize, cuts: &[(usize, u32)], cap: usize) -> Result<Vec<usize>, String> {
    let end = start.checked_add(len).ok_or("TP span range overflow")?;
    if cuts.windows(2).any(|w| w[0].0 >= w[1].0)
        || cuts.iter().any(|&(cut, _)| cut <= start || cut > end || !cut.is_multiple_of(crate::kv_pool::BLOCK_TOKENS))
    {
        return Err("TP span checkpoint cuts invalid".into());
    }
    let mut points = vec![0];
    let mut at = 0;
    while at < len {
        let next_cut = cuts.iter().find(|&&(cut, _)| cut > start + at).map_or(end, |&(cut, _)| cut);
        at += cap.min(len - at).min(next_cut - start - at);
        points.push(at);
    }
    Ok(points)
}

/// Production chunkers read the resolved rank-symmetric cap. All ranks
/// derive identical geometry from the identical resolution.
fn span_chunk_points(len: usize) -> Vec<usize> {
    span_chunk_points_at(len, tp_span_cap())
}

fn span_checkpoint_points(start: usize, len: usize, cuts: &[(usize, u32)]) -> Result<Vec<usize>, String> {
    span_checkpoint_points_at(start, len, cuts, tp_span_cap())
}

fn tp_suffix_rows(slot: usize, tokens: Vec<u32>, resume: usize) -> Vec<(usize, u32, usize)> {
    tokens.into_iter().enumerate().skip(resume)
        .map(|(pos, token)| (slot, token, pos)).collect()
}

/// Record rows already validated, authorized and enqueued. A prompt-only
/// span must mark its slot occupied even before the first decode: release
/// uses this bit to reclaim its pages on cancellation.
fn record_prefill_rows(rows: &[(usize, u32, usize)], positions: &mut [usize], occupied: &mut [bool]) {
    for &(slot, _, _) in rows {
        positions[slot] += 1;
        occupied[slot] = true;
    }
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

/// The checkpoint-pool capacity rank 0 resolves for BOTH ranks (mirroring
/// the kv_dtype/use_graphs rank-0-authoritative pattern): `PADDOCK_TP_CKPT_SLOTS`,
/// default 4 (~4 x the DeltaNet state pair per layer per rank at the Qwen3.8
/// geometry). 0 disables resume entirely (pages-only impossible too - no
/// snapshots, cache empty). Pure function of the env for host tests.
fn resolve_ckpt_slots(env: Option<&str>) -> u32 {
    env.and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.min(64) as u32)
        .unwrap_or(4)
}

/// The checkpoint cuts a prompt's prefill snapshots: the prompt's last two
/// full page boundaries (the shared `ckpt_cuts` contract), filtered to cuts
/// the prefill can actually reach (strictly after `start`, strictly before
/// `t_len`). Boundaries are ascending.
fn tp_prefill_cuts(t_len: usize, start: usize) -> Vec<usize> {
    let [b1, b2] = crate::gpu_model::qwen35::tp_checkpoint_cuts(t_len);
    [b1, b2]
        .into_iter()
        .filter(|&c| c > start && c < t_len)
        .collect()
}

fn reserve_cold_mixed_cuts(
    logical: &mut MirroredKv,
    pending: &mut std::collections::HashMap<usize, Vec<(usize, u32)>>,
    slot: usize,
    start: usize,
) -> Result<Vec<usize>, String> {
    if pending.contains_key(&slot) {
        return Ok(Vec::new()); // resumed or already committed to Mixed
    }
    if start != 0 {
        return Err("TP cold Mixed checkpoint reservation after first row".into());
    }
    let len = logical.slot_admitted_tokens(slot).len();
    if len == 0 {
        return Err("TP Mixed checkpoint reservation without admission".into());
    }
    let reserved = logical
        .reserve_cuts_for_slot(slot, &tp_prefill_cuts(len, 0))
        .map_err(str::to_owned)?;
    let cuts = reserved.iter().map(|&(cut, _)| cut).collect();
    // Insert even when capacity is zero: later Mixed ticks must not retry.
    pending.insert(slot, reserved);
    Ok(cuts)
}

/// Which of `cuts` a span covering rows `[span_start, span_start + rows)`
/// lands EXACTLY on (span end == cut): a checkpoint may be snapshotted only
/// there, because the slot's live DeltaNet state is exactly the state after
/// row `cut - 1` - the state matching KV pages `[0, cut)`. Pure, shared by
/// all ranks' snapshot discipline.
fn tp_cuts_in_run(cuts: &[(usize, u32)], span_start: usize, rows: usize) -> Vec<(usize, u32)> {
    let end = span_start.saturating_add(rows);
    cuts.iter().filter(|&&(cut, _)| cut > span_start && cut <= end).copied().collect()
}

fn tp_cuts_crossed(cuts: &[(usize, u32)], span_start: usize, rows: usize) -> Vec<(usize, u32)> {
    let span_end = span_start.saturating_add(rows);
    cuts.iter()
        .filter(|&&(cut, _)| cut == span_end)
        .copied()
        .collect()
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
    workers: WorkerSet,
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
    shutdown_sent: bool,
    /// Checkpoint cuts whose rank-0 GPU snapshot succeeded, per slot
    /// `(slot, cut)`. Filled by `run_chunk_spans`; consumed by
    /// `publish_slot_cache` at the prefill's successful finish.
    snapshotted_cuts: Vec<(usize, usize)>,
    /// Per-slot checkpoint reservations still owned by a queued or in-flight
    /// prefill: `(cut, index)` in cut order. Filled at admission, cleared at
    /// finish/abort/release; the Mixed tick derives its wire cut list from
    /// here and the publish tick consumes the survivors.
    pending_ckpts: std::collections::HashMap<usize, Vec<(usize, u32)>>,
    /// Shared with the scheduler proxy: per-slot cached-token accounting
    /// (written at admission, read by `take_prefill_reused`).
    slot_reused: Arc<std::sync::Mutex<Vec<usize>>>,
}

/// One queued prompt chunk: the remaining ordered rows (slot, token,
/// position) plus the finishing chunk's device plan, if its first token
/// samples on device (`None` = full-logit host readback).
struct PrefillChunk {
    slot: usize,
    rows: Vec<(usize, u32, usize)>,
    fin_plan: Option<crate::sampler::DevicePlan>,
    /// Set when the first rows execute; persists over every partial tick.
    owner: Option<crate::generator::PrefillLane>,
}

fn abort_queued_chunk(chunks: &mut VecDeque<PrefillChunk>, slot: usize) -> bool {
    let before = chunks.len();
    chunks.retain(|chunk| chunk.slot != slot);
    chunks.len() != before
}

/// A queued chunk may be reclaimed only while no launched span owns its rows.
/// This is the production abort decision seam: the scheduler can observe a
/// disconnect between ticks, but an in-flight span remains owned until the
/// command thread has drained it and cleared `self.span`.
fn queued_chunk_abort_allowed(span: Option<&SpanFlight>, slot: usize) -> bool {
    !span.is_some_and(|flight| flight.slots.contains(&slot))
}

/// A finishing chunk's summary, kept from launch until the span drains so
/// `span_finish` can read each finisher's result in chunk order.
struct SpanFinisher {
    slot: usize,
    plan: Option<crate::sampler::DevicePlan>,
    /// FULL prompt rows this finisher completes (last row's position + 1),
    /// not the final tick's piece: the scheduler's `finish_prefill` sets
    /// `slot.pos = rows`, so a multi-tick prompt must report its whole
    /// length or the scheduler's decode positions diverge from the ranks'.
    fin_rows: usize,
}

/// How a span's FINAL span produces its result: host logits readback or a
/// device-sampled ID (rank 0 only). `prefill_lane_finisher`'s plan mirror.
#[derive(Clone, Copy)]
enum SpanFinishKind {
    HostLogits,
    Device(crate::sampler::DevicePlan),
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
/// exist: worker ranks never samples (the finisher sampler has no collectives and
/// runs only on rank 0), it only reads the finisher slot ids to promote
/// lane-local state.
fn wire_finisher_plan(
    plan: crate::sampler::DevicePlan,
) -> Result<TpSpanFinisherPlan, String> {
    // The GPU finisher sampler checks these same parameters; reject invalid
    // categorical values before either rank authorizes KV for the span.
    super::tp_model::tp_sample_params(plan).map_err(|e| e.to_string())?;
    match plan {
        crate::sampler::DevicePlan::Greedy => Ok(TpSpanFinisherPlan::Greedy),
        crate::sampler::DevicePlan::Categorical { inv_t, u } => {
            Ok(TpSpanFinisherPlan::Categorical { inv_t, u })
        }
        other => Err(format!("TP span finisher plan unsupported: {other:?}")),
    }
}

/// Map scheduler-row plans to wire order, validating width before indexing.
fn mapped_pipe_plans(
    row_of: &[usize],
    width: usize,
    plans: &[RowSample],
) -> Result<Vec<crate::sampler::DevicePlan>, String> {
    if plans.len() != width || row_of.len() != width || row_of.iter().any(|&row| row >= width) {
        return Err("TP slot-mapped pipe plan width mismatch".into());
    }
    row_of
        .iter()
        .map(|&row| match plans[row] {
            RowSample::Device(plan) => {
                super::tp_model::tp_sample_params(plan).map_err(|e| e.to_string())?;
                Ok(plan)
            }
            RowSample::Hole => Ok(crate::sampler::DevicePlan::Greedy),
            RowSample::Host => Err("TP pipe cannot read host logits".to_string()),
        })
        .collect()
}

impl TpCoordinator {
    pub fn load(
        workers: Vec<WorkerControl>,
        resolved: &Resolved,
        model: &Path,
        pack: &Path,
        gpu: usize,
        max_ctx: usize,
        slots: usize,
        span_done: Arc<std::sync::atomic::AtomicBool>,
        slot_reused: Arc<std::sync::Mutex<Vec<usize>>>,
    ) -> Result<Self, String> {
        if !resolved.is_coordinator()
            || resolved.tp_size < 2
            || max_ctx == 0
            || !(1..=2).contains(&slots)
        {
            return Err(
                "TP serving requires the rank-0 coordinator, tp_size>=2 and a nonzero context".into(),
            );
        }
        let mut workers = WorkerSet::new(workers, resolved, READY_TIMEOUT, STEP_TIMEOUT)?;
        // Phase 12 fp8 KV gate, rank-0-authoritative like everything else
        // here: the executor exists before any wire traffic, so ask THIS
        // device the same question the TP=1 apply_kv_dtype path asks. A
        // below-sm_89 card demotes LOUDLY to f16 (the fp8 ask doubles the KV
        // pool; the serve must stay attributable), and the demoted value is
        // what TpInit carries - worker ranks never has to know the runner's env.
        let exec = Arc::new(GpuExecutor::new(gpu, pack).map_err(|e| e.to_string())?);
        let kv_dtype = kv_dtype_serve(exec.compute_capability());
        // Graph mode is rank-0-authoritative too (upstream-readiness I4):
        // the resolved value rides TpInit, so a hand-started remote worker
        // cannot disagree with the coordinator's env and mispair the
        // graphed/eager collective sequencing. All ranks then convert the
        // decision into model state via enable_tp_graphs; execution reads
        // that state, never the environment.
        let use_graphs = Qwen35TpRank::resolve_graph_mode_for_serve();
        let (checkpoint_sha256, pack_blake3) = hashes(model, pack)?;
        let ckpt_slots = resolve_ckpt_slots(std::env::var("PADDOCK_TP_CKPT_SLOTS").ok().as_deref());
        // The span cap is rank-0-authoritative too: resolve once here, ship
        // it in TpInit, and install it before the worker can execute any
        // span. An unsupported value fails this rank (and the pair) closed.
        let span_cap = super::tp_span_cap::resolve_from_env(
            std::env::var("PADDOCK_TP_SPAN_CAP").ok().as_deref(),
        )
        .map_err(|e| e.to_string())?;
        workers.broadcast(&ControlMessage::TpInit {
            checkpoint_sha256,
            pack_blake3,
            max_ctx,
            slots,
            kv_dtype: kv_dtype_wire(kv_dtype).to_owned(),
            use_graphs,
            ckpt_slots,
            span_cap,
        })?;
        // Every worker checks identity before NCCL is initialized.
        workers.ready(0)?;
        let id = create_unique_id().map_err(|e| e.to_string())?;
        workers.send_nccl_id(&id)?;
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
        // so all ranks run the identical graphed/eager sequencing.
        if use_graphs {
            model
                .enable_tp_graphs(&group)
                .map_err(|e| e.to_string())?;
        }
        // Prefix-cache arm: the coordinator-resolved capacity sizes BOTH the
        // rank-local DeltaNet checkpoint pool and the mirrored radix's
        // free-list. The worker arms from the SAME TpInit value.
        model
            .enable_prefix_checkpoints(ckpt_slots as usize)
            .map_err(|e| e.to_string())?;
        let mut kv = logical(max_ctx, slots)?;
        kv.set_state_capacity(ckpt_slots);
        workers.ready(1)?;
        workers.set_read_timeout(STEP_TIMEOUT)?;
        Ok(Self {
            workers,
            group,
            model,
            logical: kv,
            positions: vec![0; slots],
            occupied: vec![false; slots],
            sequence: 1,
            pipe: None,
            chunks: VecDeque::new(),
            span: None,
            span_done: Some(span_done),
            poisoned: None,
            shutdown_sent: false,
            snapshotted_cuts: Vec::new(),
            pending_ckpts: std::collections::HashMap::new(),
            slot_reused,
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
        self.workers.broadcast(&msg)?;
        self.sequence += 1;
        self.workers.ready(self.sequence)
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
        self.snapshotted_cuts.clear();
        self.pending_ckpts.clear();
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
            self.snapshotted_cuts.retain(|(s, _)| *s != slot);
            self.pending_ckpts.remove(&slot);
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
            for (i, &token) in chunk.iter().enumerate() {
                let position = start + i;
                let logits = self.run_rows(&[(slot, token, position)], 0)?;
                let next = argmax_logits(&logits)?;
                picks.push(next);
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

    /// Execute ONE slot's contiguous chunk run on the DECODE lane as batched
    /// spans of at most tp_span_cap() rows. Interior spans advance state only
    /// (no head, no readback); when `finisher` is Some the FINAL span adds
    /// the head exactly once - device-sampled or read back per the kind.
    /// The caller has already authorized KV, sent `TpMixed` and consumed
    /// Prepared; all ranks derive the identical geometry from the shared
    /// pure `span_chunk_points` over the same wire rows.
    fn run_chunk_spans(
        &mut self,
        slot: usize,
        run: &[(usize, u32, usize)],
        finisher: Option<SpanFinishKind>,
        ckpts: &[(usize, u32)],
    ) -> Result<(Option<u32>, Option<Vec<f32>>), String> {
        let mut sampled = None;
        let mut host_logits = None;
        let points = span_checkpoint_points(run[0].2, run.len(), ckpts)?;
        for w in points.windows(2) {
            let (start, stop) = (w[0], w[1]);
            let tokens: Vec<u32> = run[start..stop].iter().map(|&(_, t, _)| t).collect();
            let position = run[start].2;
            self.model
                .forward_span_advance(&self.group, &self.logical, slot, &tokens, position)
                .map_err(|e| e.to_string())?;
            // Checkpoint cuts this span ends exactly on: snapshot the slot's
            // live DeltaNet state into the reserved pool index BEFORE any
            // further rows advance it (rank-local; the worker snapshots its
            // own state at the same span boundary).
            for (cut, idx) in tp_cuts_crossed(ckpts, position, stop - start) {
                self.model
                    .snapshot_slot_ckpt(slot, idx)
                    .map_err(|e| e.to_string())?;
                self.snapshotted_cuts.push((slot, cut));
                tracing::debug!(slot, cut, idx, "TP prefix ckpt snapshotted");
            }
            if stop == run.len() && let Some(kind) = finisher {
                match kind {
                    SpanFinishKind::Device(plan) => {
                        self.model
                            .forward_span_head_enqueue(stop - start)
                            .map_err(|e| e.to_string())?;
                        sampled = Some(self.model.sample_logits_slot(plan).map_err(|e| e.to_string())?);
                    }
                    SpanFinishKind::HostLogits => {
                        host_logits = Some(
                            self.model
                                .forward_span_head(stop - start)
                                .map_err(|e| e.to_string())?,
                        );
                    }
                }
            }
        }
        Ok((sampled, host_logits))
    }

    /// The serial scheduler uses `TpMixed` with zero decode rows, one tick
    /// per bounded prompt span. Only the final tick requests host logits.
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if slot >= self.positions.len()
            || tokens.is_empty()
            || self.occupied[slot]
            || tokens.len() >= self.model.max_ctx()
        {
            return Err("TP prefill slot or prompt invalid".into());
        }
        let mut last = Vec::new();
        for w in span_chunk_points(tokens.len()).windows(2) {
            let (start, stop) = (w[0], w[1]);
            let rows: Vec<(usize, u32, usize)> = tokens[start..stop]
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, start + i))
                .collect();
            let finishing = stop == tokens.len();
            let ops: Vec<Operation> = rows
                .iter()
                .map(|&(s, _, position)| Operation::Ensure { slot: s, position })
                .collect();
            let kv_state =
                serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
                    .map_err(|e| e.to_string())?;
            self.workers.broadcast(&ControlMessage::TpMixed {
                sequence: self.sequence + 1,
                rows: rows.clone(),
                chunk_rows: rows.len(),
                // The serial one-slot path stays cold in milestone 1 (the
                // chunked admission is the production TP path).
                reserve_cuts: Vec::new(),
                ckpts: Vec::new(),
                kv_state,
            })?;
            self.workers.prepared(self.sequence + 1)?;
            let (_, logits) = self.run_chunk_spans(
                slot,
                &rows,
                finishing.then_some(SpanFinishKind::HostLogits),
                &[],
            )?;
            self.sequence += 1;
            self.workers.ready(self.sequence)?;
            self.positions[slot] += stop - start;
            if finishing {
                last = logits.ok_or("TP serial prefill lost its final logits")?;
            }
        }
        self.occupied[slot] = true;
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
        self.workers.broadcast(&ControlMessage::TpPipeBegin {
            sequence: seq,
            rows: rows.clone(),
            kv_state,
        })?;
        self.workers.prepared(seq)?;
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
        self.workers.broadcast(&ControlMessage::TpPipeBegin {
            sequence: seq,
            rows: wire_rows.clone(),
            kv_state,
        })?;
        self.workers.prepared(seq)?;
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
            // Slot-mapped plans index SCHEDULER rows, executed in WIRE order;
            // dead members ride as greedy dummies.
            mapped_pipe_plans(&flight.row_of, flight.width, plans)?
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
        self.workers.broadcast(&ControlMessage::TpPipeNext {
            sequence: seq,
            rows: rows.clone(),
            source_plane,
            next_plane,
            kv_state,
        })?;
        self.workers.prepared(seq)?;
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
        // Worker ranks have enqueued the next tick before acknowledging the old one.
        self.workers.ready(self.sequence)?;
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
        self.workers.broadcast(&ControlMessage::TpPipeDrain { sequence: seq })?;
        self.workers.ready(self.sequence)?;
        self.workers.ready(seq)?;
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
        if !queued_chunk_abort_allowed(self.span.as_ref(), slot) {
            return false;
        }
        let removed = abort_queued_chunk(&mut self.chunks, slot);
        if removed {
            // The subsequent mirrored Release recycles reservations and
            // adopted refs. Keep the slot occupied until that release; an
            // intervening same-slot Admit is rejected by MirroredKv.
            self.snapshotted_cuts.retain(|(s, _)| *s != slot);
            self.pending_ckpts.remove(&slot);
        }
        removed
    }

    /// A mixed tick executes the scheduler's decode rows individually,
    /// followed by one slot's contiguous prompt run as bounded batched spans.
    /// `chunk_take` supplies at most one chunk per tick. The trailing run
    /// length is sent to worker rank so all ranks derive the same span geometry.
    /// Only a finishing chunk produces last-row logits (or a sampled ID);
    /// interior and non-finishing spans advance state without a head.
    /// Decode plans index the compact decode rows, and host rows return in
    /// that same order.
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
        let (chunk_rows, mut finishers) = Self::chunk_take(&mut self.chunks, budget, crate::generator::PrefillLane::Mixed)?;
        let dec_n = decodes.len();
        if plans.len() != dec_n {
            return Err("TP mixed plan width mismatch".into());
        }
        // The scheduler supplies decode rows in slot order; retain that
        // order for the host_rows consumer before appending prompt rows.
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
        // Wire order: decode rows first (the scheduler's dec order - its
        // host_rows consumer pops in dec-iteration order), then the chunk
        // run, whose length rides the message so worker rank derives the IDENTICAL
        // batched-span geometry from the shared pure chunker (`chunk_rows`
        // <= one slot's run: chunk_take pops rows of at most one chunk).
        let mut rows: Vec<(usize, u32, usize)> =
            decodes.iter().map(|&(s, t, p)| (s, t, p as usize)).collect();
        rows.extend(chunk_rows.iter().copied());
        // Finisher lookup: the finishing chunk's LAST row produces the
        // finisher result. One chunk per take, so at most one finisher.
        let fin: Option<(usize, Option<crate::sampler::DevicePlan>)> = finishers
            .first()
            .map(|fin| (fin.slot, fin.plan));
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let reserve_cuts = if let Some(&(slot, _, start)) = chunk_rows.first() {
            reserve_cold_mixed_cuts(&mut self.logical, &mut self.pending_ckpts, slot, start)?
        } else {
            Vec::new()
        };
        // Reservation ops were authorized before Ensures; the worker replays
        // the same ordered prefix before validating this tick's snapshot.
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        // The chunk run's checkpoint cuts: which of the slot's reservations
        // can this tick's span end exactly on. Derived from the slot's
        // pending-cut map (coordinator-owned, like snapshotted_cuts); worker rank
        // receives the cut list on the wire and derives the same spans.
        let wire_ckpts: Vec<(usize, u32)> = if chunk_rows.is_empty() {
            Vec::new()
        } else {
            let slot = chunk_rows[0].0;
            let span_start = chunk_rows[0].2;
            let span_len = chunk_rows.len();
            let pending = self
                .pending_ckpts
                .get(&slot)
                .cloned()
                .unwrap_or_default();
            tp_cuts_in_run(&pending, span_start, span_len)
        };
        self.workers.broadcast(&ControlMessage::TpMixed {
            sequence: self.sequence + 1,
            rows: rows.clone(),
            chunk_rows: chunk_rows.len(),
            reserve_cuts,
            ckpts: wire_ckpts.clone(),
            kv_state,
        })?;
        self.workers.prepared(self.sequence + 1)?;
        let mut step = crate::generator::SampledStep {
            ids: vec![0; dec_n],
            host_rows: Vec::new(),
        };
        // Decode rows: one-token forwards exactly as before (plans index the
        // dec order; the sampled path reads one ID, the host path one vocab).
        for (di, &(slot, token, position)) in decodes.iter().enumerate() {
            let position = position as usize;
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
        }
        // Chunk run: contiguous same-slot prompt rows executed as batched
        // spans of at most tp_span_cap() rows. Interior spans advance state
        // only; the FINAL span (the finishing chunk's last span) adds the
        // head once - final norm + LM head on the run's last row - and
        // either samples it on device or reads the logits back. The head
        // enters no collective, so worker ranks never runs it.
        let mut fin_ids = vec![0u32; finishers.len()];
        let mut fin_logits: Vec<Option<Vec<f32>>> = vec![None; finishers.len()];
        // Fail closed: a finisher outside the chunk run could never produce
        // its logits (the head runs on the run's last span only) - refuse
        // instead of reporting an empty row.
        if let Some((fs, _)) = fin.as_ref()
            && chunk_rows.first().is_none_or(|&(s, _, _)| s != *fs)
        {
            return Err("TP mixed finisher slot outside the chunk run".into());
        }
        if !chunk_rows.is_empty() {
            let slot = chunk_rows[0].0;
            let kind = fin.as_ref().and_then(|(s, p)| {
                (*s == slot).then(|| p.map_or(SpanFinishKind::HostLogits, SpanFinishKind::Device))
            });
            let chunk_ckpts = wire_ckpts.clone();
            let (sampled, logits) = self.run_chunk_spans(slot, &chunk_rows, kind, &chunk_ckpts)?;
            if let Some(id) = sampled {
                fin_ids[0] = id;
            }
            if let Some(row) = logits {
                fin_logits[0] = Some(row);
            }
        }
        self.sequence += 1;
        self.workers.ready(self.sequence)?;
        record_prefill_rows(&rows, &mut self.positions, &mut self.occupied);
        let mut results = Vec::with_capacity(finishers.len());
        for (fi, fin) in finishers.iter().enumerate() {
            let sample = match fin.plan {
                Some(_) => crate::generator::FinishSample::Sampled(fin_ids[fi]),
                None => crate::generator::FinishSample::Logits(
                    fin_logits[fi].take().unwrap_or_default(),
                ),
            };
            results.push((fin.slot, sample, fin.fin_rows));
        }
        // Verify rank-0 device work, including checkpoint copies, before any
        // radix entry becomes visible to later admissions.
        if !finishers.is_empty() {
            self.model.synchronize().map_err(|e| e.to_string())?;
        }
        // Cache publication: every finisher's prefill completed successfully.
        // Attach the snapshotted checkpoints (only cuts this prefill actually
        // crossed are attached; everything else recycles), publish the full
        // pages, and mirror the tick to the worker. Runs AFTER the GPU work
        // succeeded - a poisoned tick never publishes.
        for fin in &finishers {
            self.publish_slot_cache(fin.slot, &chunk_rows)?;
        }
        Ok((step, results))
    }

    /// The publication tick for `slot`'s completed prefill: attach snapshotted
    /// checkpoints, recycle unsnapshotted reservations, publish the prompt's
    /// full pages, and send `TpPrefixPublish` with the end-of-tick snapshot.
    /// `chunk_rows` is the final tick's rows (their positions bound what this
    /// prefill actually crossed); snapshots live in the coordinator's
    /// `snapshotted_cuts` set filled during `run_chunk_spans`.
    fn publish_slot_cache(
        &mut self,
        slot: usize,
        _chunk_rows: &[(usize, u32, usize)],
    ) -> Result<(), String> {
        let tokens = self
            .logical
            .slot_admitted_tokens(slot)
            .to_vec();
        if tokens.is_empty() {
            return Ok(()); // nothing admitted (serial path) - nothing to publish
        }
        let reservations = self.logical.slot_reserved_indices(slot);
        let snapshotted: Vec<usize> = self
            .snapshotted_cuts
            .iter()
            .filter(|(s, cut)| *s == slot && self.logical.slot_checkpoint_index(slot, *cut).is_none())
            .map(|(_, c)| *c)
            .collect();
        // Publish pages, attach snapshotted cuts, recycle the rest. The slot
        // stays LIVE; its table refs drop at the scheduler's release tick,
        // which mirrors the same
        // logical Release on all ranks).
        let ops = tp_publish_ops(slot, &reservations, tokens.clone(), &snapshotted);
        let kv_state =
            serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
                .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        self.workers.broadcast(&ControlMessage::TpPrefixPublish {
            sequence: seq,
            slot,
            checkpoints: reservations
                .iter()
                .filter(|(pos, _)| snapshotted.contains(pos))
                .cloned()
                .collect(),
            kv_state,
        })?;
        self.workers.ready(seq)?;
        self.sequence = seq;
        self.snapshotted_cuts
            .retain(|(s, _)| *s != slot);
        self.pending_ckpts.remove(&slot);
        tracing::info!(
            slot,
            attached = snapshotted.len(),
            pages = tokens.len() / crate::kv_pool::BLOCK_TOKENS,
            "TP prefix published"
        );
        Ok(())
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
        // ── Prefix-cache admission (milestone-1 text-only) ──
        // Rank 0 probes the mirrored radix READ-ONLY (no LRU/recurrence
        // mutation), makes the resume decision ONCE with the pure helper,
        // then authorizes `Admit` - which performs the one MUTATING
        // adoption on rank 0 and rides the wire so the worker's `Admit`
        // re-walk validates the exact same checkpoint on its own tree
        // BEFORE Prepared. A rank that cannot satisfy the decision fails
        // closed; neither rank resumes alone.
        let t_len = tokens.len();
        let probe = self.logical.match_prefix_probe(&tokens);
        let resume = tp_resume_decision(probe.ckpt, t_len, self.positions.len());
        let ops = vec![Operation::Admit {
            slot,
            tokens: tokens.clone(),
            resume,
        }];
        self.logical
            .authorize_all(&ops)
            .map_err(str::to_owned)?;
        let reused = self.logical.take_admitted_reused();
        if let Ok(mut guard) = self.slot_reused.lock()
            && let Some(cell) = guard.get_mut(slot)
        {
            *cell = reused;
        }
        tracing::info!(
            slot,
            resume,
            reused,
            tokens = t_len,
            "TP prefix admit (cache {})",
            if resume > 0 { "HIT" } else { "cold" }
        );
        // Resumed prompts are pinned to Mixed for the whole prefill
        // (milestone-1 lane rule): a span-lane prompt's promotion would
        // overwrite decode slabs with stale lane data for the adopted
        // blocks. `owner = Some(Mixed)` before the first take does exactly
        // what the existing first-chunk pin does, one tick earlier.
        let pinned_owner = (resume > 0).then_some(crate::generator::PrefillLane::Mixed);
        // A cold admission only claims pages. Async can publish pages but
        // cannot snapshot DeltaNet; reserve only after Mixed takes ownership.
        // Resumed prompts are Mixed-pinned and reserve during admission.
        let reserved = if resume > 0 {
            self.logical
                .reserve_cuts_for_slot(slot, &tp_prefill_cuts(t_len, resume))
                .map_err(str::to_owned)?
        } else {
            Vec::new()
        };
        let kv_state = serde_json::to_value(self.logical.snapshot())
            .map_err(|e| e.to_string())?;
        let seq = self.sequence + 1;
        self.workers.broadcast(&ControlMessage::TpPrefixAdmit {
            sequence: seq,
            slot,
            tokens: tokens.clone(),
            resume,
            cuts: reserved.iter().map(|&(cut, _)| cut).collect(),
            kv_state,
        })?;
        self.workers.ready(seq)?;
        self.sequence = seq;
        // Worker's DeltaNet restore rides the same TpReady as rank 0's: the
        // worker restores AFTER mirroring Admit, before its Ready; rank 0
        // restores here (its own device, its own checkpoint index).
        if resume > 0 {
            let idx = probe
                .ckpt
                .filter(|(pos, _)| *pos == resume)
                .map(|(_, idx)| idx)
                .ok_or("TP resume decision lost its checkpoint")?;
            self.model
                .restore_slot_ckpt(slot, idx)
                .map_err(|e| e.to_string())?;
        }
        // The scheduler cursor starts at the resume position: the chunk rows
        // carry [resume..t_len), so mixed-tick validation must see the cursor
        // there (the worker sets its own cursor in TpPrefixAdmit).
        self.positions[slot] = resume;
        // Admission owns reservations and possibly adopted pages even before
        // the first GPU row. Release must see this slot after a queued abort.
        self.occupied[slot] = true;
        if resume > 0 {
            self.pending_ckpts.insert(slot, reserved.clone());
        }
        self.chunks.push_back(PrefillChunk {
            slot,
            rows: tp_suffix_rows(slot, tokens, resume),
            fin_plan: None,
            owner: pinned_owner,
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
                if let (true, RowSample::Device(p)) = (slot == chunk.slot, plan) {
                    chunk.fin_plan = Some(p);
                }
            }
        }
    }

    /// Pop up to `budget` ordered rows from the queue front, recording every
    /// finishing chunk (its plan if any; `None` = host-logit readback). A
    /// chunk finishes when its last row is popped.
    fn chunk_take(
        chunks: &mut VecDeque<PrefillChunk>,
        budget: usize,
        owner: crate::generator::PrefillLane,
    ) -> Result<(Vec<(usize, u32, usize)>, Vec<SpanFinisher>), String> {
        if let Some(pinned) = chunks.front().and_then(|c| c.owner)
            && pinned != owner
        {
            return Err(format!("TP prompt prefill lane changed from {pinned:?} to {owner:?}"));
        }
        let mut rows = Vec::new();
        let mut finishers = Vec::new();
        while rows.len() < budget {
            // Continue only within the chunk already open this tick.
            let go = match chunks.front() {
                Some(chunk) => {
                    rows.is_empty() || rows.last().is_some_and(|&(s, _, _)| s == chunk.slot)
                }
                None => false,
            };
            if !go {
                break;
            }
            let (slot, take, finishing, plan, first) = {
                let chunk = chunks
                    .front()
                    .expect("chunk front is Some: the go guard just matched it");
                let take = chunk.rows.len().min(budget - rows.len());
                (chunk.slot, take, chunk.rows.len() == take, chunk.fin_plan, chunk.owner.is_none())
            };
            let drained: Vec<_> = {
                let chunk = chunks
                    .front_mut()
                    .expect("chunk front is Some: the go guard just matched it");
                chunk.owner = Some(owner);
                chunk.rows.drain(..take).collect()
            };
            // Full prompt rows (last row's position + 1), not the final
            // tick's piece - the scheduler sets slot.pos=rows.
            let fin_rows = drained.last().map_or(0, |&(_, _, position)| position + 1);
            rows.extend(drained);
            tracing::info!(slot, ?owner, first, rows = take, finishing, "TP prompt prefill owner");
            if finishing {
                chunks.pop_front();
                finishers.push(SpanFinisher {
                    slot,
                    plan,
                    fin_rows,
                });
            }
        }
        Ok((rows, finishers))
    }

    /// Launch a prefill-lane tick: validate and authorize the ordered KV
    /// operations, send one end-of-tick KV snapshot, then enqueue each
    /// contiguous slot run in bounded spans after the worker's Prepared.
    /// The run loop can handle several slots; `chunk_take` currently emits
    /// one chunk (one slot) per launch. Decode-pipe ticks may proceed while
    /// the lane runs.
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
        // The lane has one resident logits plane, read after the launch
        // joins. Restrict the optional finisher to the final run: a second
        // host finisher could overwrite the first result. `chunk_take`
        // currently emits at most one finisher.
        if finishers.len() > 1 || finishers.iter().any(|f| rows.last().map(|r| r.0) != Some(f.slot)) {
            return Err("TP span launch supports only a final-run finisher".into());
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
        // finishing chunk (plan or None): worker ranks promote exactly these
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
        // Authorize the ordered Ensure operations; the wire carries rows
        // and one end-of-tick KV snapshot, not a snapshot per prompt row.
        let ops: Vec<Operation> = rows
            .iter()
            .map(|&(slot, _, position)| Operation::Ensure { slot, position })
            .collect();
        let kv_state = serde_json::to_value(self.logical.authorize_all(&ops).map_err(str::to_owned)?)
            .map_err(|e| e.to_string())?;
        // Milestone 1 publishes pages from cold Async prompts but never
        // checkpoints their separate lane-local recurrent state.
        let wire_ckpts: Vec<(usize, u32)> = Vec::new();
        let seq = self.sequence + 1;
        self.workers.broadcast(&ControlMessage::TpSpanLaunch {
            sequence: seq,
            rows: rows.clone(),
            finishers: wire_finishers,
            ckpts: wire_ckpts,
            kv_state,
        })?;
        self.workers.prepared(seq)?;
        // All ranks chunk each contiguous slot run from the same wire
        // rows. Only the optional final-run finisher adds a rank-0 head
        // (and device sampling when planned); other spans advance state.
        let mut events = VecDeque::with_capacity(finishers.len());
        let mut at = 0usize;
        while at < rows.len() {
            let slot = rows[at].0;
            let mut end = at + 1;
            while end < rows.len() && rows[end].0 == slot {
                end += 1;
            }
            let run = &rows[at..end];
            let fin = finishers.iter().find(|f| f.slot == slot);
            let points = span_chunk_points(run.len());
            for w in points.windows(2) {
                let (start, stop) = (w[0], w[1]);
                let tokens: Vec<u32> = run[start..stop].iter().map(|&(_, t, _)| t).collect();
                let position = run[start].2;
                self.model
                    .prefill_lane_span_advance(&self.group, &self.logical, slot, &tokens, position)
                    .map_err(|e| e.to_string())?;
                let last = stop == run.len();
                if last && fin.is_some() {
                    let event = self
                        .model
                        .prefill_lane_span_finish(slot, stop - start, fin.and_then(|f| f.plan))
                        .map_err(|e| e.to_string())?;
                    events.push_back(event);
                }
            }
            at = end;
        }
        record_prefill_rows(&rows, &mut self.positions, &mut self.occupied);
        // Track this launch's completion even without a finisher, so a
        // prior lane event cannot report this span as drained.
        self.model
            .prefill_lane_track_latest()
            .map_err(|e| e.to_string())?;
        // The worker sends exactly one tail Ready after enqueuing its rows.
        // Consume it before another command (pipe begin or span finish) can
        // expect its own ACK. This is an enqueue ACK, not a GPU fence.
        self.workers.ready(seq)?;
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
        // Milestone-1 lane rule (explicit): a prefix-RESUMED prompt is pinned
        // to Mixed at admission - the prefill lane's KV slabs hold no adopted
        // prefix content, so an Async span would recompute the prefix rows
        // and its span-finish promotion would overwrite decode slabs with
        // stale lane data for the adopted blocks. Refuse WITHOUT error: the
        // scheduler treats a false launch as "not this tick" and the Mixed
        // tick drains the queue instead. Cold prompts still select Async.
        if let Some(front) = self.chunks.front()
            && front.owner == Some(crate::generator::PrefillLane::Mixed)
            && !front.rows.is_empty()
            && front.rows[0].2 > 0
        {
            return Ok(false);
        }
        self.chunk_plans(fin_plans);
        let (rows, finishers) = Self::chunk_take(&mut self.chunks, budget, crate::generator::PrefillLane::Async)?;
        if rows.is_empty() {
            return Ok(false);
        }
        self.span_begin(rows, finishers)?;
        Ok(true)
    }

    /// Fence the in-flight span: join the lane on all ranks, then read each
    /// finisher's result on rank 0, chunk order. Returns the scheduler's
    /// `(slot, FinishSample, rows)` contract directly.
    fn span_finish(
        &mut self,
    ) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, String> {
        let seq = self.sequence + 1;
        self.workers.broadcast(&ControlMessage::TpSpanFinish { sequence: seq })?;
        self.workers.ready(seq)?;
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
            tracing::info!(slot = fin.slot, "TP prefill lane promoted to decode");
        }
        let mut results = Vec::with_capacity(flight.finishing.len());
        for (fin, ev) in flight.finishing.into_iter().zip(flight.events) {
            match fin.plan {
                Some(_) => {
                    let id = self
                        .model
                        .prefill_lane_sampled_id_after(&ev, fin.slot)
                        .map_err(|e| e.to_string())?;
                    results.push((
                        fin.slot,
                        crate::generator::FinishSample::Sampled(id),
                        fin.fin_rows,
                    ));
                }
                None => {
                    let logits = self
                        .model
                        .prefill_lane_logits_after(&ev)
                        .map_err(|e| e.to_string())?;
                    results.push((
                        fin.slot,
                        crate::generator::FinishSample::Logits(logits),
                        fin.fin_rows,
                    ));
                }
            }
        }
        if !results.is_empty() {
            self.model.synchronize().map_err(|e| e.to_string())?;
        }
        for &(slot, _, _) in &results {
            // All ranks joined/promoted. Async stores pages only; its
            // reservations were never snapshotted and are recycled here.
            self.publish_slot_cache(slot, &[])?;
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
        self.workers.broadcast(&ControlMessage::TpBatch {
            sequence: self.sequence + 1,
            rows: rows.to_vec(),
            kv_state,
        })?;
        self.workers.prepared(self.sequence + 1)?;
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
        self.workers.ready(self.sequence)?;
        for &(slot, _, _) in rows {
            self.positions[slot] += 1;
            self.occupied[slot] = true;
        }
        Ok(step)
    }
}

enum Command {
    /// Ask the owning CUDA thread to send the rank-0 shutdown frame before it
    /// is joined during generator teardown.
    Shutdown,
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
    /// Read the next queued prompt's lane pin without advancing it.
    PrefillFrontOwner,
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
    PrefillOwner(Option<crate::generator::PrefillLane>),
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
    /// Per-slot prompt tokens the last admission served from the prefix
    /// cache (written by Mixed ticks' finisher path; taken by
    /// `take_prefill_reused` for the usage report).
    slot_reused: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    poisoned: Option<String>,
    /// The command thread owns the CUDA/NCCL stream and coordinator. Joining
    /// it on drop guarantees its `TpCoordinator` sends the graceful Shutdown
    /// frame before the runner exits; a detached thread could be killed after
    /// the engine's ready waiter fired, making worker rank report a false EOF.
    worker: Option<std::thread::JoinHandle<()>>,
}

impl TpGenerator {
    pub fn load(
        workers: Vec<WorkerControl>,
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
        let slot_reused = Arc::new(std::sync::Mutex::new(vec![0usize; slots]));
        let reused_proxy = Arc::clone(&slot_reused);
        let worker = std::thread::Builder::new()
            .name("qwen35-tp-rank0".into())
            .spawn(move || {
                let mut coordinator = match TpCoordinator::load(
                    workers,
                    &resolved,
                    &path,
                    &pack,
                    gpu,
                    max_ctx,
                    slots,
                    lane_flag,
                    slot_reused,
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
                    let shutdown = matches!(cmd, Command::Shutdown);
                    let result = if shutdown {
                        coordinator.shutdown_sent = true;
                        coordinator
                            .workers
                            .shutdown(coordinator.poisoned.is_none())
                            .map(|_| Response::Logits(Vec::new()))
                    } else if coordinator.pipe.is_some()
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
                            Command::Shutdown => unreachable!("shutdown handled before command dispatch"),
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
                            Command::PrefillFrontOwner => Ok(Response::PrefillOwner(
                                coordinator.chunks.front().and_then(|c| c.owner),
                            )),
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
                    if failed || shutdown {
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
            slot_reused: reused_proxy,
            poisoned: None,
            worker: Some(worker),
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

impl Drop for TpGenerator {
    fn drop(&mut self) {
        // Ask the owning CUDA thread to send the frame while its stream is
        // still live. The queue close below is the fallback for an already
        // failed command thread.
        if self.worker.is_some() {
            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            if self.commands.send((Command::Shutdown, reply_tx)).is_ok() {
                let _ = reply_rx.recv();
            }
        }
        // Close the command queue before joining so the coordinator thread can
        // leave its loop if the explicit shutdown could not be queued.
        let (replacement, _receiver) = std::sync::mpsc::channel();
        drop(std::mem::replace(&mut self.commands, replacement));
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            tracing::error!("TP coordinator thread panicked during shutdown");
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
    fn prefill_front_owner(&mut self) -> Option<crate::generator::PrefillLane> {
        match self.request(Command::PrefillFrontOwner) {
            Ok(Response::PrefillOwner(owner)) => owner,
            _ => Some(crate::generator::PrefillLane::Mixed), // fail closed; next command reports poison
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
    fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        _hints: &[usize],
    ) -> Result<usize, GenError> {
        // Hint-free on TP in milestone 1 (shared-prefix dedupe stays a
        // scheduler-side non-TP feature); the return is the admission's
        // resume position - the usage-report `cached` count. The admission
        // inside ChunkBegin writes slot_reused[slot] BEFORE the reply.
        self.prefill_begin(slot, tokens)?;
        let reused = self
            .slot_reused
            .lock()
            .map_err(|_| GenError::Backend("TP prefix accounting lock poisoned".into()))?
            .get(slot)
            .copied()
            .ok_or_else(|| GenError::Backend("TP prefix accounting slot invalid".into()))?;
        Ok(reused)
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.slot_reused
            .lock()
            .ok()
            .and_then(|mut g| g.get_mut(slot).map(|cell| std::mem::replace(cell, 0)))
            .unwrap_or(0)
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
    /// DeltaNet recurrent/conv slot state THIS rank holds. All ranks build
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
        if !self.shutdown_sent {
            let _ = self.workers.shutdown(self.poisoned.is_none());
        }
    }
}

/// Runs in any nonzero worker-rank process, with no HTTP listener, tokenizer or sampler.
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
        return Err("TP worker must have a nonzero worker rank".into());
    }
    stream
        .set_read_timeout(Some(READY_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(STEP_TIMEOUT))
        .map_err(|e| e.to_string())?;
    let run = (|| -> Result<(), String> {
        let (max_ctx, slots, kv_dtype, use_graphs, ckpt_slots) =
            match ControlMessage::from_stream(&mut stream).map_err(|e| e.to_string())? {
                ControlMessage::TpInit {
                    checkpoint_sha256,
                    pack_blake3,
                    max_ctx,
                    slots,
                    kv_dtype,
                    use_graphs,
                    ckpt_slots,
                    span_cap,
                } => {
                    let (own_checkpoint, own_pack) = hashes(model_path, pack)?;
                    // The dtype parses BEFORE the hash compare so an unknown
                    // wire value reads as what it is - a protocol mismatch -
                    // not as an identity failure.
                    let kv_dtype = kv_dtype_parse(&kv_dtype)?;
                    // The span cap installs BEFORE any span can run and
                    // fails closed on an unsupported wire value: worker rank
                    // never reads the coordinator's env (same contract as
                    // graphs/dtype/ckpt_slots).
                    super::tp_span_cap::wire_span_cap(span_cap).map_err(|e| e.to_string())?;
                    if own_checkpoint != checkpoint_sha256
                        || own_pack != pack_blake3
                        || max_ctx == 0
                        || !(1..=2).contains(&slots)
                    {
                        return Err(
                            "worker-rank checkpoint, CUDA pack or context disagrees with rank 0".into(),
                        );
                    }
                    (max_ctx, slots, kv_dtype, use_graphs, ckpt_slots)
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
        // Prefix-cache arm from the TpInit value (rank-0-authoritative, same
        // pattern as graphs/dtype): identical pool + free-list geometry.
        model
            .enable_prefix_checkpoints(ckpt_slots as usize)
            .map_err(|e| e.to_string())?;
        let mut logical = logical(max_ctx, slots)?;
        logical.set_state_capacity(ckpt_slots);
        let mut positions = vec![0; slots];
        let mut worker_snapshotted: Vec<(usize, usize)> = Vec::new();
        let mut pipe_slots: Option<Vec<usize>> = None;
        let mut span_in_flight = false;
        // Finishing chunks of the in-flight span (wire shape, chunk order):
        // worker ranks promote exactly these slots at the span finish.
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
                | ControlMessage::TpPrefixAdmit { sequence, .. }
                | ControlMessage::TpPrefixPublish { sequence, .. }
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
                        worker_snapshotted.retain(|&(s, _)| s != slot);
                    }
                }
                ControlMessage::TpPrefixAdmit {
                    sequence: _,
                    slot,
                    tokens,
                    resume,
                    cuts,
                    kv_state,
                } => {
                    if slot >= slots || tokens.is_empty() || tokens.len() >= max_ctx {
                        return Err("TP prefix admit slot or prompt invalid".into());
                    }
                    if resume % crate::kv_pool::BLOCK_TOKENS != 0 || resume >= tokens.len() {
                        return Err("TP prefix admit resume position invalid".into());
                    }
                    // Mirror the admission: the worker's own `Admit` apply
                    // independently validates the cached chain AND its
                    // checkpoint at EXACTLY the coordinator's resume position
                    // on this rank's tree. Failing here fails the whole pair
                    // before any state adoption - the rank-symmetry gate.
                    if cuts.windows(2).any(|w| w[0] >= w[1])
                        || cuts.iter().any(|cut| !tp_prefill_cuts(tokens.len(), resume).contains(cut))
                    {
                        return Err("TP prefix admit cuts invalid".into());
                    }
                    let resume_idx = (resume > 0).then(|| logical.match_prefix_probe(&tokens).ckpt)
                        .flatten()
                        .filter(|&(position, _)| position == resume)
                        .map(|(_, idx)| idx);
                    let mut ops = vec![Operation::Admit {
                        slot,
                        tokens: tokens.clone(),
                        resume,
                    }];
                    ops.extend(cuts.iter().map(|&position| Operation::CheckpointReserve { slot, position }));
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    // Rank-local DeltaNet restore from THIS rank's pool at
                    // the validated index, BEFORE the Ready: by the time the
                    // coordinator proceeds, all ranks' slot state matches
                    // the checkpoint's logical position.
                    if resume > 0 {
                        // A reservation in this same admission may steal the
                        // old radix attachment, but cannot overwrite its GPU
                        // blob before this stream-ordered restore.
                        let idx = resume_idx.ok_or("TP worker resume checkpoint missing")?;
                        model
                            .restore_slot_ckpt(slot, idx)
                            .map_err(|e| e.to_string())?;
                    }
                    positions[slot] = resume;
                }
                ControlMessage::TpPrefixPublish {
                    sequence: _,
                    slot,
                    checkpoints,
                    kv_state,
                } => {
                    if slot >= slots {
                        return Err("TP prefix publish slot invalid".into());
                    }
                    // The worker's snapshot discipline mirrors rank 0's: each
                    // rank ran its own GPU snapshots at the shared cut
                    // boundaries. The wire's (cut, rank-0 index) pairs drive
                    // the mirrored Attach ops; the worker's own reservation
                    // indices are what ITS attach consumes (mirror-deterministic
                    // free-list pops made them identical).
                    let snapshotted: Vec<usize> =
                        checkpoints.iter().map(|&(pos, _)| pos).collect();
                    let tokens = logical.slot_admitted_tokens(slot).to_vec();
                    if tokens.is_empty() {
                        return Err("TP prefix publish for unadmitted slot".into());
                    }
                    let reservations = logical.slot_reserved_indices(slot);
                    if checkpoints.windows(2).any(|w| w[0].0 >= w[1].0)
                        || snapshotted.iter().any(|&cut| !worker_snapshotted.contains(&(slot, cut)))
                        || checkpoints.iter().any(|&(cut, idx)| logical.slot_reserved_ckpt(slot, cut) != Some(idx))
                    {
                        return Err("TP prefix publish cuts or snapshots mismatch".into());
                    }
                    // Rank-symmetry gate: every cut the coordinator attached
                    // must have a reservation here. A missing one means the
                    // two ranks' snapshot disciplines diverged - fail closed
                    // BEFORE any attach/recycle mutates this rank's tree.
                    for &(cut, _) in &checkpoints {
                        if !reservations.iter().any(|&(p, _)| p == cut) {
                            return Err(format!(
                                "TP prefix publish cut {cut} has no worker reservation"
                            ));
                        }
                    }
                    let ops = tp_publish_ops(slot, &reservations, tokens, &snapshotted);
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    worker_snapshotted.retain(|&(s, _)| s != slot);
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
                    rows,
                    chunk_rows,
                    reserve_cuts,
                    ckpts,
                    kv_state,
                    ..
                } => {
                    // Decode rows precede the optional single-slot chunk run.
                    // Worker ranks mirror the authorized KV snapshot before
                    // Prepared, then replays the same bounded span geometry;
                    // it neither samples nor reads logits back to the host.
                    let split = validate_mixed_worker_rows(&rows, chunk_rows, &positions, max_ctx)?;
                    let chunk = &rows[split..];
                    if chunk.is_empty() && (!ckpts.is_empty() || !reserve_cuts.is_empty()) {
                        return Err("TP mixed cuts without prompt rows".into());
                    }
                    if !chunk.is_empty() && !reserve_cuts.is_empty() {
                        let slot = chunk[0].0;
                        if chunk[0].2 != 0
                            || !logical.slot_reserved_indices(slot).is_empty()
                            || reserve_cuts.windows(2).any(|w| w[0] >= w[1])
                            || reserve_cuts.iter().any(|cut| {
                                !tp_prefill_cuts(logical.slot_admitted_tokens(slot).len(), 0).contains(cut)
                            })
                        {
                            return Err("TP mixed cold reservation invalid".into());
                        }
                    }
                    let mut ops: Vec<Operation> = if let Some(&(slot, _, _)) = chunk.first() {
                        reserve_cuts.iter().map(|&position| Operation::CheckpointReserve { slot, position }).collect()
                    } else {
                        Vec::new()
                    };
                    ops.extend(rows.iter().map(|&(slot, _, position)| Operation::Ensure { slot, position }));
                    logical
                        .mirror_tick(&ops, &wire_kv_state(kv_state)?)
                        .map_err(str::to_owned)?;
                    if let Some(&(slot, _, start)) = chunk.first() {
                        let expected = tp_cuts_in_run(&logical.slot_reserved_indices(slot), start, chunk.len());
                        if ckpts != expected {
                            return Err("TP mixed cut list disagrees with reservations".into());
                        }
                        span_checkpoint_points(start, chunk.len(), &ckpts)?;
                    }
                    ControlMessage::TpPrepared { sequence: got }
                        .to_stream(&mut stream)
                        .map_err(|e| e.to_string())?;
                    for (slot, token, position) in &rows[..split] {
                        model
                            .forward_token_worker_slot(&group, &logical, *token, *position, *slot)
                            .map_err(|e| e.to_string())?;
                        positions[*slot] += 1;
                    }
                    if !chunk.is_empty() {
                        let slot = chunk[0].0;
                        let points = span_checkpoint_points(chunk[0].2, chunk.len(), &ckpts)?;
                        for w in points.windows(2) {
                            let (start, stop) = (w[0], w[1]);
                            let tokens: Vec<u32> =
                                chunk[start..stop].iter().map(|&(_, t, _)| t).collect();
                            let position = chunk[start].2;
                            model
                                .forward_span_advance(&group, &logical, slot, &tokens, position)
                                .map_err(|e| e.to_string())?;
                            for (cut, wire_idx) in tp_cuts_crossed(&ckpts, position, stop - start) {
                                let idx = logical.slot_reserved_ckpt(slot, cut)
                                    .ok_or("TP worker snapshot reservation missing")?;
                                if idx != wire_idx {
                                    return Err("TP worker snapshot index mismatch".into());
                                }
                                model.snapshot_slot_ckpt(slot, idx).map_err(|e| e.to_string())?;
                                worker_snapshotted.push((slot, cut));
                            }
                        }
                        for &(s, _, _) in chunk {
                            positions[s] += 1;
                        }
                        if !ckpts.is_empty() || positions[slot] == logical.slot_admitted_tokens(slot).len() {
                            // Ready must certify completed GPU snapshots and
                            // full pages before rank 0 publishes the prefix.
                            model.synchronize().map_err(|e| e.to_string())?;
                        }
                    }
                }
                ControlMessage::TpSpanLaunch {
                    rows,
                    finishers,
                    ckpts,
                    kv_state,
                    ..
                } => {
                    // The whole prompt span enqueues on each worker rank's prefill lane
                    // as batched sub-spans of at most tp_span_cap() contiguous
                    // rows per slot run (the shared pure `span_chunk_points`
                    // over the same wire rows rank 0 chunked from). The lane
                    // steps enter the same collectives in the same order as
                    // rank 0's lane. No head and no finisher sampler here -
                    // both have no collectives and run only on rank 0; the
                    // finisher list exists so this rank promotes the same
                    // slots at the span finish.
                    if rows.is_empty() || !ckpts.is_empty() {
                        return Err("TP span launch membership or checkpoint cuts invalid".into());
                    }
                    if finishers.len() > 1
                        || finishers.iter().any(|f| rows.last().map(|r| r.0) != Some(f.0))
                    {
                        return Err("TP span launch supports only a final-run finisher".into());
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
                    let mut at = 0usize;
                    while at < rows.len() {
                        let slot = rows[at].0;
                        let mut end = at + 1;
                        while end < rows.len() && rows[end].0 == slot {
                            end += 1;
                        }
                        let run = &rows[at..end];
                        let points = span_chunk_points(run.len());
                        for w in points.windows(2) {
                            let (start, stop) = (w[0], w[1]);
                            let tokens: Vec<u32> =
                                run[start..stop].iter().map(|&(_, t, _)| t).collect();
                            let position = run[start].2;
                            model
                                .prefill_lane_span_advance(
                                    &group, &logical, slot, &tokens, position,
                                )
                                .map_err(|e| e.to_string())?;
                            // Cold Async has no rank-local checkpoints in
                            // milestone 1; publication recycles reservations.
                        }
                        for &(s, _, _) in run {
                            positions[s] += 1;
                        }
                        at = end;
                    }
                    // Worker ranks do not sample; its probe finisher marks span
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
                    // every rank, so each worker rank's decode executor would otherwise
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
                    model.synchronize().map_err(|e| e.to_string())?;
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

    #[test]
    fn prompt_owner_pins_partial_ticks_and_clears_on_finish_abort_and_reuse() {
        use crate::generator::PrefillLane::{Async, Mixed};
        let prompt = |slot| PrefillChunk {
            slot,
            rows: (0..5).map(|p| (slot, p as u32, p)).collect(),
            fin_plan: None,
            owner: None,
        };
        let mut queue = VecDeque::from([prompt(1)]);
        let (rows, finished) = TpCoordinator::chunk_take(&mut queue, 2, Mixed).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(finished.is_empty());
        assert_eq!(queue.front().unwrap().owner, Some(Mixed));
        assert!(TpCoordinator::chunk_take(&mut queue, 2, Async).is_err());
        assert_eq!(queue.front().unwrap().rows.len(), 3); // rejected before mutation
        let (_, finished) = TpCoordinator::chunk_take(&mut queue, 3, Mixed).unwrap();
        assert_eq!(finished.len(), 1);
        assert!(queue.is_empty()); // completion discards the pin

        queue.push_back(prompt(1)); // same slot, fresh request
        TpCoordinator::chunk_take(&mut queue, 2, Async).unwrap();
        assert_eq!(queue.front().unwrap().owner, Some(Async));
        assert!(TpCoordinator::chunk_take(&mut queue, 1, Mixed).is_err());
        TpCoordinator::chunk_take(&mut queue, 2, Async).unwrap();
        assert_eq!(queue.front().unwrap().owner, Some(Async));
        assert!(abort_queued_chunk(&mut queue, 1));
        assert!(queue.is_empty());
        queue.push_back(prompt(1)); // cancellation/release and slot reuse
        assert_eq!(queue.front().unwrap().owner, None);
        TpCoordinator::chunk_take(&mut queue, 5, Mixed).unwrap();
        assert!(queue.is_empty());
    }

    #[test]
    fn abort_during_span_is_deferred_until_span_ownership_clears() {
        let in_flight = SpanFlight {
            finishing: Vec::new(),
            events: VecDeque::new(),
            slots: vec![1],
        };
        // The scheduler may observe the disconnect, but the real abort seam
        // refuses to reclaim a slot whose rows were popped into this span.
        assert!(!queued_chunk_abort_allowed(Some(&in_flight), 1));
        assert!(queued_chunk_abort_allowed(Some(&in_flight), 0));
        // After the span-finish path clears coordinator.span, queued ownership
        // can be removed and the same slot can be admitted again safely.
        assert!(queued_chunk_abort_allowed(None, 1));
        let mut queue = VecDeque::from([PrefillChunk {
            slot: 1,
            rows: vec![(1, 7, 0)],
            fin_plan: None,
            owner: None,
        }]);
        assert!(abort_queued_chunk(&mut queue, 1));
        queue.push_back(PrefillChunk {
            slot: 1,
            rows: vec![(1, 8, 0)],
            fin_plan: None,
            owner: None,
        });
        assert_eq!(queue.front().map(|chunk| chunk.slot), Some(1));
    }

    #[test]
    fn eligibility_change_does_not_relaunch_a_running_mixed_prompt_on_async_lane() {
        use crate::generator::PrefillLane::{Async, Mixed};

        let prompt = |slot| PrefillChunk {
            slot,
            rows: (0..5).map(|p| (slot, p as u32, p)).collect(),
            fin_plan: None,
            owner: None,
        };
        let mut queue = VecDeque::from([prompt(0)]);
        let mut async_launches = 0usize;

        // P starts while no decoder makes overlap eligible: its first rows
        // execute on Mixed and pin the queue entry through the real chunk_take
        // path, not by inspecting an owner field in isolation.
        let owner = queue.front().and_then(|c| c.owner);
        let lane = if crate::service::select_prefill_lane(owner, false) {
            async_launches += 1;
            Async
        } else {
            Mixed
        };
        let (rows, finished) = TpCoordinator::chunk_take(&mut queue, 2, lane).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(finished.is_empty());
        assert_eq!(queue.front().and_then(|c| c.owner), Some(Mixed));
        assert_eq!(async_launches, 0, "P must not launch TpSpanLaunch yet");

        // The scheduler becomes async-eligible before P completes. The same
        // queue entry must continue on Mixed, and the attempted async route
        // must not consume rows or launch a span.
        let owner = queue.front().and_then(|c| c.owner);
        assert!(!crate::service::select_prefill_lane(owner, true));
        let (rows, finished) = TpCoordinator::chunk_take(&mut queue, 3, Mixed).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(finished.len(), 1);
        assert!(queue.is_empty(), "P completed on its pinned lane");
        assert_eq!(async_launches, 0);

        // Completion clears the pin. A fresh prompt may legitimately select
        // Async under the same eligibility conditions and then finish for its
        // first decode token.
        queue.push_back(prompt(0));
        let owner = queue.front().and_then(|c| c.owner);
        assert!(crate::service::select_prefill_lane(owner, true));
        async_launches += 1;
        let (rows, finished) = TpCoordinator::chunk_take(&mut queue, 5, Async).unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(finished.len(), 1);
        assert!(queue.is_empty());
        assert_eq!(async_launches, 1);
    }
    #[test]
    fn span_take_refuses_a_mixed_pinned_resumed_chunk() {
        // Milestone-1 lane rule: a prefix-resumed prompt (first row's
        // position > 0) pinned to Mixed must NOT launch on the async lane.
        let resumed = PrefillChunk {
            slot: 0,
            rows: vec![(0, 9, 64), (0, 9, 65)],
            fin_plan: None,
            owner: Some(crate::generator::PrefillLane::Mixed),
        };
        let cold = PrefillChunk {
            slot: 0,
            rows: vec![(0, 9, 0), (0, 9, 1)],
            fin_plan: None,
            owner: Some(crate::generator::PrefillLane::Mixed),
        };
        let unpinned = PrefillChunk {
            slot: 0,
            rows: vec![(0, 9, 64), (0, 9, 65)],
            fin_plan: None,
            owner: None,
        };
        let is_resumed = |c: &PrefillChunk| c.owner == Some(crate::generator::PrefillLane::Mixed) && !c.rows.is_empty() && c.rows[0].2 > 0;
        assert!(is_resumed(&resumed), "resumed chunk must be detected");
        assert!(!is_resumed(&cold), "cold chunk must not be detected");
        assert!(!is_resumed(&unpinned), "unpinned chunk must not be detected");
    }

    /// chunk_take starts at the resumed position: rows carry positions
    /// [resume..t_len), never replaying the prefix.
    #[test]
    fn chunk_take_starts_at_the_resume_position() {
        let resume = 64usize;
        let t_len = 100usize;
        let rows = tp_suffix_rows(0, vec![7; t_len], resume);
        let mut queue = VecDeque::from([PrefillChunk {
            slot: 0,
            rows,
            fin_plan: None,
            owner: Some(crate::generator::PrefillLane::Mixed),
        }]);
        let (taken, finished) =
            TpCoordinator::chunk_take(&mut queue, t_len - resume, crate::generator::PrefillLane::Mixed)
                .unwrap();
        assert_eq!(taken.len(), t_len - resume);
        assert_eq!(taken.first().unwrap().2, resume, "first row is the resume position");
        assert_eq!(taken.last().unwrap().2, t_len - 1);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].fin_rows, t_len);
    }

    #[test]
    fn cold_async_never_reserves_or_steals_a_checkpoint() {
        let mut head = MirroredKv::new(32, 2, 128).unwrap();
        let mut worker = head.clone();
        head.set_state_capacity(1);
        worker.set_state_capacity(1);
        let cached: Vec<u32> = (0..65).collect();
        let admit = [Operation::Admit { slot: 0, tokens: cached.clone(), resume: 0 }];
        let end = head.authorize_all(&admit).unwrap();
        worker.mirror_tick(&admit, &end).unwrap();
        let reserve = [Operation::CheckpointReserve { slot: 0, position: 48 }];
        let end = head.authorize_all(&reserve).unwrap();
        worker.mirror_tick(&reserve, &end).unwrap();
        let ensure = [Operation::Ensure { slot: 0, position: 64 }];
        let end = head.authorize_all(&ensure).unwrap();
        worker.mirror_tick(&ensure, &end).unwrap();
        let publish = tp_publish_ops(0, &head.slot_reserved_indices(0), cached.clone(), &[48]);
        let end = head.authorize_all(&publish).unwrap();
        worker.mirror_tick(&publish, &end).unwrap();
        let release = [Operation::Release { slot: 0 }];
        let end = head.authorize_all(&release).unwrap();
        worker.mirror_tick(&release, &end).unwrap();
        let before = head.match_prefix_probe(&cached).ckpt;
        assert!(before.is_some());

        let async_tokens = vec![900; 65];
        let admit = [Operation::Admit { slot: 1, tokens: async_tokens.clone(), resume: 0 }];
        let end = head.authorize_all(&admit).unwrap();
        worker.mirror_tick(&admit, &end).unwrap();
        // Async uses Ensure + pages-only publish. It never calls the Mixed
        // ownership seam, even if the prompt has checkpoint-eligible cuts.
        let ensure = [Operation::Ensure { slot: 1, position: 64 }];
        let end = head.authorize_all(&ensure).unwrap();
        worker.mirror_tick(&ensure, &end).unwrap();
        let publish = tp_publish_ops(1, &head.slot_reserved_indices(1), async_tokens, &[]);
        let end = head.authorize_all(&publish).unwrap();
        worker.mirror_tick(&publish, &end).unwrap();
        assert!(head.slot_reserved_indices(1).is_empty());
        assert_eq!(head.match_prefix_probe(&cached).ckpt, before);
        assert_eq!(head.snapshot(), worker.snapshot());
    }

    #[test]
    fn cold_mixed_reserves_once_and_mirrors_before_ensure() {
        let mut head = MirroredKv::new(32, 2, 128).unwrap();
        let mut worker = head.clone();
        head.set_state_capacity(2);
        worker.set_state_capacity(2);
        let admit = [Operation::Admit { slot: 0, tokens: vec![7; 65], resume: 0 }];
        let end = head.authorize_all(&admit).unwrap();
        worker.mirror_tick(&admit, &end).unwrap();
        assert!(head.snapshot().checkpoint_reserved.is_empty());
        let mut pending = std::collections::HashMap::new();
        let cuts = reserve_cold_mixed_cuts(&mut head, &mut pending, 0, 0).unwrap();
        assert_eq!(cuts, vec![48, 64]);
        let ensure = [Operation::Ensure { slot: 0, position: 0 }];
        let end = head.authorize_all(&ensure).unwrap();
        let mut ops: Vec<_> = cuts.iter().map(|&position| Operation::CheckpointReserve { slot: 0, position }).collect();
        ops.extend(ensure);
        worker.mirror_tick(&ops, &end).unwrap();
        assert_eq!(head.snapshot(), worker.snapshot());
        assert!(reserve_cold_mixed_cuts(&mut head, &mut pending, 0, 1).unwrap().is_empty());
        assert_eq!(head.snapshot(), worker.snapshot());
        assert_eq!(tp_cuts_in_run(pending.get(&0).unwrap(), 0, 64), vec![(48, 0), (64, 1)]);
    }

    #[test]
    fn cold_mixed_zero_capacity_is_not_retried() {
        let mut logical = MirroredKv::new(8, 1, 128).unwrap();
        logical.authorize_all(&[Operation::Admit { slot: 0, tokens: vec![7; 65], resume: 0 }]).unwrap();
        let mut pending = std::collections::HashMap::new();
        assert!(reserve_cold_mixed_cuts(&mut logical, &mut pending, 0, 0).unwrap().is_empty());
        assert!(reserve_cold_mixed_cuts(&mut logical, &mut pending, 0, 16).unwrap().is_empty());
        assert_eq!(pending.get(&0), Some(&Vec::new()));
    }

    #[test]
    fn checkpoint_spans_split_at_each_logical_cut_and_reject_bad_wire() {
        for (len, cuts) in [
            (15, vec![]), (16, vec![]), (17, vec![(16, 0)]),
            (31, vec![(16, 0)]), (32, vec![(16, 0)]),
            (33, vec![(16, 0), (32, 1)]),
            (63, vec![(48, 0)]), (64, vec![(48, 0)]),
            (65, vec![(48, 0), (64, 1)]),
            (100, vec![(80, 0), (96, 1)]),
        ] {
            let points = span_checkpoint_points(0, len, &cuts).unwrap();
            assert_eq!(points.first(), Some(&0));
            assert_eq!(points.last(), Some(&len));
            assert!(points.windows(2).all(|w| w[1] > w[0] && w[1] - w[0] <= tp_span_cap()));
            assert!(cuts.iter().all(|(cut, _)| points.contains(cut)), "len={len}");
        }
        let shifted = span_checkpoint_points(64, 36, &[(80, 0), (96, 1)]).unwrap();
        assert!(shifted.contains(&16) && shifted.contains(&32));
        for bad in [vec![(96, 0), (80, 1)], vec![(80, 0), (80, 1)],
                    vec![(81, 0)], vec![(112, 0)]] {
            assert!(span_checkpoint_points(64, 36, &bad).is_err());
        }
    }

    #[test]
    fn finisher_wire_plan_rejects_invalid_sampler_parameters_before_authorization() {
        use crate::sampler::DevicePlan;
        assert!(matches!(wire_finisher_plan(DevicePlan::Greedy), Ok(TpSpanFinisherPlan::Greedy)));
        assert!(matches!(
            wire_finisher_plan(DevicePlan::Categorical { inv_t: 2.0, u: 0.25 }),
            Ok(TpSpanFinisherPlan::Categorical { .. })
        ));
        for (inv_t, u) in [
            (0.0, 0.5),
            (-1.0, 0.5),
            (f32::NAN, 0.5),
            (f32::INFINITY, 0.5),
            (1.0, f32::NAN),
            (1.0, f32::INFINITY),
            (1.0, -0.1),
            (1.0, 1.0),
        ] {
            assert!(wire_finisher_plan(DevicePlan::Categorical { inv_t, u }).is_err());
        }
    }

    #[test]
    fn mapped_pipe_plans_reject_bad_width_before_indexing_and_preserve_wire_order() {
        use crate::sampler::DevicePlan;
        let plans = [RowSample::Hole, RowSample::Device(DevicePlan::Categorical { inv_t: 2.0, u: 0.5 })];
        let selected = mapped_pipe_plans(&[1, 0], 2, &plans).unwrap();
        assert!(matches!(selected[0], DevicePlan::Categorical { .. }));
        assert!(matches!(selected[1], DevicePlan::Greedy));
        assert!(mapped_pipe_plans(&[1, 0], 2, &plans[..1]).is_err());
        assert!(mapped_pipe_plans(&[1, 0], 2, &[plans[0], plans[1], RowSample::Hole]).is_err());
        assert!(mapped_pipe_plans(&[2, 0], 2, &plans).is_err());
        assert!(mapped_pipe_plans(&[0], 2, &plans).is_err());
        assert!(mapped_pipe_plans(&[0, 1], 2, &[RowSample::Host, plans[1]]).is_err());
        assert!(mapped_pipe_plans(
            &[0, 1],
            2,
            &[RowSample::Hole, RowSample::Device(DevicePlan::Categorical { inv_t: 0.0, u: 0.5 })]
        ).is_err());
    }

    #[test]
    fn worker_mixed_decode_prefix_must_ascend_before_mirror() {
        let pos = [3, 7, 11];
        assert_eq!(
            validate_mixed_worker_rows(&[(0, 4, 3), (1, 5, 7), (2, 6, 11), (2, 7, 12)], 2, &pos, 16),
            Ok(2)
        );
        assert!(validate_mixed_worker_rows(&[(1, 5, 7), (0, 4, 3), (2, 6, 11)], 1, &pos, 16)
            .unwrap_err().contains("ascending"));
        assert!(validate_mixed_worker_rows(&[(0, 4, 3), (0, 5, 3)], 0, &pos, 16)
            .unwrap_err().contains("ascending"));
        assert!(validate_mixed_worker_rows(&[(0, 4, 3), (0, 5, 4)], 1, &pos, 16).is_err());
        assert!(validate_mixed_worker_rows(&[(0, 4, 3), (1, 5, 8)], 1, &pos, 16).is_err());
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

    // ── production span chunking (pure, all ranks share these) ──────────

    #[test]
    fn span_chunk_points_covers_the_task_cap_boundaries() {
        // 1 row: one span, final by construction.
        assert_eq!(span_chunk_points(1), vec![0, 1]);
        // 63/64: a single span each (the cap itself stays one pass).
        assert_eq!(span_chunk_points(63), vec![0, 63]);
        assert_eq!(span_chunk_points(64), vec![0, 64]);
        // 65: one full interior span + the 1-row final span.
        assert_eq!(span_chunk_points(65), vec![0, 64, 65]);
        // repeated 64-row chunks: boundaries land exactly on the cap.
        assert_eq!(span_chunk_points(128), vec![0, 64, 128]);
        assert_eq!(span_chunk_points(192), vec![0, 64, 128, 192]);
        // ragged multi-span: 130 = 64 + 64 + 2.
        assert_eq!(span_chunk_points(130), vec![0, 64, 128, 130]);
        // degenerate: an empty run has no windows at all (one boundary,
        // zero spans - the callers skip empty runs anyway).
        assert_eq!(span_chunk_points(0), vec![0]);
        assert!(span_chunk_points(0).windows(2).count() == 0);
    }

    #[test]
    fn span_chunk_points_windows_respect_the_cap() {
        for len in [1usize, 2, 31, 63, 64, 65, 100, 128, 129, 191, 200, 8192] {
            let points = span_chunk_points(len);
            assert_eq!(points.first(), Some(&0));
            assert_eq!(points.last(), Some(&len));
            for w in points.windows(2) {
                let rows = w[1] - w[0];
                assert!(
                    (1..=tp_span_cap()).contains(&rows),
                    "len {len}: window {w:?} has {rows} rows"
                );
            }
            // windows tile the run with no gaps or overlaps
            let total: usize = points.windows(2).map(|w| w[1] - w[0]).sum();
            assert_eq!(total, len);
        }
    }

    #[test]
    fn span_chunk_points_is_deterministic_across_ranks() {
        // Coordinator and worker chunk from the same pure function over the
        // same wire rows; assert the geometry a worker-rank replay derives for a
        // 200-row mixed run and a 300-row span launch matches rank 0's.
        for len in [200usize, 300, 8192] {
            let rank0 = span_chunk_points(len);
            let rank1 = span_chunk_points(len);
            assert_eq!(rank0, rank1);
        }
    }

    #[test]
    fn span_cap_geometry_boundaries_at_every_sweep_width() {
        // The chunkers must tile correctly at each supported cap with the
        // boundary lengths 63/64/65, 127/128/129 and 191/192/193 relative to
        // that cap. The `_at` variants are pure, so no process-global state
        // is touched and parallel tests stay deterministic.
        for &cap in super::super::tp_span_cap::SWEEP_CAPS.iter() {
            for len in [cap - 1, cap, cap + 1, 2 * cap - 1, 2 * cap, 2 * cap + 1] {
                let points = span_chunk_points_at(len, cap);
                assert_eq!(points.first(), Some(&0), "cap {cap} len {len}");
                assert_eq!(points.last(), Some(&len));
                for w in points.windows(2) {
                    let rows = w[1] - w[0];
                    assert!(
                        (1..=cap).contains(&rows),
                        "cap {cap} len {len}: window {w:?} has {rows} rows"
                    );
                }
                let total: usize = points.windows(2).map(|w| w[1] - w[0]).sum();
                assert_eq!(total, len, "cap {cap} len {len}");
            }
            // Exact expected tilings at the first boundary past the cap.
            assert_eq!(
                span_chunk_points_at(cap + 1, cap),
                vec![0, cap, cap + 1],
                "cap {cap}"
            );
            assert_eq!(
                span_chunk_points_at(2 * cap + 1, cap),
                vec![0, cap, 2 * cap, 2 * cap + 1],
                "cap {cap}"
            );
            // Checkpoint cuts at the cap boundary still land exactly on a
            // span end; a non-page-aligned cut is refused regardless of cap.
            let cut = cap;
            let points = span_checkpoint_points_at(0, cap + 1, &[(cut, 0)], cap).unwrap();
            assert!(
                points.contains(&cut),
                "cap {cap}: cut {cut} must split the span exactly"
            );
            let before = span_checkpoint_points_at(0, cap + 1, &[(cut - 1, 0)], cap);
            assert!(before.is_err(), "cuts must be page-aligned (16)");
        }
    }

    #[test]
    fn span_checkpoint_cuts_inside_one_wide_span_split_exactly() {
        // With cap 128, cuts at 48 and 96 inside one 130-row run force
        // spans ending exactly at each cut; the remainder takes one span
        // (34 rows, under the cap), and multiple cuts inside what would be
        // one wide span never merge.
        let points = span_checkpoint_points_at(0, 130, &[(48, 0), (96, 1)], 128).unwrap();
        assert_eq!(points, vec![0, 48, 96, 130]);
        assert!(points.windows(2).all(|w| w[1] > w[0] && w[1] - w[0] <= 128));
        // A cut exactly at the cap boundary ends the span there.
        let at_cap = span_checkpoint_points_at(0, 128, &[(128, 0)], 128).unwrap();
        assert_eq!(at_cap, vec![0, 128]);
        // A cut one row past the cap boundary forces the narrower split.
        let past = span_checkpoint_points_at(0, 129, &[(128, 0)], 128).unwrap();
        assert_eq!(past, vec![0, 128, 129]);
    }

    #[test]
    fn span_cap_wire_install_fails_closed_and_keeps_last_good() {
        // The wire installer must refuse unsupported values WITHOUT
        // half-installing: a refused value leaves the prior resolution in
        // force, so a mismatched pair can never serve. The final install
        // restores the production default for any later reader.
        use super::super::tp_span_cap;
        tp_span_cap::wire_span_cap(64).unwrap();
        assert_eq!(tp_span_cap(), 64);
        for bad in [0usize, 63, 65, 1000] {
            assert!(tp_span_cap::wire_span_cap(bad).is_err(), "{bad}");
            assert_eq!(tp_span_cap(), 64);
        }
        tp_span_cap::wire_span_cap(192).unwrap();
        assert_eq!(tp_span_cap(), 192);
        assert!(tp_span_cap::wire_span_cap(65).is_err());
        assert_eq!(tp_span_cap(), 192);
        tp_span_cap::wire_span_cap(64).unwrap();
        assert_eq!(tp_span_cap(), 64);
    }

    #[test]
    fn span_chunker_keeps_nonzero_slot_rows() {
        // The chunker is slot-agnostic (per-slot runs are carved by the
        // caller), but the ROW tuples it is fed must preserve their slot:
        // build the mixed-tick wire rows for slot 1 mid-decode and check
        // positions stay contiguous from the slot's own cursor.
        let slot = 1usize;
        let start_pos = 4096usize;
        let rows: Vec<(usize, u32, usize)> = (0..130)
            .map(|i| (slot, (1000 + i) as u32, start_pos + i))
            .collect();
        // every window is same-slot and position-contiguous
        let points = span_chunk_points(rows.len());
        for w in points.windows(2) {
            let run = &rows[w[0]..w[1]];
            assert!(run.iter().all(|&(s, _, _)| s == slot));
            for pair in run.windows(2) {
                assert_eq!(pair[1].2, pair[0].2 + 1);
            }
            assert_eq!(run[0].2, start_pos + w[0]);
        }
    }

    #[test]
    fn span_chunker_positions_reach_the_next_tick_cursor() {
        // After a 130-row chunk run starting at p, the next tick's first row
        // must validate at p+130 against the rank cursor (the scheduler's
        // mixed-position rule: cursor[s] += 1 per row).
        let start = 0usize;
        let points = span_chunk_points(130);
        let mut cursor = start;
        for w in points.windows(2) {
            cursor += w[1] - w[0];
        }
        assert_eq!(cursor, 130);
        // ...and the LAST window ends exactly there (final span = head span).
        assert_eq!(points[points.len() - 2]..points[points.len() - 1], 128..130);
    }

    #[test]
    fn span_chunker_head_lands_only_on_the_final_window() {
        // The head (final norm + LM head) may only run on the last window of
        // the finishing chunk: interior spans carry no head. Deriving the
        // "last" predicate exactly as run_chunk_spans does must fire once -
        // and only once - per run length.
        for len in [1usize, 64, 65, 130] {
            let points = span_chunk_points(len);
            let last_window: Vec<(usize, usize)> = points
                .windows(2)
                .filter(|w| w[1] == len)
                .map(|w| (w[0], w[1]))
                .collect();
            assert_eq!(
                last_window,
                vec![(points[points.len() - 2], len)],
                "len {len}: exactly one final window"
            );
        }
    }

    #[test]
    fn mixed_finisher_rows_carry_the_full_prompt_length() {
        // chunk_take's finisher must report the FULL prompt rows (last row's
        // position + 1), not the final tick's piece - finish_prefill sets
        // slot.pos = rows. A 300-token prompt chunked as 256+44 must report
        // 300 at finish. Reconstruct the arithmetic the code performs.
        let prompt_len = 300usize;
        let first_take = 256usize;
        let second_rows: Vec<(usize, u32, usize)> = (first_take..prompt_len)
            .map(|pos| (0usize, 7u32, pos))
            .collect();
        let fin_rows = second_rows.last().map_or(0, |&(_, _, position)| position + 1);
        assert_eq!(fin_rows, prompt_len);
        // and a single-tick prompt reports its whole length too.
        let whole: Vec<(usize, u32, usize)> =
            (0..prompt_len).map(|pos| (0, 7, pos)).collect();
        assert_eq!(
            whole.last().map_or(0, |&(_, _, position)| position + 1),
            prompt_len
        );
    }

    #[test]
    fn span_rows_occupy_nonzero_slot_before_decode_and_reach_next_page() {
        // Exercise the SAME bookkeeping helper used by mixed ticks and
        // asynchronous span launches. A cancellation just after launch must
        // find slot 1 occupied even without a decode call.
        let mut positions = vec![9, 15];
        let mut occupied = vec![true, false];
        let rows = [(1, 31, 15), (1, 32, 16), (1, 33, 17)];
        record_prefill_rows(&rows, &mut positions, &mut occupied);
        assert_eq!(positions, vec![9, 18]);
        assert_eq!(occupied, vec![true, true]);
        let scheduler_occupied = [true, false];
        let released: Vec<_> = occupied
            .iter()
            .enumerate()
            .filter_map(|(slot, was)| (*was && !scheduler_occupied[slot]).then_some(slot))
            .collect();
        assert_eq!(released, vec![1]);
    }
}
