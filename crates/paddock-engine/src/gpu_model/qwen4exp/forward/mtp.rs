//! MTP drafting for Flash-Next: the separate head GGUF, its forward pass, and
//! the per-slot bookkeeping that keeps the head's cache in step with the trunk.
//!
//! The head (unsloth's `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) is one
//! ordinary full-attention qwen4exp block plus the nextn glue. Head row `p`
//! consumes the pair (the trunk's PRE-final-mixer 4-stream row at `p`, the
//! token at `p + 1`) and proposes the token at `p + 2`:
//!
//!   e     = fc_embedding(enorm(embed(tok[p+1])))     once per token
//!   h_s   = fc_hidden(hnorm(stream[p])_s)            per stream s
//!   x_s   = h_s + e                                  the head's 4-stream row
//!   x     = block_48(x)                              writes head KV row p
//!   draft = argmax(lm_head(head_mixer(x)))           the TARGET's lm_head
//!
//! A chain re-enters on its own output: step i reads the block output of step
//! i-1 in place of the trunk row it does not have. That shape is ds4's
//! (`ds4_qwen4exp_mtp.c` in the cudafast track engine, MIT) - studied for the
//! technique, none of its code is here. One reading is taken on its word:
//! `hnorm` is a single statistic over the whole stream row, not grouped.
//!
//! Bookkeeping. Per slot the head has written rows for positions `[base, end)`
//! and keeps the trunk rows for `[end, pos)` in a small stash - their next
//! token is not all known yet (the newest one's never is). While a slot is
//! tracked, `end + stash_n == pos`, and `stash_n >= 1` after any walk. A draft
//! feeds the whole stash plus the pending token in one head pass (the catch-up
//! rows and chain step 0 together), so the head's cache holds TRUNK rows for
//! every committed position and chain rows only above the frontier, where the
//! next feed rewrites them before anything reads them.
//!
//! `base` is where the head's cache starts: 0 for a fresh prompt, the resume
//! point on a prefix-cache hit (the prefix cache does not carry the head's KV).
//! The head indexes its cache and its RoPE by `p - base`; rotary attention sees
//! distances only, so that is exact over the rows the head has - it just does
//! not see the resumed prefix. That costs draft quality, never correctness:
//! the verify judges every token.

use std::path::Path;

use super::*;
use crate::gpu::GpuError;

/// Deepest chain a round drafts. A verify chunk (drafts + the pending token)
/// must fit the 9-row PLE ring for the rollback in spec.rs to be exact.
pub(crate) const MTP_MAX_DRAFT: usize = 7;
/// Trunk rows a slot can hold before the head has consumed them.
const STASH_CAP: usize = 16;
/// Stash depth at which dense-tick rows are fed to the head.
const FLUSH_AT: usize = 8;
/// Widest head pass while seeding a prompt (sizes the sorted-MoE scratch).
const SEED_CHUNK: usize = 256;

pub(crate) struct Mtp {
    w: crate::gpu_model::qwen4exp::load_gguf::MtpWeights,
    /// the head block's KV, in the trunk's class: [slots, max_tokens, kv_dim]
    kv_k: CudaSlice<u8>,
    kv_v: CudaSlice<u8>,
    tracked: Vec<bool>,
    base: Vec<usize>,
    end: Vec<usize>,
    stash_n: Vec<usize>,
    /// trunk stream rows not yet fed: [slots, STASH_CAP, hc_width]
    stash: CudaSlice<f32>,
    /// the chain's carry: the head block's output row from the last step
    multi: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    pick: CudaSlice<u32>,
    /// all zeros: the inject that makes the hyper-connection combine a plain
    /// add (its weight is 2 * sigmoid(inj / hc), exactly 1.0 at 0)
    zero_inj: CudaSlice<f32>,
    wide: Q8Wide,
    seed_chunk: usize,
    /// Draft vocabulary shortlist: the head's pick is taken over ids
    /// `[0, draft_prefix)` and the added-token tail `[vocab - draft_tail,
    /// vocab)` only, reading just those lm_head rows. 0 = the whole vocab.
    /// BPE ids are merge-ordered, so the low ids are the frequent tokens; the
    /// tail comes from the file's own token types. The draft only proposes -
    /// the verify's full-vocab argmax decides every token - so a shortlist can
    /// cost acceptance, never correctness.
    draft_prefix: usize,
    draft_tail: usize,
    /// Scheduler hints (`Generator::spec_fuse_hint` / `spec_warm_hint`): is
    /// speculation the plan for the current width at all? Feeding this head is
    /// not free (every decode row is stashed, and every drain runs a head
    /// pass), so on a tick no round will consume, that work is pure loss.
    /// Measured at 8 live slots (2026-09-17): a k=0 tick with the head fed
    /// took 45.6 ms against ~26 ms with no head attached, which is the whole
    /// c8 margin.
    ///
    /// Default true: a backend must behave as before until the scheduler says
    /// otherwise, and the hints arrive once per tick.
    pub(super) feed: bool,
    pub(super) warm_prefills: bool,
    /// Canonical rejection sampling (PADDOCK_SPEC_RS). `None` when the arm is
    /// off, which is the default and costs nothing - the buffers below are the
    /// arm, exactly as gemma4 does it.
    pub(crate) rs: Option<RsBufs>,
    /// True for the duration of one chain that is drafting under RS: the round
    /// had draws for this slot and the buffers are primed. Per round, not per
    /// attach, because a greedy slot in an otherwise sampled tick still draws
    /// inv_t 0 and must take the argmax path.
    #[allow(dead_code)]
    pub(crate) rs_active: bool,
}

/// Canonical-RS device state for the Flash-Next head.
///
/// Why this exists: the chain's picks are device argmaxes, so a slot sampling
/// at temperature > 0 cannot take them without changing the emitted
/// distribution - which is why `service::run_batched`'s greedy gate sends every
/// sampled request down the dense path, and why this family measured 29.3 tok/s
/// on real traffic against 35+ on a greedy benchmark (2026-09-19). The fix is
/// the canonical rule: draft sampled from the drafter softmax q, accept with
/// probability min(1, p/q), recover from the residual. Same emitted
/// distribution as sampling without a drafter - lossless, not an approximation.
///
/// The q-store holds the fp16 exp mass of each chain row so the verify's
/// resolve can read q back for the row it is accepting against; `qsum` carries
/// the exact f32 normalizer, and 0 there marks a greedy (argmax) row so mixed
/// greedy/sampled rounds resolve correctly in one pass.
// Fields are read by the batched chain and the verify resolve, which land with
// the async round (see the module note on `rs_active`); allow until then so the
// arm can be reviewed and built in isolation.
#[allow(dead_code)]
pub(crate) struct RsBufs {
    /// [MTP_MAX_DRAFT, rows, vocab] fp16 exp mass: q of each chain row
    pub(crate) qstore: CudaSlice<u16>,
    /// exact f32 sums of the stored rows; 0 marks a greedy (argmax) row
    pub(crate) qsum: CudaSlice<f32>,
    /// per-chain-row 1/T for the draft draw (0 = argmax)
    pub(crate) invt: CudaSlice<f32>,
    /// [k, rows] per-step draft-draw uniforms, from the slot's own seed stream
    pub(crate) uplane: CudaSlice<f32>,
    /// device chain-step counter (reset per round, +1 per step)
    pub(crate) step: CudaSlice<u32>,
    /// verify-resolve params, 8 u32 words per drafted row
    pub(crate) par: CudaSlice<u32>,
}

/// Drafted-row ceiling the RS resolve param buffer is sized for. The service's
/// row budget is 32 on this family and a chunk is at most `VERIFY_MAX_CHUNK`,
/// so 256 is an order of magnitude of headroom.
#[allow(dead_code)]
pub(crate) const RS_PAR_ROWS: usize = 256;

/// PADDOCK_SPEC_RS=1 arms canonical rejection sampling. Off by default: the
/// arm is new on this arch (2026-09-19) and has not been through a
/// distribution-equivalence gate here, so it must be asked for by name.
#[allow(dead_code)]
pub(crate) fn spec_rs_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_RS").is_some())
}

impl Qwen4ExpGpu {
    /// Attach a sideloaded MTP head GGUF (`--mtp`). The GGUF lane needs this
    /// because unsloth's UD export strips the in-file block; a safetensors
    /// checkpoint that carries its own takes `attach_mtp_in_file` instead.
    pub fn attach_mtp(&mut self, path: &Path) -> Result<(), GpuModelError> {
        if !matches!(self.st, PleSource::Gguf { .. }) {
            return Err(GpuModelError::Unsupported(
                "a sideloaded qwen4exp MTP head attaches to the GGUF lane only; the \
                 safetensors lane carries its own under mtp.*"
                    .into(),
            ));
        }
        Self::mtp_precheck(&self.exec)?;
        let map = MappedGguf::open(path)
            .map_err(|e| GpuModelError::Unsupported(format!("MTP head {}: {e}", path.display())))?;
        let before = self.exec.settled_mem_used();
        let w = crate::gpu_model::qwen4exp::load_gguf::load_mtp(&self.exec, &map, &self.cfg)?;
        drop(map);
        self.attach_mtp_weights(w, before, &path.display().to_string())
    }

    /// Attach the head the checkpoint ships (`mtp.*`), safetensors lane.
    ///
    /// Called at load when the config declares one, so speculation is simply
    /// available on such a checkpoint - no `--mtp`, nothing for an operator to
    /// know. Before this the weights loaded not at all and the serve decoded
    /// dense with a resident-but-unused drafter, which is exactly the silent
    /// class the load-time warning in `service::run_batched` now names.
    pub fn attach_mtp_in_file(&mut self) -> Result<(), GpuModelError> {
        let PleSource::St(st) = &self.st else {
            return Err(GpuModelError::Unsupported(
                "the in-file qwen4exp MTP head is a safetensors-lane feature".into(),
            ));
        };
        Self::mtp_precheck(&self.exec)?;
        let before = self.exec.settled_mem_used();
        let w = crate::gpu_model::qwen4exp::load::load_mtp_st(&self.exec, st, &self.cfg)?;
        self.attach_mtp_weights(w, before, "in-file (mtp.*)")
    }

    /// Does this checkpoint carry its own MTP block? `text_config.mtp_num_
    /// hidden_layers` is the declaration; the GGUF lane always answers false
    /// because the export it reads has the block stripped.
    pub fn has_in_file_mtp(&self) -> bool {
        matches!(self.st, PleSource::St(_)) && self.cfg.mtp_layers > 0
    }

    /// The pack entries a draft chain cannot run without.
    fn mtp_precheck(exec: &Arc<GpuExecutor>) -> Result<(), GpuModelError> {
        if !exec.has_argmax_rows() {
            return Err(GpuModelError::Unsupported(
                "kernel pack has no argmax_rows - the draft chain needs it".into(),
            ));
        }
        Ok(())
    }

    /// Seat a loaded head: its KV, the chain scratch and the draft shortlist.
    /// Shared by both lanes so a head is the same resident object however it
    /// arrived.
    fn attach_mtp_weights(
        &mut self,
        w: crate::gpu_model::qwen4exp::load_gguf::MtpWeights,
        before: Option<u64>,
        src: &str,
    ) -> Result<(), GpuModelError> {
        let (c, e) = (&self.cfg, &self.exec);
        let (slots, t) = (self.slots, self.max_tokens);
        let kv_bytes = slots * t * c.n_kv_heads * c.head_dim * KV().bytes();
        let seed = SEED_CHUNK.min(t);
        // moe_align's bound: every expert rounds its pairs up to one block
        let nb = seed * c.n_active / 32 + c.n_expert;
        let wide = Q8Wide {
            srow: e.alloc_u32(nb * 32)?,
            sslot: e.alloc_u32(nb * 32)?,
            bexp: e.alloc_u32(nb)?,
            fused: e.alloc(nb * 32 * c.moe_ff)?,
            fq: e.alloc_i8(nb * 32 * c.moe_ff)?,
            fs: e.alloc(nb * c.moe_ff)?,
            part: e.alloc(seed * c.n_active * c.hidden)?,
            max_rows: seed,
            max_blocks: nb,
        };
        // The added-token tail: the trailing run of non-normal token types
        // (control / user-defined) above the BPE ids - 276 on this tokenizer.
        let draft_tail = match &self.st {
            PleSource::Gguf { map, .. } => {
                match map.gguf().metadata.get("tokenizer.ggml.token_type") {
                    Some(paddock_models::gguf::Value::Array(tt)) => tt
                        .iter()
                        .rev()
                        .take_while(|t| t.as_u64() != Some(1))
                        .count(),
                    _ => 0,
                }
            }
            PleSource::St(_) => 0,
        };
        // PADDOCK_Q38FN_DRAFT_VOCAB=<prefix ids>: a development switch until
        // the acceptance cost of a cut is measured; unset = whole vocabulary.
        let draft_prefix = std::env::var("PADDOCK_Q38FN_DRAFT_VOCAB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&p| p > 0 && p + draft_tail < c.vocab)
            .unwrap_or(0);
        if draft_prefix > 0 {
            eprintln!(
                "[q4x-mtp] draft vocabulary shortlist: ids [0, {draft_prefix}) + tail {draft_tail} of {}",
                c.vocab
            );
        }
        let m = Mtp {
            w,
            kv_k: e.alloc_u8(kv_bytes)?,
            kv_v: e.alloc_u8(kv_bytes)?,
            tracked: vec![false; slots],
            base: vec![0; slots],
            end: vec![0; slots],
            stash_n: vec![0; slots],
            stash: e.alloc(slots * STASH_CAP * c.hc_width())?,
            multi: e.alloc(c.hc_width())?,
            logits: e.alloc(c.vocab)?,
            pick: e.alloc_u32(1)?,
            zero_inj: e.alloc(seed.max(STASH_CAP) * c.hc_count)?,
            wide,
            seed_chunk: seed,
            draft_prefix,
            draft_tail,
            feed: true,
            warm_prefills: true,
            rs_active: false,
            // Parked, deliberately, and it must stay parked until the verify
            // resolve below exists.
            //
            // `supports_spec_rs` answers from this field, and the service
            // stashes per-slot chain draws exactly when that answers true. A
            // backend that claimed the arm without a `spec_rs_resolve` in its
            // verify would take sampled rounds it resolves with the greedy
            // accept-while-match rule - silently emitting the wrong
            // distribution, which is worse than declining.
            //
            // It also turned out not to be the lever this family needs first.
            // `pd_draft_rs`/`pd_spec_rs_resolve` buy the broad-distribution
            // headroom (sum min(p,q) vs p(argmax q)) and need sampled drafts
            // with a real q, which forces the chain to be batched across slots
            // (the kernel is one CTA per chain row, q at
            // `[step*rmax + row]`). But the resolve's own `qsum == 0` arm is
            // the point-mass rule, and `service`'s Phase 2b-dev already runs
            // exact rejection sampling against deterministic argmax drafts
            // through `forward_spec_batch_plans` - which is what actually
            // unblocks temperature > 0 traffic here, at a fraction of the work.
            // So: plans path first, canonical RS after, and this stays None.
            rs: None,
        };
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let held = match (before, self.exec.settled_mem_used()) {
            (Some(b), Some(a)) => a.saturating_sub(b),
            _ => 0,
        };
        tracing::info!(
            source = %src,
            resident_gib = gib(held),
            head_kv_gib = gib(2 * kv_bytes as u64),
            "qwen4exp MTP head attached"
        );
        eprintln!(
            "[q4x-mtp] head attached ({}): {:.2} GiB resident incl. {:.2} GiB head KV",
            src,
            gib(held),
            gib(2 * kv_bytes as u64)
        );
        self.mtp = Some(Box::new(m));
        Ok(())
    }

    /// A prompt is about to walk into `slot` from `start` (0 = fresh).
    pub(super) fn mtp_begin(&mut self, slot: usize, start: usize) {
        if let Some(m) = self.mtp.as_mut() {
            m.tracked[slot] = true;
            m.base[slot] = start;
            m.end[slot] = start;
            m.stash_n[slot] = 0;
        }
    }

    pub(super) fn mtp_clear(&mut self, slot: usize) {
        if let Some(m) = self.mtp.as_mut() {
            m.tracked[slot] = false;
            m.stash_n[slot] = 0;
        }
    }

    /// Can `slot` draft right now, with the model at `pos`?
    pub(super) fn mtp_warm(&self, slot: usize, pos: usize) -> bool {
        self.pos[slot] == pos
            && self.mtp.as_ref().is_some_and(|m| {
                m.tracked[slot]
                    && m.stash_n[slot] >= 1
                    && m.end[slot] + m.stash_n[slot] == self.pos[slot]
            })
    }

    fn mtp_cold(m: &mut Mtp, slot: usize, why: &str) {
        if m.tracked[slot] {
            m.tracked[slot] = false;
            tracing::debug!(slot, why, "qwen4exp MTP: slot went cold");
        }
    }

    /// A prefill span just walked rows for positions `from..to` of `slot`;
    /// their streams sit in `d_h` at `row_off..`. Feed every row whose next
    /// token the prompt holds and stash the one it does not.
    pub(super) fn mtp_seed(
        &mut self,
        slot: usize,
        row_off: usize,
        from: usize,
        to: usize,
        ids: &[u32],
    ) -> Result<(), GpuModelError> {
        let hw = self.cfg.hc_width();
        let (known, seed) = {
            let Some(m) = self.mtp.as_mut() else {
                return Ok(());
            };
            // The eager warm the scheduler's `spec_warm_hint` governs: seeding
            // a prompt whose width will not speculate is prefill cost for a
            // round that never comes.
            if !m.warm_prefills || !m.tracked[slot] || to <= from {
                return Ok(());
            }
            if m.stash_n[slot] != 0 || m.end[slot] != from {
                Self::mtp_cold(m, slot, "prefill span out of step");
                return Ok(());
            }
            let n = to - from;
            let known = if to < ids.len() { n } else { n - 1 };
            if known < n {
                // before any head pass: a pass rewrites d_h from row 0
                self.exec.copy_region(
                    &self.sc.d_h,
                    (row_off + n - 1) * hw,
                    &mut m.stash,
                    slot * STASH_CAP * hw,
                    hw,
                )?;
            }
            (known, m.seed_chunk)
        };
        let mut done = 0;
        while done < known {
            let len = (known - done).min(seed);
            // This chunk's rows are still intact: each earlier pass wrote d_h
            // rows [0, its own len), all below this chunk's offset.
            self.exec.copy_region(
                &self.sc.d_h,
                (row_off + done) * hw,
                &mut self.sc.d_gate,
                0,
                len * hw,
            )?;
            self.mtp_pass(
                slot,
                from + done,
                &ids[from + done + 1..from + done + len + 1],
                false,
            )?;
            done += len;
        }
        let m = self.mtp.as_mut().expect("tracked above");
        m.end[slot] = from + known;
        m.stash_n[slot] = to - from - known;
        Ok(())
    }

    /// A decode tick advanced `rows` (row j of `d_h` is `rows[j]`'s slot).
    pub(super) fn mtp_note_rows(&mut self, rows: &[(usize, u32)]) -> Result<(), GpuModelError> {
        let hw = self.cfg.hc_width();
        let Some(m) = self.mtp.as_mut() else {
            return Ok(());
        };
        // Not speculating at this width: stop stashing. The slots then fall
        // out of step with their stash and `mtp_note_rows` marks them cold on
        // the tick feeding resumes, which is the same self-healing path a
        // dense interlude already used - a cold slot drafts nothing until its
        // next prompt seeds it, and that is the documented trade for the ticks
        // this saves (see the `feed` field).
        if !m.feed {
            return Ok(());
        }
        for (j, &(sl, _)) in rows.iter().enumerate() {
            if !m.tracked[sl] {
                continue;
            }
            let p = self.pos[sl] - 1;
            if m.end[sl] + m.stash_n[sl] != p || m.stash_n[sl] >= STASH_CAP {
                Self::mtp_cold(m, sl, "decode row out of step");
                continue;
            }
            self.exec.copy_region(
                &self.sc.d_h,
                j * hw,
                &mut m.stash,
                (sl * STASH_CAP + m.stash_n[sl]) * hw,
                hw,
            )?;
            m.stash_n[sl] += 1;
        }
        Ok(())
    }

    /// A verify round committed `c` rows of `slot` from position `pos`; their
    /// streams sit in `d_h` at `off..`.
    pub(super) fn mtp_note_verify(
        &mut self,
        slot: usize,
        off: usize,
        pos: usize,
        c: usize,
    ) -> Result<(), GpuModelError> {
        let hw = self.cfg.hc_width();
        let Some(m) = self.mtp.as_mut() else {
            return Ok(());
        };
        if !m.tracked[slot] {
            return Ok(());
        }
        if m.end[slot] + m.stash_n[slot] != pos || m.stash_n[slot] + c > STASH_CAP {
            Self::mtp_cold(m, slot, "verify rows out of step");
            return Ok(());
        }
        self.exec.copy_region(
            &self.sc.d_h,
            off * hw,
            &mut m.stash,
            (slot * STASH_CAP + m.stash_n[slot]) * hw,
            c * hw,
        )?;
        m.stash_n[slot] += c;
        Ok(())
    }

    /// Feed the stash of every slot that dense ticks have filled past
    /// `FLUSH_AT`. Reuses the walk's scratch: call it after a tick's outputs
    /// are read.
    pub(super) fn mtp_flush(&mut self) -> Result<(), GpuModelError> {
        let due: Vec<usize> = match self.mtp.as_ref() {
            // the drain is where the head PASS happens - the expensive half of
            // feeding, and the half a non-speculating width must not pay
            Some(m) if !m.feed => return Ok(()),
            Some(m) => (0..self.slots)
                .filter(|&s| m.tracked[s] && m.stash_n[s] >= FLUSH_AT)
                .collect(),
            None => return Ok(()),
        };
        for slot in due {
            self.mtp_drain(slot)?;
        }
        Ok(())
    }

    /// Feed every stashed row but the newest (its next token is not known).
    fn mtp_drain(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let hw = self.cfg.hc_width();
        let (end, f) = {
            let m = self.mtp.as_ref().expect("flush checked");
            (m.end[slot], m.stash_n[slot] - 1)
        };
        if f == 0 {
            return Ok(());
        }
        if self.stream[slot].len() < 2 + end + f + 1 {
            let m = self.mtp.as_mut().expect("flush checked");
            Self::mtp_cold(m, slot, "token stream behind the stash");
            return Ok(());
        }
        let toks: Vec<u32> = (end + 1..=end + f)
            .map(|p| self.stream[slot][2 + p] as u32)
            .collect();
        {
            let m = self.mtp.as_mut().expect("flush checked");
            let Mtp { stash, multi, .. } = &mut **m;
            let at = slot * STASH_CAP * hw;
            self.exec
                .copy_region(stash, at, &mut self.sc.d_gate, 0, f * hw)?;
            // the newest row moves to the front, through the chain carry (no
            // chain is in flight between ticks)
            self.exec.copy_region(stash, at + f * hw, multi, 0, hw)?;
            self.exec.copy_region(multi, 0, stash, at, hw)?;
        }
        self.mtp_pass(slot, end, &toks, false)?;
        let m = self.mtp.as_mut().expect("flush checked");
        m.end[slot] = end + f;
        m.stash_n[slot] = 1;
        Ok(())
    }

    /// Up to `k` drafts for `slot`, whose next token (not yet walked) is
    /// `pending` at position `pos`. Empty when the slot is not warm.
    pub(super) fn mtp_draft(
        &mut self,
        slot: usize,
        pending: u32,
        k: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        let pos = self.pos[slot];
        if !self.mtp_warm(slot, pos) {
            return Ok(Vec::new());
        }
        let k = k
            .min(MTP_MAX_DRAFT)
            .min(self.max_tokens.saturating_sub(pos + 1));
        if k == 0 {
            return Ok(Vec::new());
        }
        let hw = self.cfg.hc_width();
        let (end, sn) = {
            let m = self.mtp.as_ref().expect("warm");
            (m.end[slot], m.stash_n[slot])
        };
        // the stash rows' next tokens: committed ones, then the pending token
        let mut toks: Vec<u32> = (end + 1..pos)
            .map(|p| self.stream[slot][2 + p] as u32)
            .collect();
        toks.push(pending);
        {
            let m = self.mtp.as_ref().expect("warm");
            self.exec.copy_region(
                &m.stash,
                slot * STASH_CAP * hw,
                &mut self.sc.d_gate,
                0,
                sn * hw,
            )?;
        }
        let none = || GpuModelError::Unsupported("MTP draft pass produced no token".into());
        // each pass ends in its pick's readback, so host clocks here are
        // already synchronized - no extra sync needed to time them
        let timing = std::env::var_os("PADDOCK_Q38FN_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let mut drafts = Vec::with_capacity(k);
        drafts.push(self.mtp_pass(slot, end, &toks, true)?.ok_or_else(none)?);
        let d_feed = t0.elapsed();
        {
            let m = self.mtp.as_mut().expect("warm");
            m.end[slot] = pos;
            m.stash_n[slot] = 0;
        }
        for i in 1..k {
            {
                let m = self.mtp.as_ref().expect("warm");
                self.exec
                    .copy_region(&m.multi, 0, &mut self.sc.d_gate, 0, hw)?;
            }
            let prev = drafts[i - 1];
            drafts.push(
                self.mtp_pass(slot, pos - 1 + i, &[prev], true)?
                    .ok_or_else(none)?,
            );
        }
        if timing {
            let d_all = t0.elapsed();
            eprintln!(
                "[spec-draft] slot {slot} feed {sn} rows {:7.2} ms | chain {} steps {:7.2} ms",
                d_feed.as_secs_f64() * 1e3,
                k - 1,
                (d_all - d_feed).as_secs_f64() * 1e3
            );
        }
        Ok(drafts)
    }

    /// One head pass over `toks.len()` rows of `slot` at positions `p0..`:
    /// row i pairs the stream row the caller staged at `d_gate[i]` with
    /// `toks[i]`, the token at `p0 + i + 1`. Writes head KV. With `draft`,
    /// returns the head's pick after the last row and leaves that row's block
    /// output in `multi` for the next chain step.
    ///
    /// Reuses the walk's scratch wholesale (the head is a full qwen4exp
    /// block): never call it while a walk's outputs are still unread.
    fn mtp_pass(
        &mut self,
        slot: usize,
        p0: usize,
        toks: &[u32],
        draft: bool,
    ) -> Result<Option<u32>, GpuModelError> {
        let n = toks.len();
        let Self {
            exec: e,
            cfg: c,
            embed,
            lm_head,
            max_tokens,
            sc,
            stage,
            mtp,
            ..
        } = self;
        let m = mtp.as_mut().expect("MTP head attached");
        let (h, hw, hc, lr, eps) = (c.hidden, c.hc_width(), c.hc_count, c.hc_lowrank, c.eps);
        if n == 0 || p0 < m.base[slot] || p0 - m.base[slot] + n > *max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "MTP head pass: {n} rows at position {p0} (cache base {}, {} rows)",
                m.base[slot], *max_tokens
            )));
        }
        let hp0 = p0 - m.base[slot];
        let pos: Vec<u32> = (hp0 as u32..(hp0 + n) as u32).collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        e.upload_u32(toks, &mut sc.d_tok)?;
        e.upload_u32(&pos, &mut sc.d_pos)?;
        e.upload_u32(&mrope, &mut sc.d_mrope)?;
        e.upload_u32(&vec![slot as u32; n], &mut sc.d_slots)?;
        let Mtp {
            w,
            kv_k,
            kv_v,
            multi,
            logits,
            pick,
            zero_inj,
            wide,
            draft_prefix,
            draft_tail,
            ..
        } = &mut **m;

        // e = fc_embedding(enorm(embed(tok))), one row per token -> d_mix
        match embed {
            Embed::Bf16(t) => e.embed_gather_bf16(t, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?,
            Embed::Kq(t) => e.kquant_gather(t, &sc.d_tok, &mut sc.d_x, h, n)?,
            Embed::Q8(t) => e.embed_gather_q8(t, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?,
        }
        e.rmsnorm_batch(&sc.d_x, &w.enorm.buf, &mut sc.d_bi, h, eps, n)?;
        w.eh_e.matmul(e, &sc.d_bi, &mut sc.d_mix, n, stage)?;
        // h_s = fc_hidden(hnorm(stream)) over the four streams as plain rows,
        // straight into the stream layout [n, hc, hidden] of d_h
        e.rmsnorm_batch(&sc.d_gate, &w.hnorm.buf, &mut sc.d_xn, hw, eps, n)?;
        w.eh_h.matmul(e, &sc.d_xn, &mut sc.d_h, n * hc, stage)?;
        // x_s = h_s + e: the combine at weight exactly 1
        e.q4x_hc_combine(&mut sc.d_h, &sc.d_mix, zero_inj, n, hc, h)?;

        // the block - the trunk's own sub-passes, on the head's weights and KV.
        // The stage flags are per walk; the head's planes are all Kq, which
        // reads none of them, so they go to their inert values.
        stage.f16_ok = false;
        stage.lowm_ok = false;
        stage.f16_max = usize::MAX;
        stage.row_exact = false;
        stage.prefill = false;
        let l = &w.layer;
        let MixerW::Attn(aw) = &l.mixer else {
            return Err(GpuModelError::Unsupported(
                "MTP head block is not an attention block".into(),
            ));
        };
        // one row takes the slot-vector decode attention; a span of one slot
        // takes the prefill attention at its rows' positions
        let phase = if n == 1 {
            Phase::DecodeBatch
        } else {
            Phase::Prefill
        };
        let attn_inj = hc_mix_pass(e, c, &l.attn_hc, sc, stage, n, false, false, None)?;
        attn_pass(
            e,
            c,
            aw,
            sc,
            stage,
            kv_k,
            kv_v,
            None, // the MTP head's attention is dense: it drafts, the model verifies
            false,
            *max_tokens,
            n,
            phase,
            &[],
            0,
            false,
        )?;
        let (mlp_pre, _, _) = combine(
            e,
            sc,
            attn_inj,
            Some(&l.mlp_hc.norm),
            None,
            None,
            n,
            hc,
            h,
            eps,
            stage,
        )?;
        let mlp_inj = hc_mix_pass(e, c, &l.mlp_hc, sc, stage, n, mlp_pre, false, None)?;
        moe_pass(e, c, &l.moe, sc, stage, n, false, n == 1, Some(wide))?;
        let (normed, _, _) = combine(
            e,
            sc,
            mlp_inj,
            draft.then_some(&w.head_mix.norm),
            None,
            None,
            n,
            hc,
            h,
            eps,
            stage,
        )?;
        if !draft {
            return Ok(None);
        }
        debug_assert!(normed, "a combine with a next norm leaves it in d_xn");

        // the chain carry, then the head mixer + lm_head over the last row
        // only (its normalized state -> d_pkn, which the head never uses)
        e.copy_region(&sc.d_h, (n - 1) * hw, multi, 0, hw)?;
        e.copy_region(&sc.d_xn, (n - 1) * hw, &mut sc.d_pkn, 0, hw)?;
        w.head_mix
            .down
            .matmul(e, &sc.d_pkn, &mut sc.d_m, 1, stage)?;
        e.q4x_scale_silu(&mut sc.d_m, lr, 1.0 / hc as f32)?;
        w.head_mix.up.matmul(e, &sc.d_m, &mut sc.d_gate, 1, stage)?;
        e.q4x_hc_mix(&sc.d_pkn, &sc.d_gate, &mut sc.d_bi, None, 1, hc, h)?;
        // The shortlisted head: the batch-1 W4A8 GEMV the full head runs
        // (same quantize, same kernel, so every read row's logit is the full
        // launch's bit for bit), over the two id ranges only.
        let (prefix, tail) = (*draft_prefix, *draft_tail);
        let short = match lm_head {
            DensePlane::Kq {
                w: crate::gpu::QuantW::Kq(k),
                in_dim,
                ..
            } if prefix > 0 && !crate::gpu::kq_is_iq(k.ty) && e.has_kquant_gemv_w4a8() => {
                e.quantize_q8_sums(
                    &sc.d_bi,
                    &mut stage.q,
                    &mut stage.xs,
                    &mut stage.ssums,
                    *in_dim,
                )?;
                let sums = crate::gpu::kq_needs_sums(k.ty).then_some(&stage.ssums);
                e.kquant_gemv_w4a8_rows(k, 0, prefix, &stage.q, &stage.xs, sums, logits, 0)?;
                if tail > 0 {
                    e.kquant_gemv_w4a8_rows(
                        k,
                        c.vocab - tail,
                        tail,
                        &stage.q,
                        &stage.xs,
                        sums,
                        logits,
                        prefix,
                    )?;
                }
                e.argmax_rows(logits, pick, 1, prefix + tail)?;
                true
            }
            // Every OTHER dense class (the safetensors lanes: bf16, its f16
            // twin, f8row, MXFP8) takes the same shortlist through
            // `matmul_rows`, which is a ROW PREFIX of the plane and therefore
            // exactly the same bytes the k-quant arm above reads. Until this
            // existed the shortlist fired on the GGUF lane only and every
            // safetensors draft read the whole 248320-row head - 1.18 GB of
            // bf16 a pass, three passes a round, 26% of the round's bytes on
            // a lane whose drafts are otherwise cheap.
            //
            // Prefix only: `matmul_rows` lands its output at 0, so the added
            // -token tail would need a second landing offset the k-quant arm
            // gets for free. The prefix is the part that matters (BPE ids are
            // merge-ordered, so the low ids ARE the frequent tokens) and a
            // missing tail costs acceptance, never correctness.
            // ...but only the classes whose `matmul_rows` IS a row prefix.
            // F8Row holds a per-row-scaled plane with no row-offset entry and
            // answers Unsupported, and a draft pass that errors does not fail
            // the serve - the drafter just returns nothing and speculation
            // silently dies. Measured: f8row + this shortlist ran 8800 rounds
            // at 0.00 drafted, 1.00 tokens a round, 23.8 tok/s against 46.9,
            // and the only sign was "no folded planes" in the serve log. So
            // the class check is here rather than at the call, and an
            // unsupported class keeps the FULL-vocab head.
            _ if prefix > 0
                && tail == 0
                && matches!(
                    lm_head,
                    DensePlane::Bf16(_)
                        | DensePlane::Dual { .. }
                        | DensePlane::F16 { .. }
                        | DensePlane::Mxf8 { .. }
                ) =>
            {
                lm_head.matmul_rows(e, 0, prefix, &sc.d_bi, logits, 1, stage)?;
                e.argmax_rows(logits, pick, 1, prefix)?;
                true
            }
            _ => {
                lm_head.matmul(e, &sc.d_bi, logits, 1, stage)?;
                e.argmax_rows(logits, pick, 1, c.vocab)?;
                false
            }
        };
        let view = pick
            .try_slice(0..1)
            .ok_or_else(|| GpuError::Driver("MTP pick view".into()))?;
        let ids = e
            .stream
            .clone_dtoh(&view)
            .map_err(crate::gpu::from_driver)?;
        // a shortlisted pick is a packed position: the prefix is the id, the
        // tail sits behind it (packing prefix-first keeps id order, so the
        // argmax's lowest-index tie rule still picks the lowest id)
        let id = if short && ids[0] as usize >= prefix {
            (c.vocab - tail + (ids[0] as usize - prefix)) as u32
        } else {
            ids[0]
        };
        Ok(Some(id))
    }
}
