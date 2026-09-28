//! Speculative-decoding core for nemotron: the TRUNK verify
//! round over ragged per-slot chunks, with the mamba-2 state discipline the
//! qwen35 lane established - the recurrence is not idempotent, so the
//! verify advances the live SSM state while snapshotting every row's state
//! (`pd_mamba2_scan_seq_snap`), runs the conv on a per-slot SCRATCH window
//! (the live window stays pre-round), and snapshots the conv-input rows.
//! The commit (inside the same call - the greedy round reads its own picks)
//! re-derives the accepted count with service.rs's exact walk and rolls
//! partially-accepted slots back: state <- snap[accepted-1], window <- the
//! last k-1 conv-input rows of [pre-round window ∥ accepted rows]. KV needs
//! no rollback (stale cells past the accept are overwritten before any
//! later read). Attention/MoE/head rows ride the batch walk's own r>1
//! classes unchanged.
//!
//! Drafters attach on top: DFlash (the official nvidia checkpoint) and the
//! in-file MTP block - both consume this verify. Until one is attached,
//! `spec_capable` stays false and nothing here runs in serving.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{prefill_mm_pre_any, prefill_quant};

use super::batch::PfCuts;
use super::ssm_arena::SsmArena;
use super::*;
use paddock_models::nemotron::NemotronBlock;

/// Verify-round row budget: bounds the per-row state snapshots (each row
/// costs a full [H, hd, S] f32 state per mamba layer - ~2 MiB - so 32 rows
/// across 23 layers is ~1.5 GiB, allocated lazily at first spec use).
pub(crate) const SPEC_ROWS_NEMO: usize = 32;

/// Lazily-allocated verify planes (lives inside `NemoBatch`).
pub(crate) struct VerifyPlanes {
    /// per-mamba-layer per-row state snapshots [SPEC_ROWS, H*hd*S], in the
    /// same class as the live arena so a partial-accept rollback is a byte
    /// copy rather than a re-round
    pub snap: Vec<Option<SsmArena>>,
    /// per-mamba-layer conv-input (xBC) row snapshots [SPEC_ROWS, conv_dim]
    pub xbc: Vec<Option<CudaSlice<f32>>>,
    /// per-mamba-layer per-slot scratch conv windows [n_slots, (k-1)*conv_dim]
    pub vwin: Vec<Option<CudaSlice<f32>>>,
    /// window-rebuild bounce [(k-1), conv_dim] (overlapping same-buffer
    /// shifts are not a safe dtod copy)
    pub d_wbounce: CudaSlice<f32>,
    /// verify logits [SPEC_ROWS, vocab] + post-final-norm h [SPEC_ROWS, embd]
    pub d_logits: CudaSlice<f32>,
    pub d_h: CudaSlice<f32>,
    /// per-row picks (device argmax / device sample)
    pub d_picks: CudaSlice<u32>,
    /// per-row sampler params for device-sampled rounds [SPEC_ROWS, 4]
    /// (`sample_rows` packing + the truncation plane) - the batch scratch's
    /// own planes are sized for one row per slot, a round is not
    pub d_par: CudaSlice<u32>,
    pub d_tpar: CudaSlice<u32>,
    /// the round `forward_spec_verify` opened, awaiting the service's
    /// host-sampled counts in `spec_commit`
    pub open: Option<Vec<(usize, usize, Vec<u32>)>>,
    /// the multi-row attention partial's groups + partial planes (sized for
    /// the worst grouping of SPEC_ROWS rows - see `rows_partial_cap`)
    pub d_groups: CudaSlice<u32>,
    pub attn_o: CudaSlice<f32>,
    pub attn_ml: CudaSlice<f32>,
}

/// How a verify round's rows are picked on device before the commit.
enum VerifyPick<'a> {
    /// argmax - the greedy round
    Greedy,
    /// one sampler plan per row - the device-sampled round
    Plans(&'a [crate::sampler::DevicePlan]),
}

impl GpuNemotron {
    /// The verify machinery's kernel gate (drafters add their own).
    pub(crate) fn spec_verify_ready(&self) -> bool {
        self.exec.has_spec_verify_mamba() && self.exec.has_argmax_rows()
    }

    fn ensure_verify_planes(&mut self) -> Result<(), GpuModelError> {
        let hp = self.hp.clone();
        let e = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        if bs.verify.is_some() {
            return Ok(());
        }
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        let ssm_dt = self.ssm_dtype;
        let rows_cap = super::batch::rows_partial_cap(hp.n_kv_heads, e.sm_count(), SPEC_ROWS_NEMO);
        let mut snap = Vec::with_capacity(hp.n_layer);
        let mut xbc = Vec::with_capacity(hp.n_layer);
        let mut vwin = Vec::with_capacity(hp.n_layer);
        for li in 0..hp.n_layer {
            if matches!(hp.blocks[li], NemotronBlock::Mamba) {
                snap.push(Some(SsmArena::alloc(
                    &e,
                    SPEC_ROWS_NEMO * state_elems,
                    ssm_dt,
                )?));
                xbc.push(Some(e.alloc(SPEC_ROWS_NEMO * hp.conv_dim())?));
                vwin.push(Some(e.alloc(bs.n_slots * win_elems)?));
            } else {
                snap.push(None);
                xbc.push(None);
                vwin.push(None);
            }
        }
        bs.verify = Some(VerifyPlanes {
            snap,
            xbc,
            vwin,
            d_wbounce: e.alloc(win_elems)?,
            d_logits: e.alloc(SPEC_ROWS_NEMO * hp.vocab)?,
            d_h: e.alloc(SPEC_ROWS_NEMO * hp.hidden)?,
            d_picks: e.alloc_u32(SPEC_ROWS_NEMO)?,
            d_par: e.alloc_u32(SPEC_ROWS_NEMO * 4)?,
            d_tpar: e.alloc_u32(SPEC_ROWS_NEMO * 4)?,
            open: None,
            d_groups: e.alloc_u32(SPEC_ROWS_NEMO * 2)?,
            attn_o: e.alloc(hp.n_heads * rows_cap * hp.head_dim)?,
            attn_ml: e.alloc(hp.n_heads * rows_cap * 2)?,
        });
        tracing::info!(
            "nemotron spec: verify planes up ({} rows, {:.2} GiB snapshots)",
            SPEC_ROWS_NEMO,
            (23 * SPEC_ROWS_NEMO * state_elems * 4) as f64 / (1u64 << 30) as f64
        );
        Ok(())
    }

    /// Greedy verify round: run every request's `[pending, drafts...]` chunk
    /// through the trunk at positions `pos..pos+len`, pick per-row greedy
    /// tokens, commit the accepted prefix per slot (state rollback included)
    /// and return the flat picks. `Ok(None)` = decline (row budget).
    pub(crate) fn forward_spec_batch_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        self.spec_verify_picked(reqs, VerifyPick::Greedy)
    }

    /// Device-sampled verify round: the greedy round with every row drawn
    /// from its slot's own plan (temperature, top-k/top-p/min-p) instead of
    /// argmax'd. The drafts are deterministic, so accepting while the draw
    /// equals the draft is exact rejection sampling - the emitted stream has
    /// the dense sampler's distribution.
    pub(crate) fn forward_spec_batch_plans_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        use crate::sampler::DevicePlan;
        let total: usize = reqs.iter().map(|r| r.2.len()).sum();
        let runnable = plans.iter().all(|p| match p {
            DevicePlan::Greedy | DevicePlan::Categorical { .. } => true,
            DevicePlan::TruncCat { .. } => self.device_trunc_supported(),
            // no rejection-sampling resolve on this lane: its drafts are
            // argmaxes, which accept-while-match already samples exactly
            DevicePlan::RsVerify { .. } | DevicePlan::RsTrunc { .. } => false,
        });
        if plans.len() != total || !runnable || !self.supports_device_sampling_impl() {
            return Ok(None);
        }
        self.spec_verify_picked(reqs, VerifyPick::Plans(plans))
    }

    /// Host-sampled verify round, phase 1: the rows' raw logits for the
    /// service's own sampler - the round a constrained slot rides (a tool
    /// grammar walks each pick on the host). Phase 2 is `spec_commit_impl`
    /// with the counts the service accepted.
    pub(crate) fn forward_spec_verify_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GpuModelError> {
        let Some(total) = self.spec_verify_walk(reqs)? else {
            return Ok(None);
        };
        let vocab = self.hp.vocab;
        let bs = self.batch.as_mut().expect("batch enabled");
        let vp = bs.verify.as_mut().expect("verify planes");
        let view = vp
            .d_logits
            .try_slice(0..total * vocab)
            .ok_or_else(|| GpuError::Driver("verify logits view".into()))?;
        let rows = self
            .exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        vp.open = Some(reqs.to_vec());
        Ok(Some(rows))
    }

    /// Host-sampled verify round, phase 2: commit `committed[i]` rows of the
    /// open round's request `i` (the pending row plus the accepted drafts).
    pub(crate) fn spec_commit_impl(&mut self, committed: &[u32]) -> Result<(), GpuModelError> {
        let reqs = self
            .batch
            .as_mut()
            .and_then(|bs| bs.verify.as_mut())
            .and_then(|vp| vp.open.take())
            .ok_or_else(|| GpuModelError::Config("spec_commit with no open verify".into()))?;
        if committed.len() != reqs.len() {
            return Err(GpuModelError::Config(format!(
                "spec_commit: {} counts for {} requests",
                committed.len(),
                reqs.len()
            )));
        }
        let mut off = 0usize;
        let mut accepts = Vec::with_capacity(reqs.len());
        for (&(slot, _, ref chunk), &c) in reqs.iter().zip(committed) {
            let c = c as usize;
            if c == 0 || c > chunk.len() {
                return Err(GpuModelError::Config(format!(
                    "spec_commit: slot {slot} commits {c} of {} rows",
                    chunk.len()
                )));
            }
            accepts.push((slot, off, c));
            off += chunk.len();
        }
        self.spec_round_commit(&reqs, &accepts)
    }

    /// The verify walk every round shares: flatten the ragged chunks, run
    /// them through the trunk (mamba rows snapshot their state per row) and
    /// the head into the verify logits. `Ok(None)` = decline.
    fn spec_verify_walk(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<usize>, GpuModelError> {
        let total: usize = reqs.iter().map(|r| r.2.len()).sum();
        if total == 0 || total > SPEC_ROWS_NEMO || !self.spec_verify_ready() {
            return Ok(None);
        }
        if self.batch.is_none() {
            return Ok(None);
        }
        // A round with no draft in it is a decode tick taken the slow way:
        // the eager walk instead of the captured graph (66.7 vs 82.7 tok/s on
        // a slot whose drafter is cold for its whole sequence, GB10). Decline
        // it - the service cools down onto the dense tick, whose walk still
        // advances both drafters' coverage.
        if reqs.iter().all(|r| r.2.len() < 2) {
            return Ok(None);
        }
        self.pipe_b_abort();
        self.ensure_verify_planes()?;

        // flatten: rows in req order (one same-slot run per request)
        let mut toks = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        let mut slots = Vec::with_capacity(total);
        let mut runs: Vec<(usize, usize, u32)> = Vec::with_capacity(reqs.len());
        for &(slot, pos, ref chunk) in reqs {
            runs.push((toks.len(), chunk.len(), slot as u32));
            for (i, &t) in chunk.iter().enumerate() {
                toks.push(t);
                positions.push((pos + i) as u32);
                slots.push(slot as u32);
            }
        }
        self.ensure_rows(&slots, &positions)?;
        self.upload_rows(&toks, &positions, &slots)?;
        self.embed_rows(total)?;
        let cuts = PfCuts {
            runs,
            dec: 0,
            breaks: Vec::new(),
        };
        self.layer_walk(total, Some(&cuts), true)?;

        // head over every row: final norm -> h (the drafters' h source) ->
        // logits
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.hidden, self.hp.eps);
        let final_norm = self.final_norm.buf.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        let vp = bs.verify.as_mut().expect("verify planes");
        // a host round the service never committed is abandoned here - its
        // rows were overwritten by this walk
        vp.open = None;
        exec.rmsnorm_batch(&sc.d_x, &final_norm, &mut vp.d_h, embd, eps, total)?;
        match &self.lm_head {
            HeadW::Nvf4(h) => super::head_nvf4_batch(
                &exec,
                h,
                &vp.d_h,
                &mut vp.d_logits,
                total,
                super::batch::w16_class(&exec, total),
            )?,
            HeadW::Qw(q) => {
                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                prefill_quant(
                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &vp.d_h, embd, total,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    q,
                    &s8.xq,
                    &s8.xs,
                    &s8.yq,
                    &mut s8.xsums,
                    &mut s8.ssums,
                    &mut s8.skfix,
                    &mut vp.d_logits,
                    total,
                )?;
            }
        }
        Ok(Some(total))
    }

    /// Walk, pick every row on device, accept per slot and commit.
    fn spec_verify_picked(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        pick: VerifyPick<'_>,
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        let Some(total) = self.spec_verify_walk(reqs)? else {
            return Ok(None);
        };
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let picks: Vec<u32> = {
            let bs = self.batch.as_mut().expect("batch enabled");
            let vp = bs.verify.as_mut().expect("verify planes");
            match pick {
                VerifyPick::Greedy => {
                    exec.argmax_rows(&vp.d_logits, &mut vp.d_picks, total, vocab)?
                }
                VerifyPick::Plans(plans) => {
                    let rows: Vec<crate::generator::RowSample> = plans
                        .iter()
                        .map(|&p| crate::generator::RowSample::Device(p))
                        .collect();
                    let (par, tpar, any5, any6) = Self::pack_samp_par(&rows);
                    let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                    let mut v = vp
                        .d_par
                        .try_slice_mut(0..total * 4)
                        .ok_or_else(|| GpuError::Driver("verify d_par".into()))?;
                    exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
                    exec.sample_rows_at(
                        &vp.d_logits,
                        &vp.d_par,
                        0,
                        &mut vp.d_picks,
                        0,
                        total,
                        vocab,
                    )?;
                    if let Some(t) = &tpar {
                        let mut v = vp
                            .d_tpar
                            .try_slice_mut(0..total * 4)
                            .ok_or_else(|| GpuError::Driver("verify d_tpar".into()))?;
                        exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
                        if any5 {
                            exec.sample_rows_t_at(
                                &vp.d_logits,
                                &vp.d_par,
                                0,
                                &vp.d_tpar,
                                0,
                                &mut vp.d_picks,
                                0,
                                total,
                                vocab,
                            )?;
                        }
                        if any6 {
                            exec.sample_rows_p_at(
                                &vp.d_logits,
                                &vp.d_par,
                                0,
                                &vp.d_tpar,
                                0,
                                &mut vp.d_picks,
                                0,
                                total,
                                vocab,
                            )?;
                        }
                    }
                }
            }
            let view = vp
                .d_picks
                .try_slice(0..total)
                .ok_or_else(|| GpuError::Driver("picks view".into()))?;
            exec.stream
                .clone_dtoh(&view)
                .map_err(|e| GpuError::Driver(e.to_string()))?
        };

        // per-slot accepted counts - service.rs's exact walk, re-derived
        // here so the state rollback can never disagree with the tokens the
        // service streams
        let mut base = 0usize;
        let accepts: Vec<(usize, usize, usize)> = reqs
            .iter()
            .map(|&(slot, _, ref chunk)| {
                let off = base;
                let mut a = 0usize;
                while a + 1 < chunk.len() && chunk[a + 1] == picks[off + a] {
                    a += 1;
                }
                base += chunk.len();
                (slot, off, a + 1)
            })
            .collect();
        self.spec_round_commit(reqs, &accepts)?;
        Ok(Some(picks))
    }

    /// Commit a verified round: `accepts[i] = (slot, first row, committed
    /// rows)` for request `i`. The reply checkpoints read the pre-round conv
    /// windows, so they go before the rollback rewrites them.
    fn spec_round_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        accepts: &[(usize, usize, usize)],
    ) -> Result<(), GpuModelError> {
        self.reply_after_spec(reqs, accepts)?;
        self.spec_verify_commit(reqs, accepts)?;
        // the verified rows carry the drafter's features (the aux taps ran
        // during the walk); coverage advances only through ACCEPTED rows -
        // KV cells past that get overwritten by the next round's append
        let total: usize = reqs.iter().map(|r| r.2.len()).sum();
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            self.dflash_append_features(total)?;
            for (&(slot, pos0, _), &(_, _, acc)) in reqs.iter().zip(accepts) {
                self.dflash_note_rows(slot, pos0, acc);
            }
        }
        if self.mtp.as_ref().is_some_and(|m| m.state.is_some()) {
            // the MTP block consumes the round's rows with the verify h
            // (still in sc.d_x); coverage and the h chain advance only
            // through ACCEPTED rows - rejected cells get rewritten by the
            // next draft/verify before anything reads them
            let mut mruns = Vec::with_capacity(reqs.len());
            let mut base2 = 0usize;
            for &(slot, _, ref chunk) in reqs {
                mruns.push((slot, base2, chunk.len()));
                base2 += chunk.len();
            }
            self.mtp_append_rows(&mruns)?;
            for (&(slot, pos0, _), &(_, off, acc)) in reqs.iter().zip(accepts) {
                self.mtp_advance(slot, pos0, pos0 + acc, off + acc - 1)?;
            }
        }
        Ok(())
    }

    /// Roll every partially-accepted slot's mamba state back to the accepted
    /// row and rebuild its conv window; fully-accepted slots take the fast
    /// path (state already ended at the right row; window = the advanced
    /// scratch window).
    fn spec_verify_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        accepts: &[(usize, usize, usize)],
    ) -> Result<(), GpuModelError> {
        let hp = self.hp.clone();
        let exec = self.exec.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let conv_dim = hp.conv_dim();
        let km1 = hp.d_conv - 1;
        let win_elems = km1 * conv_dim;
        let bs = self.batch.as_mut().expect("batch enabled");
        for li in 0..hp.n_layer {
            if !matches!(hp.blocks[li], NemotronBlock::Mamba) {
                continue;
            }
            let win = bs.conv_win[li].as_mut().expect("conv arena");
            let ssm = bs.ssm[li].as_mut().expect("ssm arena");
            let vp = bs.verify.as_mut().expect("verify planes");
            let vw = vp.vwin[li].as_ref().expect("vwin");
            let snap = vp.snap[li].as_ref().expect("snap");
            let xbc = vp.xbc[li].as_ref().expect("xbc");
            for (ri, &(slot, off, acc)) in accepts.iter().enumerate() {
                let len = reqs[ri].2.len();
                let s = slot;
                if acc == len {
                    // full accept: live state already ended at the last row;
                    // the advanced scratch window is the new window
                    exec.copy_region(vw, s * win_elems, win, s * win_elems, win_elems)?;
                    continue;
                }
                // partial: state <- snapshot after the accepted row
                ssm.copy_region_from(
                    &exec,
                    snap,
                    (off + acc - 1) * state_elems,
                    s * state_elems,
                    state_elems,
                )?;
                // window <- last km1 rows of [pre-round window ∥ xBC rows
                // off..off+acc], assembled in the bounce (the live window is
                // both a source and the destination)
                let keep_old = km1.saturating_sub(acc); // pre-round rows kept
                let take_new = km1 - keep_old;
                for j in 0..keep_old {
                    exec.copy_region(
                        win,
                        s * win_elems + (acc + j) * conv_dim,
                        &mut vp.d_wbounce,
                        j * conv_dim,
                        conv_dim,
                    )?;
                }
                exec.copy_region(
                    xbc,
                    (off + acc - take_new) * conv_dim,
                    &mut vp.d_wbounce,
                    keep_old * conv_dim,
                    take_new * conv_dim,
                )?;
                exec.copy_region(&vp.d_wbounce, 0, win, s * win_elems, win_elems)?;
            }
        }
        Ok(())
    }
}
