//! Qwen3.8-Flash-Next prefix cache - the qwen35 design over this family's
//! DENSE slot-major carried state.
//!
//! What the other hybrid families do (qwen35, nemotron): a radix over 16-token
//! KV pages that later prompts adopt by refcount, plus a pool of recurrent-state
//! checkpoints taken at a prompt's last two page boundaries (`ckpt_cuts`), so
//! the next turn - the same history re-sent plus a reply and a new message -
//! resumes at the deepest checkpoint under its match and prefills only the
//! divergent tail.
//!
//! This family's attention KV is not paged: every slot owns a contiguous
//! `[max_tokens, kv_dim]` strip per attention layer and the paged kernels ride
//! an identity block table over it. So the cache keeps a SIDE STORE of pages
//! (the same `KvPool` bookkeeping, refcounted by the radix) and COPIES rows in
//! and out. A page of this geometry (2 kv heads x hd 256, a dozen attention
//! layers) is 16 x 512 x 1-2 bytes per layer per direction - a few hundred KB
//! for the whole set, a few MB per agentic turn, noise next to a weight pass.
//! The carried state (the GDN recurrence, its conv window, the PLE conv ring)
//! snapshots into a checkpoint pool at the cuts exactly as qwen35's does; the
//! token stream the PLE n-gram gather hashes is host state and is re-derived
//! from the prompt.
//!
//! The walk continues a sequence mid-way (`walk_span` with `from > 0`):
//! attention already takes per-row positions against the slot's cache, the
//! recurrence starts from the slot's state, and the two causal convs - which
//! left-pad with zeros at their base row - get their window rows re-staged in
//! front of the span's first rows (`resume_*` in forward.rs), which is the
//! whole-sequence conv bit for bit.

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::gpu::{GpuError, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_pool::{BlockId, KvPool};
use crate::paged_radix::PagedRadix;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

/// A dev switch's value as a count, or `default` (the macro wants a literal).
macro_rules! env_usize {
    ($name:literal, $default:expr) => {
        paddock_models::dev_var!($name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or($default)
    };
}

/// Don't bother snapshotting checkpoints for prompts shorter than this.
pub(super) const MIN_SNAPSHOT_LEN: usize = 3 * BLOCK_TOKENS;
/// A resume must skip at least this much, or the restore costs more than the
/// rows it saves (two pages).
const MIN_RESUME: usize = 2 * BLOCK_TOKENS;

/// The checkpoint boundaries for a prompt: its last two full page boundaries,
/// ascending ([0, 0] when the prompt is too short). Two, not one: a re-rendered
/// multi-turn history diverges inside the trailing generation header, and
/// whenever the prompt's final partial page is shorter than that header the
/// divergence crosses the last boundary - a checkpoint only there is
/// unreachable for the next turn (the qwen35 law).
pub(super) fn ckpt_cuts(t_len: usize) -> [usize; 2] {
    if t_len < MIN_SNAPSHOT_LEN {
        return [0, 0];
    }
    let b1 = (t_len - 1) / BLOCK_TOKENS * BLOCK_TOKENS;
    [b1.saturating_sub(BLOCK_TOKENS), b1]
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    /// side store / checkpoint pool -> the slot's carried state
    Load,
    /// the slot's carried state -> side store / checkpoint pool
    Store,
}

pub(super) struct PrefixCache {
    radix: PagedRadix,
    /// side-store page bookkeeping: the radix's refcounts are the only
    /// references (a slot's dense strips never hold a page)
    pool: KvPool,
    /// per layer, attention layers only: `[pages][BLOCK_TOKENS][kv_dim]` bytes
    side_k: Vec<Option<CudaSlice<u8>>>,
    side_v: Vec<Option<CudaSlice<u8>>>,
    row_bytes: usize,
    page_bytes: usize,
    /// `[n_ckpt][ckpt_f32]`: per GDN layer the slot's recurrence then its conv
    /// window, then the PLE ring (if the model has one)
    state_pool: CudaSlice<f32>,
    ckpt_f32: usize,
    st_elems: usize,
    win_elems: usize,
    ple_elems: usize,
    /// batched-copy descriptors (src, dst, bytes) x max_descs
    descs: CudaSlice<u64>,
    max_descs: usize,
    last_reused: Vec<usize>,
    stats: bool,
    /// per pool index: the (length, hash) of the prompt that took the
    /// checkpoint INSIDE its own prefill walk - an exact re-send of that prompt
    /// prefills cold (see `resume`). None for a checkpoint a walk boundary
    /// took (cut walks, the reply checkpoint).
    src: Vec<Option<(usize, u64)>>,
}

/// A checkpoint blob in the pool as a prefill walk writes it from inside
/// itself - `copy_state`'s layout: per GDN layer, in layer order, the
/// recurrence then its conv window, then the PLE ring.
pub(super) struct CkptSink<'a> {
    pub(super) pool: &'a mut CudaSlice<f32>,
    pub(super) ckpt_f32: usize,
    pub(super) st_elems: usize,
    pub(super) win_elems: usize,
}

impl CkptSink<'_> {
    /// Element offset of GDN layer `gdn_ord`'s recurrence in checkpoint `idx`.
    pub(super) fn state_off(&self, idx: u32, gdn_ord: usize) -> usize {
        idx as usize * self.ckpt_f32 + gdn_ord * (self.st_elems + self.win_elems)
    }
    /// Element offset of GDN layer `gdn_ord`'s conv window in checkpoint `idx`.
    pub(super) fn win_off(&self, idx: u32, gdn_ord: usize) -> usize {
        self.state_off(idx, gdn_ord) + self.st_elems
    }
    /// Element offset of the PLE ring in checkpoint `idx` (after all `n_gdn`).
    pub(super) fn ple_off(&self, idx: u32, n_gdn: usize) -> usize {
        idx as usize * self.ckpt_f32 + n_gdn * (self.st_elems + self.win_elems)
    }
}

/// FNV-1a over a prompt's ids: which prompt took an in-walk checkpoint.
fn prompt_hash(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

impl PrefixCache {
    /// Size and allocate the cache for `slots` seats of `max_tokens`, or
    /// `None` when the model has nothing to cache (no attention layers) or
    /// the budget leaves no room.
    pub(super) fn new(
        exec: &GpuExecutor,
        cfg: &Qwen4ExpConfig,
        slots: usize,
        max_tokens: usize,
        kv_bytes: usize,
    ) -> Result<Option<Self>, GpuModelError> {
        let n_attn = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b, Qwen4ExpBlock::Attention))
            .count();
        let n_gdn = cfg.n_layer - n_attn;
        if n_attn == 0 || slots == 0 {
            return Ok(None);
        }
        let row_bytes = cfg.n_kv_heads * cfg.head_dim * kv_bytes;
        let page_bytes = BLOCK_TOKENS * row_bytes;
        // pages: two turns of every seat by default, under a byte budget
        let want_tokens = env_usize!("PADDOCK_Q38FN_PREFIX_TOKENS", 2 * slots * max_tokens);
        let budget = env_usize!("PADDOCK_Q38FN_PREFIX_MB", 2048) << 20;
        let pages = (want_tokens / BLOCK_TOKENS).min(budget / (2 * n_attn * page_bytes).max(1));
        if pages < 8 {
            return Ok(None);
        }
        // checkpoints: the qwen35 rule (six per slot after the reply checkpoint
        // landed - two prompt cuts + one reply cut, two waves), under a byte cap
        let st_elems = cfg.gdn_v_heads * cfg.gdn_k_dim * cfg.gdn_v_dim;
        let win_elems = (cfg.gdn_conv - 1) * cfg.gdn_qkv_rows();
        let ple_elems = if cfg.ple_layers.is_empty() {
            0
        } else {
            (cfg.ple_conv - 1) * super::forward::PLE_DILATION * cfg.hc_width()
        };
        let ckpt_f32 = n_gdn * (st_elems + win_elems) + ple_elems;
        let ckpt_bytes = ckpt_f32 * 4;
        let want_ckpt = env_usize!("PADDOCK_KV_STATE_CKPTS", (slots * 6).clamp(8, 64));
        let cap = env_usize!("PADDOCK_Q38FN_PREFIX_STATE_MB", 6144) << 20;
        let n_ckpt = want_ckpt.min(cap / ckpt_bytes.max(1)).max(2);
        let mut side_k = Vec::with_capacity(cfg.n_layer);
        let mut side_v = Vec::with_capacity(cfg.n_layer);
        for b in &cfg.blocks {
            match b {
                Qwen4ExpBlock::Attention => {
                    side_k.push(Some(exec.alloc_u8(pages * page_bytes)?));
                    side_v.push(Some(exec.alloc_u8(pages * page_bytes)?));
                }
                Qwen4ExpBlock::Gdn => {
                    side_k.push(None);
                    side_v.push(None);
                }
            }
        }
        let state_pool = exec.alloc(n_ckpt * ckpt_f32)?;
        let max_descs = (2 * n_attn * pages).max(2 * n_gdn + 2);
        let descs = exec.alloc_u64(3 * max_descs)?;
        let mut radix = PagedRadix::new();
        radix.set_state_capacity(n_ckpt as u32);
        tracing::info!(
            "qwen4exp prefix cache: {pages} side pages ({} MB over {n_attn} attention layers), \
             {n_ckpt} state checkpoints ({} MB each)",
            (2 * n_attn * pages * page_bytes) >> 20,
            ckpt_bytes >> 20
        );
        Ok(Some(Self {
            radix,
            pool: KvPool::with_blocks(pages as u32),
            side_k,
            side_v,
            row_bytes,
            page_bytes,
            state_pool,
            ckpt_f32,
            st_elems,
            win_elems,
            ple_elems,
            descs,
            max_descs,
            last_reused: vec![0; slots],
            stats: paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some(),
            src: vec![None; n_ckpt],
        }))
    }

    /// How many leading tokens the last prefill of `slot` took from the cache
    /// (the usage line's `cached_tokens`); cleared on read.
    pub(super) fn take_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    /// The resume point for `tokens` in `slot`: the deepest checkpoint under
    /// the radix match, with its KV pages copied into the slot's strips and
    /// its state restored. 0 = cold (nothing touched).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resume(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        tokens: &[u32],
        max_tokens: usize,
        kv_k: &mut [Option<CudaSlice<u8>>],
        kv_v: &mut [Option<CudaSlice<u8>>],
        recur: &mut [Option<CudaSlice<f32>>],
        gdn_win: &mut [Option<CudaSlice<f32>>],
        ple_win: Option<&mut CudaSlice<f32>>,
    ) -> Result<usize, GpuModelError> {
        let t_len = tokens.len();
        let m = self.radix.match_full(tokens);
        let Some((pos, idx)) = m.ckpt else {
            return Ok(0);
        };
        // An exact re-send of the prompt that took this checkpoint inside its
        // own walk prefills cold: resuming would replay rows the first run
        // computed inside one walk through a shorter one, and the two agree
        // only to the last ulp. Cold, the re-send is bit-identical to the
        // first run (the in-walk checkpoint trade, chosen 2026-09-15).
        if self.src.get(idx as usize).copied().flatten() == Some((t_len, prompt_hash(tokens))) {
            if self.stats {
                tracing::info!("qwen4exp-resume: t_len {t_len} is an exact re-send - cold");
            }
            return Ok(0);
        }
        if pos < MIN_RESUME || pos >= t_len || m.blocks.len() * BLOCK_TOKENS < pos {
            if self.stats {
                tracing::info!(
                    "qwen4exp-resume: t_len {t_len} matched {} tok, ckpt {pos} - not resumable",
                    m.blocks.len() * BLOCK_TOKENS
                );
            }
            return Ok(0);
        }
        let pages = &m.blocks[..pos / BLOCK_TOKENS];
        self.copy_pages(exec, slot, 0, pages, max_tokens, kv_k, kv_v, Dir::Load)?;
        self.copy_state(exec, slot, idx, recur, gdn_win, ple_win, Dir::Load)?;
        self.last_reused[slot] = pos;
        if self.stats {
            tracing::info!(
                "qwen4exp-resume: t_len {t_len} matched {} tok, resumed at {pos} (ckpt {idx})",
                m.blocks.len() * BLOCK_TOKENS
            );
        }
        Ok(pos)
    }

    /// After the walk reached `upto` tokens of `tokens` in `slot`: file every
    /// full page up to there under the radix (copying only the pages the
    /// radix does not already hold along this path) and, when asked, attach a
    /// state checkpoint at `upto` (a page boundary) and snapshot the slot's
    /// carried state into it. Returns the checkpoint's pool index when one
    /// was attached (the reply checkpoint tracks its own for the detach).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn publish(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        tokens: &[u32],
        upto: usize,
        snapshot: bool,
        max_tokens: usize,
        kv_k: &mut [Option<CudaSlice<u8>>],
        kv_v: &mut [Option<CudaSlice<u8>>],
        recur: &mut [Option<CudaSlice<f32>>],
        gdn_win: &mut [Option<CudaSlice<f32>>],
        ple_win: Option<&mut CudaSlice<f32>>,
    ) -> Result<Option<u32>, GpuModelError> {
        let full = upto / BLOCK_TOKENS;
        if full == 0 {
            return Ok(None);
        }
        let prefix = &tokens[..full * BLOCK_TOKENS];
        let m = self.radix.match_full(prefix);
        let have = m.blocks.len();
        if have < full {
            let need = full - have;
            while self.pool.free_blocks() < need {
                if self.radix.evict_lru(&mut self.pool).is_none() {
                    break;
                }
            }
            if self.pool.free_blocks() < need {
                return Ok(None); // nothing evictable: this prompt is not cached
            }
            let mut new: Vec<BlockId> = Vec::with_capacity(need);
            for _ in 0..need {
                new.push(
                    self.pool
                        .alloc()
                        .map_err(|_| GpuError::Driver("prefix side store exhausted".into()))?,
                );
            }
            self.copy_pages(exec, slot, have, &new, max_tokens, kv_k, kv_v, Dir::Store)?;
            let mut all = m.blocks.clone();
            all.extend_from_slice(&new);
            self.radix.insert(prefix, &all, &mut self.pool);
            // the radix now holds the only reference to every page it adopted;
            // a page it did not adopt (a diverging node in the way) goes free
            for b in new {
                self.pool.release(b);
            }
        }
        if snapshot
            && upto.is_multiple_of(BLOCK_TOKENS)
            && let Some(idx) = self.radix.attach_state(tokens, upto)
        {
            self.copy_state(exec, slot, idx, recur, gdn_win, ple_win, Dir::Store)?;
            if let Some(s) = self.src.get_mut(idx as usize) {
                *s = None;
            }
            if self.stats {
                tracing::info!("qwen4exp-ckpt: slot {slot} cut {upto} idx {idx}");
            }
            return Ok(Some(idx));
        }
        Ok(None)
    }

    /// The pool as a prefill walk writes in-walk checkpoints into it.
    pub(super) fn ckpt_sink(&mut self) -> CkptSink<'_> {
        CkptSink {
            pool: &mut self.state_pool,
            ckpt_f32: self.ckpt_f32,
            st_elems: self.st_elems,
            win_elems: self.win_elems,
        }
    }

    /// A state-pool index for a checkpoint the next walk writes from inside
    /// itself (attach it with [`Self::attach_reserved`] once the pages up to
    /// its cut are filed, or give it back with [`Self::recycle_ckpt`]).
    pub(super) fn reserve_ckpt(&mut self) -> Option<u32> {
        self.radix.reserve_state_slot()
    }

    /// Attach reserved checkpoint `idx` at `cut` of `tokens`, recording the
    /// prompt that took it; on a miss (the node is gone or already
    /// checkpointed) the index goes back to the pool.
    pub(super) fn attach_reserved(&mut self, tokens: &[u32], cut: usize, idx: u32) -> bool {
        if self.radix.attach_state_at(tokens, cut, idx) {
            if let Some(s) = self.src.get_mut(idx as usize) {
                *s = Some((tokens.len(), prompt_hash(tokens)));
            }
            if self.stats {
                tracing::info!("qwen4exp-ckpt: in-walk cut {cut} idx {idx}");
            }
            true
        } else {
            self.radix.recycle_state(idx);
            false
        }
    }

    /// Give back a reserved index that was never attached.
    pub(super) fn recycle_ckpt(&mut self, idx: u32) {
        self.radix.recycle_state(idx);
    }

    /// Drop the checkpoint at `cut` of `tokens` - but only if it is still
    /// `idx`: the pool may have stolen the index for another prompt since,
    /// and another slot walking the same sequence may have re-checkpointed
    /// the node, neither of which is ours to detach. (Peek first: a detach
    /// TAKES the node's checkpoint, so "detach then compare" would strip a
    /// foreign one on a mismatch.)
    pub(super) fn drop_ckpt(&mut self, tokens: &[u32], cut: usize, idx: u32) {
        if cut > tokens.len() || self.radix.match_full(&tokens[..cut]).ckpt != Some((cut, idx)) {
            return;
        }
        if self.radix.detach_state_at(tokens, cut) == Some(idx) {
            self.radix.recycle_state(idx);
        }
    }

    /// The slot's KV rows for pages `first_page..first_page+pages.len()` <->
    /// the side store's `pages`, every attention layer, both directions of
    /// the cache, one batched copy.
    #[allow(clippy::too_many_arguments)]
    fn copy_pages(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        first_page: usize,
        pages: &[BlockId],
        max_tokens: usize,
        kv_k: &mut [Option<CudaSlice<u8>>],
        kv_v: &mut [Option<CudaSlice<u8>>],
        dir: Dir,
    ) -> Result<(), GpuModelError> {
        let mut descs: Vec<u64> = Vec::with_capacity(3 * 2 * pages.len() * self.side_k.len());
        for li in 0..self.side_k.len() {
            let (Some(sk), Some(sv)) = (self.side_k[li].as_ref(), self.side_v[li].as_ref()) else {
                continue;
            };
            let (Some(kc), Some(vc)) = (kv_k[li].as_ref(), kv_v[li].as_ref()) else {
                continue;
            };
            let (skp, _g1) = sk.device_ptr(&exec.stream);
            let (svp, _g2) = sv.device_ptr(&exec.stream);
            let (kp, _g3) = kc.device_ptr(&exec.stream);
            let (vp, _g4) = vc.device_ptr(&exec.stream);
            for (i, &b) in pages.iter().enumerate() {
                let strip =
                    ((slot * max_tokens + (first_page + i) * BLOCK_TOKENS) * self.row_bytes) as u64;
                let side = (b as usize * self.page_bytes) as u64;
                let len = self.page_bytes as u64;
                match dir {
                    Dir::Load => {
                        descs.extend([skp + side, kp + strip, len]);
                        descs.extend([svp + side, vp + strip, len]);
                    }
                    Dir::Store => {
                        descs.extend([kp + strip, skp + side, len]);
                        descs.extend([vp + strip, svp + side, len]);
                    }
                }
            }
        }
        self.run_descs(exec, &descs)
    }

    /// The slot's carried state <-> checkpoint `idx` of the pool.
    #[allow(clippy::too_many_arguments)]
    fn copy_state(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        idx: u32,
        recur: &mut [Option<CudaSlice<f32>>],
        gdn_win: &mut [Option<CudaSlice<f32>>],
        ple_win: Option<&mut CudaSlice<f32>>,
        dir: Dir,
    ) -> Result<(), GpuModelError> {
        let mut descs: Vec<u64> = Vec::with_capacity(3 * (2 * recur.len() + 1));
        {
            let (pp, _g) = self.state_pool.device_ptr(&exec.stream);
            let (st_elems, win_elems, ple_elems) = (self.st_elems, self.win_elems, self.ple_elems);
            let mut boff = (idx as usize * self.ckpt_f32 * 4) as u64;
            let mut push = |descs: &mut Vec<u64>, slot_ptr: u64, len_elems: usize| {
                let len = (len_elems * 4) as u64;
                match dir {
                    Dir::Load => descs.extend([pp + boff, slot_ptr, len]),
                    Dir::Store => descs.extend([slot_ptr, pp + boff, len]),
                }
                boff += len;
            };
            for li in 0..recur.len() {
                let Some(r) = recur[li].as_ref() else {
                    continue;
                };
                let w = gdn_win[li].as_ref().expect("gdn layer has a conv window");
                let (rp, _g1) = r.device_ptr(&exec.stream);
                let (wp, _g2) = w.device_ptr(&exec.stream);
                push(&mut descs, rp + (slot * st_elems * 4) as u64, st_elems);
                push(&mut descs, wp + (slot * win_elems * 4) as u64, win_elems);
            }
            if ple_elems > 0 {
                let w = ple_win.expect("model has a PLE ring");
                let (wp, _g3) = w.device_ptr(&exec.stream);
                push(&mut descs, wp + (slot * ple_elems * 4) as u64, ple_elems);
            }
        }
        self.run_descs(exec, &descs)
    }

    fn run_descs(&mut self, exec: &GpuExecutor, descs: &[u64]) -> Result<(), GpuModelError> {
        for chunk in descs.chunks(3 * self.max_descs) {
            let n = chunk.len() / 3;
            {
                let mut v = self.descs.slice_mut(0..chunk.len());
                exec.stream
                    .memcpy_htod(chunk, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            exec.batched_copy(&self.descs, n)?;
        }
        Ok(())
    }
}
