# TP=2 text-only prefix-cache resume — design note (milestone 1)

Starting HEAD `0f73352bab63ebb41e54d1c1b54cbf855c88bd85`, branch
`review/qwen38-tp2-final`. This note is the Phase 1 trace + Phase 2 design the
implementation follows. Scope: text-only prompts, in-GPU-memory cache, TP=2,
Mixed/decode-lane prefill, cold + identical + shared-prefix-divergent-tail
reuse. Out of scope per the milestone: NVMe/RAM tiering, remote restore,
multimodal, reply-checkpoint sophistication, Async-lane restore, perf tuning.

## Phase 1 — the non-TP Qwen35 prefix cache, end to end

- **Allocation.** `PagedRadix` is allocated in `BatchState.paged_prefix`
  (batch.rs ~655) next to `KvPool` (`pool`) and the DeltaNet checkpoint device
  pool `d_state_pool`. `radix.set_state_capacity(n)` arms checkpoint indices.
- **Keys.** Raw token ids; block identity is FNV-1a over each 16-token block
  (`paged_radix::hash_block`) with exact-token comparison on match, so a hash
  collision costs a miss, never a wrong reuse.
- **Match.** `match_full(tokens)` walks full blocks (`cap = len-1` keeps one
  token to prefill) and returns `PagedMatch { blocks, ckpt, tail }`; `ckpt` is
  the deepest `(position, state_idx)` on the matched path.
- **GQA adoption.** `tables[slot].clear(pool)` then
  `tables[slot].share_prefix(&blocks[..pos/16], pool)` — zero-copy refcount.
  Block-aligned resume means the slot never writes into a shared block, so no
  CoW is needed.
- **DeltaNet restore.** `restore_paged_state(slot, idx)` batched-copies the
  checkpoint blob (per linear layer: recurrent state + conv window) from
  `d_state_pool[idx]` into the slot's live state. Fresh sequences instead run
  `zero_slot_state`.
- **Fresh slot init.** Table cleared; recurrent + conv zeroed
  (`zero_slot_state`).
- **Checkpoint snapshots.** During the paged tail, prefill splits at
  `ckpt_cuts(t_len, step)` = the last two full page boundaries; at each cut the
  state is snapshotted (`snapshot_paged_state`) BEFORE the following rows
  advance it, pages are inserted, and the state is attached
  (`attach_state`). After prefill, every full page is inserted
  (idempotent). `MIN_SNAPSHOT_LEN` skips tiny prompts.
- **Refcounts.** `insert` retains (tree ref), `share_prefix` retains (slot
  ref), `BlockTable::clear` drops slot refs, `evict_lru` drops the tree ref; a
  block frees only at refcount 0. `evictable_blocks` treats refcount-1 tree
  pages as reclaimable capacity.
- **Accounting.** `last_reused[slot] = start` → `take_prefill_reused(k)` →
  `finish_prefill` emits `TokenEvent::Prefilled { cached, rows }` and feeds
  `prefill_tokens_cached`.
- **Block alignment.** Resume positions are multiples of `BLOCK_TOKENS = 16`
  by radix construction, gated `pos >= 32`, `pos < t_len`, and
  `pos >= min_cache_prefix() || slots <= resume_live_max()`.

**Deep agentic resume on the non-TP path comes from stage F reply
checkpoints** (snapshots every page boundary during decode). That machinery is
explicitly out of scope here, so milestone-1 TP resume depth is bounded by
prompt-prefill checkpoints (the last two page boundaries of a previously
prefilled prompt). Scenario B (shared history + divergent tail) resumes
exactly when the divergent suffix starts at/after a published checkpoint —
i.e. the shared prefix extends through a cached boundary. Identical prompts
resume at the second-to-last boundary and recompute only the trailing partial
page. This is the honest milestone-1 contract and is documented as a
limitation.

## Phase 1b — the TP=2 serving path, end to end

- `TpCoordinator` (rank 0, `tp_serve.rs`) owns `logical: MirroredKv`
  (`tp_kv.rs`) — a `KvPool` + per-slot `BlockTable`s + **a `PagedRadix`** —
  and replays every mutation to rank 1 as ordered `Operation`s plus one
  end-of-tick `Snapshot`; `mirror_tick` re-applies deterministically and fails
  closed on any divergence. Physical block ids are therefore
  mirror-deterministic and snapshot-validated; the protocol never needs to
  carry rank-0 page ids.
- `MirroredKv` already defines `Publish` (radix insert from a slot's table)
  and `Reuse` (radix match + share_prefix) — **neither is called anywhere in
  production**. The radix exists, mirrored, but is never populated or
  consulted.
- Prompts enter via `chunk_enqueue(slot, tokens)` (full tokens, rank 0 only)
  and flow out through `chunk_take` as `(slot, token, position)` rows starting
  at position 0; both ranks validate row positions against their mirrored
  cursors. There is no resume: every request re-prefills from 0.
- DeltaNet TP state is per-layer per-slot (`DeltaTpRank::slot_states`, slot 0
  in home buffers); there is no checkpoint pool, no snapshot, no restore. The
  prefill lane (`PreFillLane`) is a structural re-home with its own zeroed
  KV slabs and DeltaNet state; `promote_lane_slot` copies lane→decode at span
  finish.

## Phase 2 — TP cache ownership decision

**Chosen: mirrored metadata radix on both ranks (the existing
`MirroredKv::radix`), rank-0-authoritative decisions, logical operations on
the wire.** No second radix, no coordinator-only index: the deterministic
replay + end-state equality already guarantees the two trees cannot drift
silently, and `Publish`/`Reuse` were built for exactly this.

Invariant: for every mirrored operation, both ranks' `PagedRadix` evolve
identically (same op sequence, same hash, same LRU clock, same steal policy);
the `Snapshot` comparison validates the KV-visible projection (tables,
refcounts, free, plus the new token-length/digest and checkpoint free-list
below), and the resume-time exact-checkpoint validation catches any tree
divergence at the only place it could matter. **The wire carries logical
identity only** (tokens, position, checkpoint index as a validated
reservation) — never rank-0 physical page ids.

Flow per the milestone contract:

1. rank 0 admits: read-only `match_full_peek` on its radix → pure decision
   `tp_resume_decision(ckpt, t_len, slots)`;
2. rank 0 authorizes `Operation::Admit { slot, tokens, resume }` (clear table,
   store tokens, on resume: mutating match, require the checkpoint at exactly
   `resume`, share the prefix blocks, record the local state index);
3. rank 0 sends `TpPrefixAdmit` (full tokens once); the worker mirrors the op,
   independently validates the same checkpoint exists at exactly `resume` on
   its own radix, replies Prepared;
4. both ranks restore rank-local DeltaNet state (pool blob → slot state) and
   set `positions[slot] = resume`; only then is the request authorized to
   prefill `tokens[resume..]`;
5. during prefill, when a span ends exactly at a checkpoint cut, each rank
   snapshots its own DeltaNet state into its own pool at the reserved index;
6. at successful finish, rank 0 authorizes `[CheckpointAttach.., Publish]`
   (attach reserved indices at the snapshotted cuts, insert all full pages,
   recycle unused reservations) and mirrors `TpPrefixPublish`.

If either rank cannot satisfy the selected checkpoint, the mirror/validation
fails closed and the pair poisons — the existing all-or-nothing failure model.
No partial adoption survives: `Admit` is atomic (staged authorize), and any
later error kills the pair before reuse.

## Design decisions and invariants

- **Rank symmetry.** The decision is made once on rank 0 from a read-only
  match; the authoritative `Admit` op performs exactly one mutating match on
  each rank and requires the checkpoint at exactly the coordinator-chosen
  position. A worker that cannot satisfy it fails the tick (no divergent
  resume).
- **No physical page ids on the wire.** Tables/refocks are mirrored and
  snapshot-compared; resume carries tokens + position.
- **State/page correspondence.** A checkpoint is snapshotted only when a span
  ends exactly at the cut position, so the blob is the state after row
  `cut-1` = the state for KV pages `[0, cut)`. Attach only happens for cuts
  the prefill actually snapshotted (`cut > start`); everything else recycles.
- **Refcounts.** Adoption retains via the existing `share_prefix`; slot
  release (`Operation::Release`) drops slot refs, recycles that slot's unused
  reservations, and clears stored tokens; radix refs persist until eviction
  (`Flush`/LRU). A failed restore poisons the pair (no leak path reuses the
  slot).
- **Lane safety (milestone-1 pin).** The prefill lane's KV slabs hold no
  adopted-prefix content, and span-finish promotion copies lane→decode for
  every live block — a resumed prompt on the lane would overwrite decode
  slabs with stale lane data for the adopted blocks. Resumed prompts are
  therefore pinned to Mixed at admission (`owner = Some(Mixed)` before the
  first take); `prefill_front_owner` reports Mixed so the scheduler never
  selects Async for them, and `span_take` refuses a Mixed-pinned front chunk
  without erroring. Cold prompts may still use Async (they publish pages only
  — no checkpoints — so nothing stale is ever promoted over adopted content
  that resume depends on).
- **Pool pressure.** `Ensure` on an exhausted logical pool first evicts LRU
  radix leaves (refcount-1 pages are reclaimable capacity), deterministically
  on both ranks, mirroring the non-TP "cache as reclaimable capacity" rule.
- **Checkpoint capacity.** Rank-0 resolves `PADDOCK_TP_CKPT_SLOTS` (default 4
  ≈ 4 × ~57 MB/rank at the Qwen3.8 geometry) and sends it in `TpInit`; both
  ranks size pool + radix capacity from the wire value. Capacity exhaustion
  steals the LRU checkpoint (deterministic); zero capacity degrades to
  always-cold.
- **Serial path stays cold.** The single-slot `Command::Prefill` path does not
  resume in milestone 1 (chunked admission is the production TP path).

## Reuse vs. new code

Reused unchanged: `PagedRadix` (one new read-only `match_full_peek`),
`KvPool`/`BlockTable` refcount lifecycle, the whole `MirroredKv`
authorize/mirror machinery, `span_chunk_points`/span execution, lane
promotion, release/reset ordering, `finish_prefill` accounting.
New: `Admit`/`CheckpointReserve`/`CheckpointAttach`/extended `Publish` ops in
`tp_kv.rs`; the DeltaNet checkpoint pool + snapshot/restore on
`Qwen35TpRank`; the pure decision helper; `TpPrefixAdmit`/`TpPrefixPublish`
wire messages (protocol v4); admission/publish hooks + Mixed pin in
`tp_serve.rs`; `take_prefill_reused`/`prefill_begin_hinted` on
`TpGenerator`.


## Milestone-1 implementation record

Commits on `review/qwen38-tp2-final`, starting HEAD `0f73352`:

1. `229be9f` - this design note (Phase 1/2);
2. `06d24e0` - cache metadata: `Admit`/`CheckpointReserve`/`CheckpointAttach`/
   `CheckpointRecycle` in `MirroredKv`, `probe_ckpt`/`match_full_upto`/
   reservations in `PagedRadix`, the pure `tp_resume_decision` +
   `tp_publish_ops`;
3. `f4ef4eb` - protocol v4 (`ckpt_slots` in `TpInit`, `TpPrefixAdmit`,
   `TpPrefixPublish`), coordinator admission/publish hooks, the rank-local
   DeltaNet checkpoint pool (`enable_prefix_checkpoints`,
   `snapshot_slot_ckpt`, `restore_slot_ckpt`);
4. `190681c` - resumed-prompt Mixed pin, wire cut lists (`TpMixed.ckpts`,
   `TpSpanLaunch.ckpts`), worker rank-local snapshots at cut boundaries,
   fail-closed publish, admission-cursor integration;
5. `92590cc` - wire-frame cap/roundtrip tests for the new messages.

Validation: `cargo test -p paddock-engine --lib` 500 passed (12 new
`tp_kv` + 2 new `tp_serve` tests); `cargo test -p paddock-dist` 5 + 22
passed (2 new protocol tests); `--test tp_wire_frame` 6 passed (1 new);
`cargo clippy -p paddock-engine -p paddock-dist --all-targets` clean apart
from the pre-existing `cuda.rs`/example warnings; `git diff --check` clean;
`cargo build --release -p paddock-runner --bin paddock-runner` passed. The
GPU-touching `--all-targets` binaries were NOT run in this session (the host
GPU is occupied by the serving model); they belong to the validation session
below.

### How the invariants are met

- **Rank symmetry.** Rank 0 probes read-only, decides once with
  `tp_resume_decision`, and authorizes `Admit`; each rank's `Admit` apply
  re-walks its own tree and requires the checkpoint at EXACTLY the chosen
  position (`match_full_upto` + attached-checkpoint check) before adopting.
  A rank that cannot satisfy it fails the tick - recompute, never a
  one-sided resume. The worker additionally fail-closes a publish whose cuts
  have no local reservation.
- **No physical page ids on the wire.** Resume carries tokens + position;
  publish carries `(cut, rank-0 index)` pairs where the index is
  validation-only (each rank attaches its own mirror-deterministic
  reservation).
- **GQA restore.** `Admit` clears the slot's table and `share_prefix`es the
  matched blocks (zero-copy refcount adoption) - the same mechanism the
  non-TP path uses; both ranks' tables are snapshot-validated.
- **DeltaNet restore.** Both ranks `restore_slot_ckpt(slot, idx)` from their
  own pool at the validated index BEFORE the admission Ready; the recurrent
  state and conv window are restored together (one per-layer pair copy),
  matching the logical position of the adopted pages exactly (enforced by
  the attach-at-exact-position rule).
- **Publication.** Mid-prefill, a span ending exactly on a reserved cut
  snapshots the live DeltaNet state (rank-local) BEFORE further rows
  advance it; at successful finish the publish tick attaches the
  snapshotted reservations, publishes all full pages, and recycles the
  rest - only after the GPU work succeeded on both ranks.
- **Refs/releases.** Insert retains (tree ref); adoption retains (slot ref);
  `Release`/`Reset`/`Flush` drop slot refs, recycle that slot's
  unattached reservations, and keep radix-owned pages until LRU eviction
  (tested: release frees exactly the un-published partial-tail page).
- **Scheduler integration.** `chunk_enqueue` sets `positions[slot] = resume`
  and queues rows `[resume..t_len)`; `chunk_take` therefore starts at
  `resume` (tested); `finish_prefill` reports `cached = resume` through
  `prefill_begin_hinted`/`take_prefill_reused`.
- **Mixed pin.** A resumed prompt's queue entry is pinned
  `owner = Some(Mixed)` before the first take, `span_take` refuses it
  (returns false, no error), and `prefill_front_owner` reports the pin -
  the scheduler can never select Async for a restored prompt in milestone 1.

### Known limitations (deliberate, milestone 1)

- Resume depth is bounded by prompt-prefill checkpoints (the last two page
  boundaries): agentic depth comes from stage-F reply checkpoints, which are
  out of scope here. Identical prompts resume at the second-to-last
  boundary and recompute the trailing partial page; scenario B resumes only
  when the divergent tail starts at/after a published checkpoint.
- The serial one-slot `Prefill` path stays cold (chunked admission is the
  production TP path).
- Multimodal prompts are not covered (keys are raw token ids; the scheduler
  routes mm prompts elsewhere).
- The checkpoint pool is modest (default 4 slots/layer/rank,
  `PADDOCK_TP_CKPT_SLOTS` to change); exhaustion steals the LRU checkpoint
  deterministically on both ranks.

### GPU validation plan (next hosted-model session)

Run from a clean server (fresh coordinator + worker; `SPEC=off`,
`TP_GRAPH=0`, `PADDOCK_UNIFIED=1`, production `MAX_CTX=65536`); the exact
~20k Hermes request from the accepted run, deterministic sampling
(temperature 0, seed 1). Rank-0 and rank-1 logs both carry
`TP prefix admit`/`TP prefix published` lines - record BOTH ranks' resume
positions and prove equality (gate D).

- **A. Cold then identical.** Send P (~20k text tokens); record cold prefill
  duration. Send P again; verify `TP prefix admit (cache HIT)` with
  `resume` = the second-to-last page boundary of P, `Prefilled { cached:
  resume }` in usage, only the trailing partial page + 1 token computed
  (mixed-tick row counts), and a dramatic latency reduction. First
  generated token must match the cold run under the deterministic plan.
- **B. Shared prefix, divergent tail.** P1 = long history + user message A;
  P2 = same history + user message B. Verify P2's `resume` lands inside the
  shared history (>= the deepest checkpoint P1 published), only the
  divergent suffix appears in mixed-tick rows, and the decode is coherent.
  NOTE: the tail must start at/after a published checkpoint - if A's text
  begins inside P1's trailing partial page, extend the shared history by a
  full page before the divergence.
- **C. Release/reuse.** Complete/cache one request; reuse the same slot with
  a resumed prompt; cancel a queued/active request (post-drain per the
  existing invariant); reuse again. No stale KV/DeltaNet/page errors, no
  `KV position has no live pages`, no duplicate publish.
- **D. Rank symmetry.** Both ranks' logs show the same `resume` for every
  admitted prompt; no NCCL mismatch, no coordinator poison. A worker-side
  checkpoint mismatch fails the pair BEFORE GPU work (by construction);
  exercise the failure by staging a corrupted cache state only if a safe
  harness exists - otherwise rely on the host test covering the refusal.
- **Failure drill (optional but valuable).** With `PADDOCK_TP_CKPT_SLOTS=0`,
  every request must serve cold with no cache logging and no errors -
  proving the pages-only degrade path.
