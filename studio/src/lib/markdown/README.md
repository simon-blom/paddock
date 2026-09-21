# Studio Markdown runtime

`components/chat/Markdown.vue` owns product settings and the map extension.
`MarkdownContent.vue` is the store-free production renderer, also mounted by
the native WebKit Rendering Lab. There is no lab-only highlighting adapter.

## Work ownership

- **Parse worker:** one shared, lazy module worker uses the pinned full-document
  `stream-markdown-parser`. Streaming parses use its incremental cache. Final
  parses use a fresh parser without the stream cache. Exact serialized AST
  equality identifies reusable leading nodes; global references may invalidate
  that prefix. A response carries its base/revision, changed suffix and length.
- **UI:** apply a revision-checked patch to a shallow, immutable AST. Groups of
  64 nodes retain their array identity when unchanged, so settled components do
  not update with every token. A **single document-wide scheduler** gives every
  message/lane a turn and budgets 6 ms per frame, including the awaited Vue DOM
  flush. One group is the indivisible quantum, so a complex group can exceed
  that soft budget. Small warm edits may run immediately, debiting the same
  budget rather than paying an extra frame of latency. Immutable group entries
  keep unchanged child props stable; each group owns its final flag. Explicit
  `v-memo` was tested and removed after adverse reopen-memory measurements.
  Markstream's supported `nodes`/fragment API renders these groups.
- **Mounting is progressive, not virtualization.** Every group eventually
  mounts and stays mounted; there is no viewport window or content cap. Initial
  parsing shows a bounded loading status, not a second layout of the entire raw
  source. `aria-busy` / `data-markdown-state=pending` stay set until all groups
  commit. Existing rich DOM stays visible while a replacement parses/mounts.
  Native Find/selection see the mounted content; complete-content readiness is
  measured separately from first rich content. Full source remains available
  to product copy/export commands, and error/oversize fallback is complete text.
- **Syntax worker:** a second shared module worker owns Shiki and its regex
  engine. Grammars load on demand; both themes share the engine. Closed fences
  are highlighted, while unfinished fences remain selectable plain code.
  Unified diffs display/copy the raw patch, not only the parser's updated side.
- **Math, Mermaid, maps:** continue through the component extensions. Moving
  the Markdown parser does not move SVG/DOM rendering into a worker. Maps are
  parsed fence-language extensions; never split the source around fences.

## Backpressure and lifecycle

`WorkerRpc` keeps one physical job active and at most one queued job per owner.
A newer request rejects the obsolete caller but cannot pretend its synchronous
worker computation stopped. The physical slot/byte accounting remain occupied
until reply, failure or watchdog termination. Only current revisions apply.
If a cancelled reply changed the worker's cache, the next request's base revision
forces a full resynchronization rather than applying a patch to the wrong AST.

Admission callbacks retry from **current component props**. They contain no
copied input payload and are cancelled on replacement/unmount. Notifications
are completion-driven, not microtask polling that could starve worker replies.
Parser leases are acquired only for nonempty content inside the rich-rendering
budget. Clearing a view, switching to oversized plain text, or unmounting releases
its lease and cancels mounting/admission. An empty sibling cannot keep a worker
alive; releasing the last populated consumer terminates it. Reopening acquires a
fresh owner/base revision, and old replies cannot affect that lease. If populated
consumers remain, the existing 15-second idle policy still releases the worker.

Native WebKit profiling found a repeatable ~1-second round-trip
after a long idle, outside the parser's compute span, with CFRunLoop waiting in
the sampled worker stack. A one-second idle policy removed that delay but its
candidate failed repeated-memory qualification. Later tests did not isolate
worker lifetime as the sole memory cause. The previous component kept an empty
owner during stream reopen tests, preserving the parser across cycles despite
having no content left to parse. Lazy leases correct that ownership mistake;
the lab's empty view itself and the workload are unchanged.
A 5-second idle policy was also rejected after the unlocked tests showed
worse fifth-cycle memory. The current design retains 15 seconds and targets an
unresponsive job instead.

The parser now acknowledges **starting** each job on an already-used worker.
If that ACK has not arrived after 100 ms, terminate the old worker and retry the
same input **once** on a fresh one. Acknowledged parsing retains the normal
15-second watchdog; a legitimately expensive parse is not killed by the start
deadline. Fresh module startup/retries also retain that normal deadline. No
keepalive polling or additional worker pool is created; healthy caches survive.
Input stays inside the existing byte/job accounting until ACK. Cancellation
never replays a superseded caller; stale-port replies cannot commit. An ACK can
race termination, so this option is explicitly limited to replay-safe rendering
computations, not database writes. This does not prove every WebKit wake-up issue
fixed, nor qualify the syntax worker's idle behavior (it has no start recovery).

A 15-second watchdog terminates a stuck worker and recovers queued
work. These workers contain only disposable rendering state, not a database.

Mount jobs are latest-only per owner and rotate fairly. Small atomic warm edits
keep the previous rich revision readable until their commit rather than forcing
two whole-parent Vue flushes to toggle busy for every token. Multi-group changes
remain busy until complete. Replacement/cancellation
during a Vue flush cannot resurrect a job; unmount removes its work. Stable
groups are not removed/reinserted, preserving selection and scroll in settled
history. Sticky scrolling also rechecks the user's pin state when its queued
frame executes, so a later user scroll wins.

| Budget | Limit / behavior |
| --- | --- |
| Per worker's admitted jobs | 32 including the physical active job; 8 MiB estimated UTF-16 input bytes |
| Deferred admission | 512 payload-free owner callbacks; beyond this, explicit plain fallback |
| Markdown source | 1,048,576 UTF-16 code units; complete plain text beyond this |
| One parsed result | 16,384 top-level nodes / 8 MiB serialized-AST accounting |
| Streaming parser cache | LRU, 16 sessions / 16 MiB estimated source + AST; final sessions discarded |
| Syntax input | 131,072 code units, maximum line 8,192; complete plain code beyond this |
| Syntax output | 1,048,576 code units; complete plain code beyond this |
| Syntax cache | LRU, 32 entries / 4 MiB source-key + HTML accounting |
| Grammar residency | Finite allowed set, lazy loaded; recycle engine when 32 loaded language IDs are reached before another load |

These are admission/cache/output budgets, **not hard whole-process heap limits**.
The Markdown parser necessarily creates its AST before the output-size check.
The DOM, dependency internals, GPU allocations and WebKit caches are separate.
This is not constant-time streaming: source snapshots and worker-side exact
signature comparison scale with document size; UI grouping still scans node
references. Only unchanged component subtrees avoid updating.
The frame budget does not preempt a single enormous table/paragraph or later
asynchronous KaTeX/Mermaid work, and does not cover graph construction, browser
style/layout/GPU execution or other applications. It is not a 6-ms global
main-thread latency guarantee. Background tabs rely on normal rAF suspension;
we do not run a timer loop to force painting invisible content.

Model HTML is escaped. `v-html` accepts only escaped Shiki output from the
bundled syntax worker, never model-supplied HTML. Parser/worker errors surface
as complete plain text; no synchronous main-thread fallback retries expensive
parsing. Invalid maps remain code. Remote map tiles still require the existing
explicit user action in `PhotoLocation`.

## Validation

From `studio`: `npm test` (includes the production-component lifecycle tests),
then `node --test renderer-lab/*.test.mjs` for the separate diagnostic suites.
Native parity, admission, security and mixed-document tests use the actual
shared component, not a mock Markdown implementation.

What the native measurements settled, and what they did not: the 64-node
grouping passes automated parity and the Markdown latency targets, and
Markdown scrolling layers are stable. Memory after a mixed graph/Markdown
reopen is still open - a closed-owner parser-cache release and the shorter
worker idle policies were each tried and removed after they made repeated or
fresh-launch memory worse. None of this is a whole-process memory claim.

Design references: Markstream's [performance guide](https://markstream.simonhe.me/guide/performance),
[2.0 migration](https://markstream.simonhe.me/guide/migration-2-0) and
[renderer API](https://markstream.simonhe.me/guide/props), plus Streamdown's
[block memoization](https://streamdown.ai/docs/memoization). We use the library's
document grammar and extension APIs; queueing, revisioned transport and immutable
grouping are Paddock code. This is not an implementation copied from a rival.

Scheduling references: Vue's [update/memoization guidance](https://vuejs.org/guide/best-practices/performance.html),
WebKit's [rendering-frame profiling](https://webkit.org/blog/3996/introducing-the-rendering-frames-timeline/),
Sigma's [process/render lifecycle](https://www.sigmajs.org/docs/advanced/lifecycle/),
and WebKit's [worker run loop](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/workers/WorkerRunLoop.cpp)
(the Cocoa path can enter a one-second CFRunLoop wait for a stale timer).
