// One physical job per worker. Superseding/cancelling a caller does not free
// that slot: synchronous WASM/tokenization can still be running in the worker.
export interface RpcPort {
  onmessage: ((event: MessageEvent) => void) | null
  onerror: ((event: ErrorEvent) => void) | null
  onmessageerror: ((event: MessageEvent) => void) | null
  postMessage(message: unknown): void
  terminate(): void
}
interface Job<O> {
  id: number; owner: string; input: unknown; bytes: number; settled: boolean
  queuedAt?: number; startedAt?: number
  acknowledged?: boolean; retried?: boolean
  resolve(value: O): void; reject(error: Error): void
}
interface RpcOptions {
  maxJobs: number; maxBytes: number; timeoutMs: number; idleMs: number
  /** Only for replay-safe computations with a start-ACK protocol. An ACK can
   * race with termination; this is not suitable for mutating database RPCs. */
  startTimeoutMs?: number
}
const cancelled = () => new DOMException('Superseded or disposed', 'AbortError')
export const isCancelled = (error: unknown) => error instanceof Error && error.name === 'AbortError'
export class RendererBusyError extends Error {
  constructor() { super('Renderer work budget is busy'); this.name = 'RendererBusyError' }
}

export class WorkerRpc<I, O> {
  /** Opt-in lab attribution; clocks and callbacks are absent on normal calls. */
  trace?: (stage: string, started: number) => void
  private port?: RpcPort
  private active?: Job<O>
  private queue = new Map<string, Job<O>>()
  private owners = new Set<string>()
  // Admission notifications retain only a callback, never an input payload or
  // a promise. A component retries from its current props when capacity frees.
  private waiting = new Map<string, () => void>()
  private sequence = 0
  private ownerSequence = 0
  private timer?: ReturnType<typeof setTimeout>
  private idle?: ReturnType<typeof setTimeout>
  private startTimer?: ReturnType<typeof setTimeout>
  private hasReplied = false
  private recoveries = 0
  constructor(private create: () => RpcPort, private options: RpcOptions = {
    maxJobs: 32, maxBytes: 8 * 1024 * 1024, timeoutMs: 15000, idleMs: 15000,
  }) {}

  open(): string {
    const owner = `md-${++this.ownerSequence}`
    this.owners.add(owner)
    return owner
  }
  get stats() {
    return { workers: this.port ? 1 : 0, active: this.active ? 1 : 0,
      queued: this.queue.size, bytes: this.bytes(), owners: this.owners.size, waiting: this.waiting.size, recoveries: this.recoveries }
  }
  private bytes() {
    let bytes = this.active?.bytes ?? 0
    for (const job of this.queue.values()) bytes += job.bytes
    return bytes
  }
  private reject(job: Job<O> | undefined, error: Error) {
    if (!job || job.settled) return
    job.settled = true; job.reject(error)
  }
  cancel(owner: string) {
    this.waiting.delete(owner)
    this.reject(this.queue.get(owner), cancelled()); this.queue.delete(owner)
    if (this.active?.owner === owner) this.reject(this.active, cancelled())
  }
  whenAvailable(owner: string, retryCurrentProps: () => void): boolean {
    if (!this.owners.has(owner) || this.waiting.size >= 512) return false
    this.waiting.set(owner, retryCurrentProps)
    // Do not poll a busy worker with microtasks: spare bytes may still be too
    // few for this owner, and a retry loop would starve the worker's reply.
    if (!this.active && !this.queue.size) this.notifyCapacity()
    return true
  }
  private notifyCapacity() {
    if (this.queue.size + (this.active ? 1 : 0) >= this.options.maxJobs || this.bytes() >= this.options.maxBytes) return
    const next = this.waiting.entries().next().value
    if (!next) return
    const [owner, retry] = next
    // Keep the token until its microtask runs so disposal/supersession cancels
    // it too. Delete-before-callback permits fair requeue under byte pressure.
    queueMicrotask(() => {
      if (this.waiting.get(owner) !== retry || !this.owners.has(owner)) return
      this.waiting.delete(owner); retry()
    })
  }
  close(owner: string) {
    this.cancel(owner); this.owners.delete(owner)
    // No consumers remain. Unlike caller-only cancellation, terminating the
    // unused worker really does release its parser/highlighter and WASM heap.
    if (!this.owners.size) this.stop()
  }
  submit(owner: string, input: I, bytes: number): Promise<O> {
    if (!this.owners.has(owner)) return Promise.reject(cancelled())
    this.cancel(owner)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > this.options.maxBytes) {
      return Promise.reject(new Error('Renderer work budget exceeded; showing plain text'))
    }
    if (bytes + this.bytes() > this.options.maxBytes || this.queue.size + (this.active ? 1 : 0) >= this.options.maxJobs) return Promise.reject(new RendererBusyError())
    clearTimeout(this.idle)
    return new Promise<O>((resolve, reject) => {
      this.queue.set(owner, { id: ++this.sequence, owner, input, bytes, resolve, reject, settled: false, queuedAt: this.trace ? performance.now() : undefined })
      this.pump()
    })
  }
  private stop() {
    clearTimeout(this.timer); clearTimeout(this.idle); clearTimeout(this.startTimer)
    if (this.port) {
      this.port.onmessage = null; this.port.onerror = null; this.port.onmessageerror = null
      this.port.terminate(); this.port = undefined
    }
    this.active = undefined
    this.hasReplied = false
  }
  private fail(error: Error) {
    this.reject(this.active, error); this.stop(); this.pump(); this.notifyCapacity()
  }
  private recoverStart(job: Job<O>) {
    if (this.active !== job || job.acknowledged || job.retried) return
    // The worker has not acknowledged starting this disposable computation.
    // Termination makes the old physical slot genuinely free. Retry once with
    // retained input; never replay an acknowledged (potentially expensive) job.
    // A late start/ACK can race the timer, so this is for replay-safe work only.
    if (job.startedAt !== undefined) this.trace?.('worker-start-recovery', job.startedAt)
    this.stop(); this.recoveries++
    if (!job.settled) {
      job.retried = true
      this.queue = new Map([[job.owner, job], ...this.queue])
    }
    this.pump(); this.notifyCapacity()
  }
  private pump() {
    if (this.active) return
    const next = this.queue.entries().next().value
    if (!next) {
      clearTimeout(this.idle)
      if (this.port) this.idle = setTimeout(() => this.stop(), this.options.idleMs)
      return
    }
    const [owner, job] = next
    this.queue.delete(owner); this.active = job
    try {
      if (!this.port) {
        const port = this.create(); this.port = port
        port.onmessage = event => {
          if (this.port !== port) return
          const active = this.active
          if (!active || event.data?.id !== active.id) { this.fail(new Error('Invalid renderer worker response')); return }
          if (event.data.started === true) {
            active.acknowledged = true; active.input = undefined
            clearTimeout(this.startTimer)
            if (active.startedAt !== undefined) this.trace?.('worker-start-wait', active.startedAt)
            return
          }
          clearTimeout(this.startTimer); this.hasReplied = true
          if (active.startedAt !== undefined) this.trace?.('worker-roundtrip', active.startedAt)
          // Parser supplies its own synchronous compute duration. Place this
          // diagnostic at reply time; this is not clock-synchronized tracing.
          if (this.trace && typeof event.data.value?.parseMs === 'number') this.trace('worker-compute-duration', performance.now() - event.data.value.parseMs)
          clearTimeout(this.timer)
          if (event.data.error) this.reject(active, new Error(String(event.data.error)))
          else if (!active.settled) { active.settled = true; active.resolve(event.data.value as O) }
          this.active = undefined; this.pump(); this.notifyCapacity()
        }
        port.onerror = event => { event.preventDefault?.(); if (this.port === port) this.fail(new Error(event.message || 'Renderer worker failed')) }
        port.onmessageerror = () => { if (this.port === port) this.fail(new Error('Renderer worker message failed')) }
      }
      this.timer = setTimeout(() => this.fail(new Error('Renderer worker timed out; showing plain text')), this.options.timeoutMs)
      if (job.queuedAt !== undefined) this.trace?.('worker-queue-wait', job.queuedAt)
      if (this.trace) job.startedAt = performance.now()
      const acknowledge = !!this.options.startTimeoutMs && this.hasReplied && !job.retried
      if (acknowledge) this.startTimer = setTimeout(() => this.recoverStart(job), this.options.startTimeoutMs)
      this.port.postMessage({ id: job.id, owner: job.owner, input: job.input, acknowledge })
      // Keep retry input only until the start acknowledgement. It remains
      // inside the existing byte/job budget, including caller cancellation.
      if (!acknowledge) job.input = undefined
    } catch (error) { this.fail(error instanceof Error ? error : new Error(String(error))) }
  }
}
