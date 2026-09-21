export interface MountJob {
  /** One indivisible DOM quantum. Resolve after the framework's DOM flush.
   * Return true if this owner still has work; never wait for network/workers. */
  step: () => Promise<boolean>
  error: (error: unknown) => void
}
interface Clock {
  now: () => number
  frame: (callback: FrameRequestCallback) => number
  cancel: (id: number) => void
}

// One budget for the entire document, not one per message/lane. The FIFO map
// rotates after each quantum; replacing pending work never multiplies owners.
// Awaiting nextTick inside the quantum charges Vue's actual flush to the budget
// (merely timing the reactive assignment would miss almost all mount work).
export class MountScheduler {
  private jobs = new Map<string, MountJob>()
  private frame: number | null = null
  private running = false
  private spent = 0
  /** Optional bounded lab collector; no retained samples in the product. */
  trace?: (stage: string, started: number) => void
  private counters = { flushes: 0, quanta: 0, totalMs: 0, maxFrameMs: 0, maxQuantumMs: 0, over50: 0 }
  constructor(private clock: Clock = {
    now: () => performance.now(), frame: callback => requestAnimationFrame(callback), cancel: id => cancelAnimationFrame(id),
  }, private budgetMs = 6) {}
  get stats() { return { ...this.counters, owners: this.jobs.size, scheduled: this.frame !== null, running: this.running } }
  resetStats() { this.counters = { flushes: 0, quanta: 0, totalMs: 0, maxFrameMs: 0, maxQuantumMs: 0, over50: 0 } }
  schedule(owner: string, job: MountJob, eager = false) {
    this.jobs.set(owner, job)
    this.request()
    // Small warm edits need not wait an extra refresh interval. They debit the
    // same budget as cold mounts; repeated worker events cannot reset it.
    if (eager && !this.running && this.spent < this.budgetMs) {
      this.running = true
      void Promise.resolve().then(() => this.flush())
    }
  }
  cancel(owner: string) {
    this.jobs.delete(owner)
    if (!this.jobs.size && this.frame !== null) { this.clock.cancel(this.frame); this.frame = null }
  }
  private request() {
    if (this.running || this.frame !== null || !this.jobs.size) return
    this.frame = this.clock.frame(() => { this.frame = null; this.spent = 0; if (this.jobs.size) void this.flush() })
  }
  private async flush() {
    this.running = true
    const start = this.clock.now()
    const previouslySpent = this.spent
    try {
      do {
        const next = this.jobs.entries().next().value
        if (!next) break
        const [owner, job] = next
        const before = this.clock.now()
        let more = false
        try { more = await job.step() }
        catch (error) {
          if (this.jobs.get(owner) === job) this.jobs.delete(owner)
          job.error(error)
        }
        const ms = this.clock.now() - before
        this.trace?.('markdown-mount-quantum', before)
        this.counters.quanta++; this.counters.maxQuantumMs = Math.max(this.counters.maxQuantumMs, ms)
        // An in-flight quantum may have been cancelled/replaced during its
        // Vue flush. Never resurrect that old job or remove its replacement.
        if (this.jobs.get(owner) === job) {
          this.jobs.delete(owner)
          if (more) this.jobs.set(owner, job)
        }
        this.spent = previouslySpent + this.clock.now() - start
      } while (this.spent < this.budgetMs)
    } finally {
      const ms = this.clock.now() - start
      this.counters.flushes++; this.counters.totalMs += ms
      this.counters.maxFrameMs = Math.max(this.counters.maxFrameMs, this.spent)
      if (ms > 50) this.counters.over50++
      this.trace?.('markdown-mount-flush', start)
      this.running = false; this.request()
    }
  }
}
export const markdownMounts = new MountScheduler()
