import { execFileSync } from 'node:child_process'

// Repeat plans already include their accepted idle inside the fixture. The
// fresh graph plan owns it in the external runner: capture must wait for that
// original observation too, never use an earlier/later census as its idle bar.
export function finalMemoryIdleSeconds(plan, idleSeconds) {
  if (plan === 'scalegraphrepeat') return 0
  if (plan === 'scalegraph' && idleSeconds === 10) return 10
  throw new Error('--trace-final-memory requires scalegraphrepeat or scalegraph with the original 10-second idle')
}

/** One allocation inspection in the final second of each original idle phase.
 * Never move a stale/missed checkpoint into the following cycle's workload. */
export function reopenCheckpoint(profile, seen, now = Date.now()) {
  const event = profile?.events?.at(-1)
  if (!event || profile.phase !== event.phase || !/^graph-cycle-[1-5]\/idle-10s$/.test(event.phase)
    || seen.has(event.phase) || now - event.at < 9000 || now - event.at >= 10000) return null
  return event.phase
}

// A missing key is not proof of foreground eligibility. A positive lock report
// is enough to refuse a visual benchmark; never bypass rAF suspension on lock.
export const hasScreenLock = text => /"CGSSessionScreenIsLocked"\s*=\s*Yes\b/.test(text)
export function screenLockState() {
  try {
    const text = execFileSync('/usr/sbin/ioreg', ['-l', '-n', 'Root', '-d', '1'], { encoding: 'utf8', timeout: 3000, maxBuffer: 2 * 1024 ** 2 })
    return { at: Date.now(), locked: hasScreenLock(text) }
  } catch (error) { return { at: Date.now(), locked: null, error: error.message } }
}
