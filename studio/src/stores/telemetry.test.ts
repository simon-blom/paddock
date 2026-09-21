import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useTelemetryStore } from './telemetry'

class Socket {
  static latest: Socket
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  constructor() { Socket.latest = this }
  close() {}
}
beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date(100_000))
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => {} })
  vi.stubGlobal('location', { protocol: 'http:', host: 'localhost' })
  vi.stubGlobal('WebSocket', Socket)
})
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

it('retains absent sensors as chart gaps and never records fake zero peaks', () => {
  const store = useTelemetryStore()
  store.beginCapture()
  Socket.latest.onmessage!({ data: JSON.stringify({ available: true, ts: 100,
    gpus: [{ index: 0, name: 'Apple M5 Max', metal: { unified_memory_total: 128, recommended_working_set: 96 } }],
  }) })
  expect(store.history[0].util).toEqual([null])
  expect(store.history[0].memPct).toEqual([null])
  expect(store.history[0].power).toEqual([null])
  expect(store.endCapture()).toEqual({ device: 'Apple M5 Max' })
})

it('splits restart and sleep histories and keeps missing-device series aligned', () => {
  const store = useTelemetryStore()
  store.setOpen(true)
  const send = (ts: number, pid: number, present = true) => {
    vi.setSystemTime(new Date(ts * 1000))
    Socket.latest.onmessage!({ data: JSON.stringify({ available: true, ts,
      gpus: present ? [{ index: 0, name: 'Apple M5 Max', metal: { recommended_working_set: 100 } }] : [],
      reconciliation: { ts, runners: [{ port: 11540, pid, metal: { allocated_bytes: 20 } }] },
    }) })
  }
  send(100, 1); send(102, 2); send(200, 2); send(202, 2, false)
  expect(store.history[0].memPct).toEqual([20, null, 20, null, 20, null, null])
  expect(store.history[0].memPct.length).toBe(store.times.length)
  expect(store.tokHistory.every((v) => v === null)).toBe(true)
  send(90, 2)
  expect(store.times).toEqual([90])
  expect(store.history[0].memPct).toEqual([20])
  for (let ts = 91; ts < 1100; ts++) send(ts, 2)
  expect(store.times.length).toBe(900)
  expect(store.history[0].memPct.length).toBe(900)
  store.setOpen(false)
})

it('accepts remote clock skew but gaps a frozen sampler', () => {
  const store = useTelemetryStore()
  store.setOpen(true)
  const data = JSON.stringify({ available: true, ts: 900000,
    gpus: [{ index: 0, name: 'remote GPU', util_gpu: 75 }],
  })
  Socket.latest.onmessage!({ data })
  expect(store.history[0].util).toEqual([75])
  vi.setSystemTime(new Date(115000))
  Socket.latest.onmessage!({ data })
  expect(store.history[0].util.slice(-1)[0]).toBeNull()
  store.setOpen(false)
})
