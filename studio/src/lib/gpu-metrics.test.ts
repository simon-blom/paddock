import { describe, expect, it } from 'vitest'
import type { GpuSnapshot } from './api'
import { gpuMemoryPercent, gpuMetrics, metalAllocated } from './gpu-metrics'

const fixture = (): GpuSnapshot => ({
  available: true, ts: 100,
  gpus: [{ index: 0, name: 'Apple M5 Max', metal: { unified_memory_total: 128, recommended_working_set: 96 } }],
  reconciliation: { ts: 100, runners: [{ port: 11540, pid: 123, anomaly: false,
    metal: { allocated_bytes: 24, completed_commands: 8, gpu_seconds_total: 0.3, last_command_ms: 20 } }],
  attribution: false, device_used: 0, device_total: 0, anomaly: false },
})

describe('Metal telemetry semantics', () => {
  it('charts runner allocations against recommended working set, not physical RAM', () => {
    const s = fixture()
    expect(metalAllocated(s)).toBe(24)
    expect(gpuMemoryPercent(s.gpus[0], s)).toBe(25)
    expect(gpuMetrics(s.gpus[0], true).map(m => m.label)).toEqual(['Allocations', 'tok/s'])
  })
  it('does not offer unsupported sensors or manufacture zeroes for old/stale runners', () => {
    const s = fixture()
    s.reconciliation!.runners[0].metal = undefined
    expect(metalAllocated(s)).toBeNull()
    expect(gpuMemoryPercent(s.gpus[0], s)).toBeNull()
    s.reconciliation!.runners = []
    expect(metalAllocated(s)).toBe(0)
    s.ts = 111
    expect(metalAllocated(s)).toBeNull()
    s.reconciliation = null
    expect(metalAllocated(s)).toBeNull()
  })
  it('retains NVIDIA measurements including valid zero readings', () => {
    const s = fixture()
    const gpu = { index: 0, name: 'NVIDIA', util_gpu: 0, mem_used: 0, mem_total: 10, power_w: 20, temp_c: 40 }
    expect(gpuMetrics(gpu).map(m => m.key)).toEqual(['util', 'mem', 'power', 'temp'])
    expect(gpuMemoryPercent(gpu, s)).toBe(0)
  })
})
