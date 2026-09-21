import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { GpuSnapshot } from '@/lib/api'
import type { RunGpuEnv } from '@/types/chat'
import { gpuMemoryPercent } from '@/lib/gpu-metrics'

// Per-GPU rolling series for the sparklines, fed by the live WebSocket push
// (no polling). The server samples on its own thread and streams each new
// snapshot; we keep a short window for the charts.
interface Series {
  util: (number | null)[]
  memPct: (number | null)[]
  power: (number | null)[]
  temp: (number | null)[]
}
// Long enough for the zoomable history chart: ~6 min at the 400ms connected
// cadence (~30 min while idle at 2s). The live sparklines slice the recent tail.
const CAP = 900
const RECONNECT_MS = 2000

function push<T>(arr: T[], v: T): void {
  arr.push(v)
  if (arr.length > CAP) arr.shift()
}

/** GPU telemetry over a WebSocket. Connects while the dock is open and streams
 *  snapshots in real time - including mid-generation, since the server samples
 *  off the inference thread. No HTTP polling. */
export const useTelemetryStore = defineStore('telemetry', () => {
  const open = ref(localStorage.getItem('pk_gpu_dock') === '1')
  const snapshot = ref<GpuSnapshot | null>(null)
  const history = ref<Record<number, Series>>({})
  const connected = ref(false)
  const fresh = ref(false)

  const available = computed(() => fresh.value && (snapshot.value?.available ?? false))
  const gpus = computed(() => fresh.value ? snapshot.value?.gpus ?? [] : [])
  // The busiest live engine section across the fleet (the §9 gauge carries one
  // per runner) - what the dock's engine strip shows.
  const engineOf = (s: GpuSnapshot | null) =>
    s?.reconciliation?.runners?.find((r) => r.engine && r.engine.phase !== 'idle')?.engine
    ?? s?.reconciliation?.runners?.find((r) => r.engine)?.engine ?? null
  const engine = computed(() => fresh.value ? engineOf(snapshot.value) : null)
  /** VRAM the running models hold, from their allocator self-reports. */
  const modelMem = computed(() =>
    (snapshot.value?.reconciliation?.runners ?? []).reduce((s, r) => s + (r.self_mem ?? 0), 0),
  )
  const tokHistory = ref<(number | null)[]>([])
  // Unix seconds (client clock, sub-second) aligned with every history series -
  // the x-axis for the uPlot history chart.
  const times = ref<number[]>([])

  let ws: WebSocket | null = null
  let reconnect: number | undefined
  // The stream should be live when the dock is open OR a run is being recorded
  // (so per-turn GPU capture works even with the dock closed). `holds` counts
  // active runs; `desired()` is the connection intent.
  let holds = 0
  const desired = () => open.value || holds > 0

  // Per-run peak aggregator (null when not capturing).
  let runAgg: RunGpuEnv | null = null

  let identity: string | undefined
  let lastReceipt: number | undefined
  let interrupted = false
  let serverTimestamp: number | undefined
  let serverAdvancedAt = 0

  function append(s: GpuSnapshot | null, time: number): void {
    push(times.value, time)
    const recon = s?.reconciliation
    const complete = recon && Math.abs(s!.ts - recon.ts) <= 10 && recon.runners.length > 0
      && recon.runners.every((r) => r.engine)
    push(tokHistory.value, complete ? recon.runners.reduce((sum, r) => sum + (r.engine!.phase === 'idle' ? 0 : r.engine!.tok_s), 0) : null)
    const indices = new Set([...Object.keys(history.value).map(Number), ...(s?.gpus.map((g) => g.index) ?? [])])
    for (const index of indices) {
      const g = s?.gpus.find((g) => g.index === index)
      const empty = () => Array<number | null>(times.value.length - 1).fill(null)
      const h = history.value[index] ?? { util: empty(), memPct: empty(), power: empty(), temp: empty() }
      push(h.util, g?.util_gpu ?? null)
      push(h.memPct, g && s ? gpuMemoryPercent(g, s) : null)
      push(h.power, g?.power_w ?? null)
      push(h.temp, g?.temp_c ?? null)
      history.value[index] = h
    }
  }

  function ingest(s: GpuSnapshot): void {
    snapshot.value = s
    const now = Date.now() / 1000
    // The web UI can run on another machine: never compare client/server
    // wall clocks for freshness. Detect a frozen sampler relative to receipt.
    if (s.ts !== serverTimestamp || interrupted || now < serverAdvancedAt) {
      serverTimestamp = s.ts
      serverAdvancedAt = now
    }
    const key = s.gpus.map((g) => `${g.index}:${g.name}`).join(',') + '|'
      + (s.reconciliation?.runners.map((r) => `${r.port}:${r.pid}`).sort().join(',') ?? '')
    if (lastReceipt != null && now <= lastReceipt) {
      // Clock rollback must not leave a non-monotonic chart axis.
      times.value = []; history.value = {}; tokHistory.value = []
    } else if (lastReceipt != null && (interrupted || key !== identity || now - lastReceipt > 12)) {
      append(null, lastReceipt + (now - lastReceipt) / 2)
    }
    fresh.value = now - serverAdvancedAt <= 10 && s.available
    append(fresh.value ? s : null, now)
    lastReceipt = now; identity = key; interrupted = false
    if (runAgg && fresh.value) accumulate(s)
  }

  function accumulate(s: GpuSnapshot): void {
    if (!runAgg) return
    const g = s.gpus[0]
    if (g) {
      runAgg.device = g.name
      if (g.util_gpu != null) runAgg.utilPeak = Math.max(runAgg.utilPeak ?? 0, g.util_gpu)
      if (g.mem_used != null) runAgg.memUsedPeak = Math.max(runAgg.memUsedPeak ?? 0, g.mem_used)
      if (g.mem_total) runAgg.memTotal = g.mem_total
      if (g.power_w != null) runAgg.powerPeakW = Math.max(runAgg.powerPeakW ?? 0, g.power_w)
      if (g.temp_c != null) runAgg.tempPeakC = Math.max(runAgg.tempPeakC ?? 0, g.temp_c)
    }
    const e = engineOf(s)
    if (e) {
      runAgg.tokSPeak = Math.max(runAgg.tokSPeak ?? 0, e.tok_s)
      runAgg.batchPeak = Math.max(runAgg.batchPeak ?? 0, e.active_slots)
      runAgg.kvPeak = Math.max(runAgg.kvPeak ?? 0, e.kv_used)
      if (e.kv_total) runAgg.kvTotal = e.kv_total
    }
  }

  function streamUrl(): string {
    // Dev: connect straight to the server (VITE_API_WS) - Vite's proxy can't
    // upgrade WS beside its HMR socket. Prod: same-origin (paddock serves us).
    const base = import.meta.env.VITE_API_WS as string | undefined
    if (base) return `${base}/api/gpu/stream`
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${location.host}/api/gpu/stream`
  }

  function connect(): void {
    if (ws || !desired()) return
    const sock = new WebSocket(streamUrl())
    ws = sock
    sock.onopen = () => {
      connected.value = true
    }
    sock.onmessage = (ev) => {
      try {
        ingest(JSON.parse(ev.data as string) as GpuSnapshot)
      } catch {
        /* ignore a malformed frame */
      }
    }
    sock.onclose = () => {
      interrupted = true
      fresh.value = false
      connected.value = false
      if (ws === sock) ws = null
      // Reconnect while still wanted (dock open or a run holds it).
      if (desired() && reconnect === undefined) {
        reconnect = window.setTimeout(() => {
          reconnect = undefined
          connect()
        }, RECONNECT_MS)
      }
    }
    sock.onerror = () => sock.close()
  }

  function disconnect(): void {
    interrupted = true
    fresh.value = false
    if (reconnect !== undefined) {
      clearTimeout(reconnect)
      reconnect = undefined
    }
    if (ws) {
      ws.onclose = null
      ws.close()
      ws = null
    }
    connected.value = false
  }

  // Reconcile the socket with the current intent (open OR a run holds it).
  function sync(): void {
    if (desired()) connect()
    else disconnect()
  }

  function setOpen(v: boolean): void {
    open.value = v
    localStorage.setItem('pk_gpu_dock', v ? '1' : '0')
    sync()
  }
  function toggle(): void {
    setOpen(!open.value)
  }

  // ── per-turn run capture (lab-notebook: the GPU env each answer ran under) ──
  /** Begin recording GPU/engine peaks for one assistant turn; connects the
   *  stream even if the dock is closed. */
  function beginCapture(): void {
    holds += 1
    runAgg = {}
    sync()
  }
  /** Finish recording; returns the peaks and drops the stream if nothing else
   *  wants it. undefined when nothing was sampled. */
  function endCapture(): RunGpuEnv | undefined {
    const agg = runAgg
    runAgg = null
    holds = Math.max(0, holds - 1)
    sync()
    return agg && Object.keys(agg).length ? agg : undefined
  }

  // Resume streaming if the dock was left open last session.
  if (open.value) connect()

  return {
    open,
    snapshot,
    history,
    tokHistory,
    times,
    connected,
    available,
    gpus,
    engine,
    modelMem,
    setOpen,
    toggle,
    beginCapture,
    endCapture,
  }
})
