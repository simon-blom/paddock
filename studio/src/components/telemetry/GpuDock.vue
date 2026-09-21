<script setup lang="ts">
// Right-hand GPU telemetry dock. Live per-GPU cards (util / VRAM / power / temp)
// as glanceable gauges, plus an ECharts history chart with a metric selector.
// Updates while a chat runs; every metric is capability-aware (null -> hidden).
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { GpuInfo } from '@/lib/api'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import GaugeBar from './GaugeBar.vue'
import HistoryChart from './HistoryChart.vue'
import { fmtVram } from '@/lib/format'
import { gpuMetrics, metalAllocated, type MetricKey } from '@/lib/gpu-metrics'

const tele = useTelemetryStore()

function gb(bytes?: number | null): string {
  if (!bytes && bytes !== 0) return '-'
  return fmtVram(bytes)
}
function memPct(g: GpuInfo): number {
  return g.mem_used && g.mem_total ? (g.mem_used / g.mem_total) * 100 : 0
}
const gpus = computed(() => tele.gpus)

// ── history chart: metric selector over the retained window ──────────────────
const METRICS = computed(() => gpuMetrics(gpus.value[0], !!tele.engine))
const metric = ref<MetricKey>('util')
const activeMetric = computed(() => METRICS.value.find((m) => m.key === metric.value) ?? METRICS.value[0])
const allocated = computed(() => tele.snapshot ? metalAllocated(tele.snapshot) : null)
const runners = computed(() => tele.snapshot?.reconciliation?.runners ?? [])
// The GPU whose series we chart: the first (per-runner GPU pinning is a
// multi-GPU follow-up).
const chartIndex = computed(() => gpus.value[0]?.index ?? 0)
const chartValues = computed<(number | null)[]>(() => {
  const key = activeMetric.value?.key
  if (key === 'tok') return tele.tokHistory
  const h = tele.history[chartIndex.value] ?? { util: [], memPct: [], power: [], temp: [] }
  return key === 'util'
    ? h.util
    : key === 'mem'
      ? h.memPct
      : key === 'power'
        ? h.power
        : h.temp
})
</script>

<template>
  <aside class="gpudock">
    <header class="gpudock__head">
      <span class="gpudock__title">
        <Icon name="graphics-card" :size="16" /> GPU
        <Tooltip :label="tele.connected ? 'Live' : 'Reconnecting...'">
          <span class="gpudock__dot" :class="{ 'gpudock__dot--live': tele.connected }" />
        </Tooltip>
      </span>
      <button class="pk-icon-btn" aria-label="Close GPU panel" @click="tele.setOpen(false)">
        <Icon name="x" :size="16" />
      </button>
    </header>

    <div class="gpudock__body">
      <!-- Engine strip: what the GPU is doing for the current chat. Lights up
           while generating; the differentiator over raw device metrics. -->
      <div
        v-if="tele.engine"
        class="estrip"
        :class="{ 'estrip--busy': tele.engine.phase !== 'idle' }"
      >
        <div class="estrip__top">
          <span class="estrip__phase" :class="`estrip__phase--${tele.engine.phase}`">
            {{ tele.engine.phase }}
          </span>
          <span class="estrip__tok">
            <strong>{{ Math.round(tele.engine.tok_s) }}</strong> tok/s
          </span>
        </div>
        <div class="estrip__stats">
          <span v-if="tele.engine.active_slots">batch {{ tele.engine.active_slots }}</span>
          <span v-if="tele.engine.kv_total">
            KV {{ tele.engine.kv_used }}/{{ tele.engine.kv_total }}
          </span>
          <span>{{ tele.engine.tokens_total.toLocaleString() }} tok total</span>
        </div>
        <GaugeBar
          v-if="tele.engine.kv_total"
          :value="tele.engine.kv_used"
          :max="tele.engine.kv_total"
          tone="power"
        />
      </div>

      <div v-if="!tele.available && !tele.engine" class="gpudock__empty">
        <Icon name="graphics-card" :size="24" />
        <p v-if="!tele.connected">Connecting to the server...</p>
        <p v-else>GPU measurements are unavailable on this server.</p>
      </div>
      <!-- NVML absent but a model is loaded (e.g. a container without the mgmt
           lib): engine metrics still stream; note that device metrics are off. -->
      <div v-else-if="!tele.available" class="gpudock__note">
        GPU stats aren't available on this server.
      </div>

      <div v-for="g in gpus" :key="g.index" class="gcard">
        <div class="gcard__head">
          <Tooltip :label="g.name">
            <span class="gcard__name">{{ g.name }}</span>
          </Tooltip>
        </div>
        <!-- The running models' own allocator ledgers (summed across the
             fleet) - works where NVML can't attribute per-process VRAM. -->
        <div v-if="!g.metal && g.index === chartIndex && tele.modelMem" class="gcard__paddockmem">
          Models use {{ gb(tele.modelMem) }}
        </div>

        <!-- Utilization -->
        <template v-if="g.metal">
          <div class="metric">
            <div class="metric__top"><span class="metric__label">Thermal pressure</span><span class="metric__val">{{ g.metal.thermal_pressure ?? 'Unavailable' }}</span></div>
            <Tooltip label="macOS reports memory-pressure changes. No reading is assumed before the first OS event."><div class="metric__top"><span class="metric__label">Memory pressure</span><span class="metric__val">{{ g.metal.memory_pressure ?? 'Not reported' }}</span></div></Tooltip>
          </div>
          <div class="metric">
            <div class="metric__top"><span class="metric__label">Unified memory</span><span class="metric__val">{{ gb(g.metal.unified_memory_total) }}</span></div>
            <Tooltip label="Apple’s recommended working-set size; shared with other applications, not dedicated VRAM."><div class="metric__top"><span class="metric__label">Metal working set</span><span class="metric__val">{{ gb(g.metal.recommended_working_set) }}</span></div></Tooltip>
          </div>
          <div v-if="allocated != null" class="metric">
            <div class="metric__top"><span class="metric__label">Paddock allocations</span><span class="metric__val">{{ gb(allocated) }}</span></div>
            <GaugeBar :value="allocated" :max="g.metal.recommended_working_set" />
          </div>
          <div v-for="r in runners" :key="`${r.port}:${r.pid}`" class="metric">
            <div class="metric__top"><span class="metric__label">Runner · {{ r.port }}</span><span class="metric__val">{{ gb(r.metal?.allocated_bytes) }}</span></div>
            <template v-if="r.metal">
              <Tooltip label="Sum of completed command-buffer GPU spans since runner start. Spans may overlap; this is not device utilization."><div class="metric__top"><span class="metric__label">GPU time · cumulative</span><span class="metric__val">{{ r.metal.gpu_seconds_total.toFixed(2) }} s</span></div></Tooltip>
              <div v-if="r.metal.last_command_ms != null" class="metric__top"><span class="metric__label">Last GPU command</span><span class="metric__val">{{ r.metal.last_command_ms.toFixed(2) }} ms</span></div>
            </template>
            <span v-else class="metric__label">Measurements unavailable · restart with the updated runner</span>
          </div>
        </template>
        <div v-if="g.util_gpu != null" class="metric">
          <div class="metric__top">
            <span class="metric__label">Utilization</span>
            <span class="metric__val">{{ g.util_gpu }}%</span>
          </div>
          <GaugeBar :value="g.util_gpu" :max="100" />
        </div>

        <!-- VRAM -->
        <div v-if="g.mem_total != null" class="metric">
          <div class="metric__top">
            <span class="metric__label">Memory</span>
            <span class="metric__val">{{ gb(g.mem_used) }} <span class="metric__sub">/ {{ gb(g.mem_total) }}</span></span>
          </div>
          <GaugeBar :value="memPct(g)" :max="100" />
        </div>

        <!-- Power -->
        <div v-if="g.power_w != null" class="metric">
          <div class="metric__top">
            <span class="metric__label">Power</span>
            <span class="metric__val">
              {{ Math.round(g.power_w) }} W
              <span v-if="g.power_limit_w" class="metric__sub">/ {{ Math.round(g.power_limit_w) }} W</span>
            </span>
          </div>
          <GaugeBar :value="g.power_w" :max="g.power_limit_w ?? 0" tone="power" />
        </div>

        <!-- Temperature -->
        <div v-if="g.temp_c != null" class="metric">
          <div class="metric__top">
            <span class="metric__label">Temperature</span>
            <span class="metric__val">{{ g.temp_c }}°C</span>
          </div>
          <GaugeBar :value="g.temp_c" :max="100" tone="temp" />
        </div>

        <!-- clocks / fan strip -->
        <div class="gcard__stats">
          <Tooltip v-if="g.sm_clock_mhz != null" label="SM clock">
            <span>SM {{ g.sm_clock_mhz }} MHz</span>
          </Tooltip>
          <Tooltip v-if="g.mem_clock_mhz != null" label="Memory clock">
            <span>MEM {{ g.mem_clock_mhz }} MHz</span>
          </Tooltip>
          <Tooltip v-if="g.fan_pct != null" label="Fan">
            <span>Fan {{ g.fan_pct }}%</span>
          </Tooltip>
        </div>
      </div>

      <!-- History: ECharts area chart over the retained window, metric-selectable -->
      <div v-if="activeMetric" class="hist">
        <div class="hist__tabs">
          <button
            v-for="m in METRICS"
            :key="m.key"
            class="hist__tab"
            :class="{ 'hist__tab--on': activeMetric.key === m.key }"
            type="button"
            @click="metric = m.key"
          >
            {{ m.label }}
          </button>
        </div>
        <HistoryChart
          :times="tele.times"
          :values="chartValues"
          :unit="activeMetric.unit"
          :max="activeMetric.max"
        />
      </div>
    </div>
  </aside>
</template>

<style scoped>
.gpudock {
  width: 320px;
  flex-shrink: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--pk-bg-surface);
  /* no border-left: the ResizeHandle draws the divider (Traverse pattern). */
}
.gpudock__head {
  height: var(--pk-header-height);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  border-bottom: 1px solid var(--pk-border-default);
}
.gpudock__title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-primary);
}
.gpudock__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--pk-text-muted);
  transition: background 0.2s ease;
}
.gpudock__dot--live {
  background: var(--pk-status-success, #22c55e);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--pk-status-success, #22c55e) 22%, transparent);
}
.gpudock__body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.gpudock__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  text-align: center;
  padding: 48px 12px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.gpudock__note {
  padding: 8px 10px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  line-height: 1.4;
}

/* engine strip */
.estrip {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  transition: border-color 0.2s ease;
}
.estrip--busy {
  border-color: var(--pk-accent);
}
.estrip__top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.estrip__phase {
  display: inline-flex;
  align-items: center;
  /* No outdent. This used to carry `margin-left: -8px` to make the LABEL TEXT
     start at the same x as the stats row below, which is a real thing to want
     and the wrong thing to optimise: a pill with a filled background is read
     by its BOX, not by the text inside it. The outdent put the box at x=4 in a
     card whose content edge is x=12, so the tinted pill hung into the padding
     and pointed 8px further left than everything under it.
     Invisible while idle (that background is nearly transparent) and obvious
     the moment a model runs and the pill turns amber or blue - which is how it
     shipped, and how it was spotted. Measured before/after: pill
     left 1121 -> 1129, stats left 1129. */
  padding: 2px 8px;
  border-radius: var(--pk-radius-full);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  text-transform: capitalize;
  background: var(--pk-bg-inset);
  color: var(--pk-text-muted);
}
.estrip__phase--prefill {
  background: var(--pk-status-warning-subtle, var(--pk-bg-inset));
  color: var(--pk-status-warning);
}
.estrip__phase--decode {
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.estrip__tok {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.estrip__tok strong {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-base);
  color: var(--pk-text-primary);
}
.estrip__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
}

.gcard {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.gcard__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.gcard__name {
  flex: 1;
  min-width: 0;
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  color: var(--pk-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.gcard__badge {
  flex-shrink: 0;
  padding: 1px 7px;
  border-radius: var(--pk-radius-full);
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  background: var(--pk-accent-subtle);
  color: var(--pk-accent-text);
}
.gcard__paddockmem {
  margin-top: -6px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.metric__top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.metric__label {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.metric__val {
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
  font-weight: 600;
}
.metric__sub {
  color: var(--pk-text-muted);
  font-weight: 400;
}
.gcard__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  font-family: var(--pk-font-mono);
  font-size: 11px;
  color: var(--pk-text-muted);
}

/* history chart */
.hist {
  display: flex;
  flex-direction: column;
  gap: 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  padding: 10px;
}
.hist__tabs {
  display: flex;
  gap: 2px;
  padding: 2px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
}
.hist__tab {
  flex: 1;
  padding: 4px 0;
  border: 0;
  border-radius: var(--pk-radius-sm);
  background: transparent;
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-xs);
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s ease, color 0.12s ease;
}
.hist__tab:hover {
  color: var(--pk-text-primary);
}
.hist__tab--on {
  background: var(--pk-bg-elevated);
  color: var(--pk-text-primary);
  box-shadow: var(--pk-shadow-sm);
}
</style>
