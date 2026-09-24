<script setup lang="ts">
// Question x read heatmap for a resampled read (Apache ECharts): one cell per
// read, its text the label that read picked, its colour the confidence bin
// of that pick. This is the self-consistency picture people already draw
// for repeated decisions, so disagreement between reads is seen as a row
// that changes colour and text, not inferred from an agreement number.
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useTheme } from '@/composables/useTheme'
import { cssVar } from '@/lib/chart-theme'
import { CONFIDENCE_EDGES, confidenceBin, fmtP, type ReadDiagRead } from '@/lib/reads'

use([CanvasRenderer, HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent])

const props = defineProps<{ rows: { id: string; reads: ReadDiagRead[] }[] }>()
const { theme } = useTheme()

const colors = computed(() => {
  void theme.value
  return {
    bins: [1, 2, 3, 4].map((i) => cssVar(`--pk-conf-${i}`) || '#888'),
    surface: cssVar('--pk-bg-surface') || '#111',
    primary: cssVar('--pk-text-primary') || '#fff',
    muted: cssVar('--pk-text-muted') || '#888',
    elevated: cssVar('--pk-bg-elevated') || '#1b1b1b',
    strong: cssVar('--pk-border-strong') || '#333',
    mono: cssVar('--pk-font-mono') || 'monospace',
    // text on the two dark bins is light, on the two bright ones dark
    onBin: ['#F5F5F7', '#F5F5F7', '#0A1118', '#0A1118'],
  }
})

const nReads = computed(() => Math.max(0, ...props.rows.map((r) => r.reads.length)))
const height = computed(() => `${props.rows.length * 28 + 34}px`)

const option = computed(() => {
  const { bins, surface, primary, muted, elevated, strong, mono, onBin } = colors.value
  const data = props.rows.flatMap((r, y) =>
    r.reads.map((rd, x) => ({
      value: [x, y, rd.confidence],
      pick: rd.pick,
      label: { color: onBin[confidenceBin(rd.confidence)] },
    })),
  )
  return {
    animation: false,
    grid: { left: 0, right: 4, top: 22, bottom: 4, containLabel: true },
    tooltip: {
      backgroundColor: elevated,
      borderColor: strong,
      textStyle: { color: primary, fontSize: 12 },
      formatter: (p: { data: { value: number[]; pick: string } }) => {
        const [x, y, c] = p.data.value
        return `${props.rows[y]?.id ?? ''} · read ${x + 1}: ${p.data.pick} (${fmtP(c)})`
      },
    },
    xAxis: {
      type: 'category',
      position: 'top',
      data: Array.from({ length: nReads.value }, (_, i) => `read ${i + 1}`),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitArea: { show: false },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: props.rows.map((r) => r.id),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: primary, fontSize: 12, fontFamily: mono, width: 140, overflow: 'truncate' },
      splitArea: { show: false },
    },
    visualMap: {
      show: false,
      type: 'piecewise',
      dimension: 2,
      pieces: [
        { max: CONFIDENCE_EDGES[0], color: bins[0] },
        { min: CONFIDENCE_EDGES[0], max: CONFIDENCE_EDGES[1], color: bins[1] },
        { min: CONFIDENCE_EDGES[1], max: CONFIDENCE_EDGES[2], color: bins[2] },
        { min: CONFIDENCE_EDGES[2], color: bins[3] },
      ],
    },
    series: [
      {
        type: 'heatmap',
        data,
        label: {
          show: true,
          fontSize: 11,
          formatter: (p: { data: { pick: string } }) => p.data.pick,
        },
        itemStyle: { borderColor: surface, borderWidth: 2, borderRadius: 3 },
        emphasis: { disabled: true },
      },
    ],
  }
})
</script>

<template>
  <VChart class="hm" :option="option" :style="{ height }" autoresize />
</template>

<style scoped>
.hm {
  width: 100%;
}
</style>
