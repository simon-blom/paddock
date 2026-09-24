<script setup lang="ts">
// The probability bars of one answer (Apache ECharts, Apache-2.0): every bar
// carries its number, the reported label sits in the accent, the rest are
// muted, and the mass the model put OUTSIDE the label set is its own hatched
// last bar - never folded into a label (the separate `refusal` field is the
// API precedent). Colours come off the design tokens per theme.
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useTheme } from '@/composables/useTheme'
import { cssVar, withAlpha } from '@/lib/chart-theme'
import { fmtP, type Bar } from '@/lib/reads'

use([CanvasRenderer, BarChart, GridComponent])

const props = defineProps<{ bars: Bar[] }>()
const { theme } = useTheme()

const colors = computed(() => {
  void theme.value
  return {
    accent: cssVar('--pk-accent') || '#0ea5e9',
    muted: cssVar('--pk-text-muted') || '#888',
    primary: cssVar('--pk-text-primary') || '#fff',
    grid: cssVar('--pk-border-subtle') || 'rgba(128,128,128,0.16)',
    mono: cssVar('--pk-font-mono') || 'monospace',
  }
})

const ROW = 24
const height = computed(() => `${props.bars.length * ROW + 6}px`)

const option = computed(() => {
  const { accent, muted, primary, grid, mono } = colors.value
  return {
    animation: false,
    grid: { left: 0, right: 44, top: 3, bottom: 3, containLabel: true },
    xAxis: { type: 'value', min: 0, max: 1, show: false },
    yAxis: {
      type: 'category',
      inverse: true,
      data: props.bars.map((b) => b.name),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: primary,
        fontSize: 12,
        width: 150,
        overflow: 'truncate',
        margin: 10,
        rich: { o: { color: muted, fontStyle: 'italic', fontSize: 12 } },
        formatter: (name: string) =>
          props.bars.find((b) => b.name === name)?.role === 'outside' ? `{o|${name}}` : name,
      },
    },
    series: [
      {
        type: 'bar',
        barWidth: 12,
        silent: true,
        showBackground: true,
        backgroundStyle: { color: grid, borderRadius: 3 },
        label: {
          show: true,
          position: 'right',
          color: primary,
          fontSize: 12,
          fontFamily: mono,
          formatter: (p: { value: unknown }) => fmtP(typeof p.value === 'number' ? p.value : undefined),
        },
        data: props.bars.map((b) => ({
          value: b.p,
          itemStyle:
            b.role === 'winner'
              ? { color: accent, borderRadius: 3 }
              : b.role === 'other'
                ? { color: withAlpha(muted, 0.45), borderRadius: 3 }
                : {
                    color: 'transparent',
                    borderRadius: 3,
                    decal: {
                      symbol: 'rect',
                      symbolSize: 1,
                      color: withAlpha(muted, 0.7),
                      backgroundColor: 'transparent',
                      dashArrayX: [1, 0],
                      dashArrayY: [2, 3],
                      rotation: -Math.PI / 4,
                    },
                  },
        })),
      },
    ],
  }
})
</script>

<template>
  <VChart class="pb" :option="option" :style="{ height }" autoresize />
</template>

<style scoped>
.pb {
  width: 100%;
}
</style>
