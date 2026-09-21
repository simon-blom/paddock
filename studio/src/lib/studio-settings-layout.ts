// SettingsPanel.vue and the native settings page use this same presentation.
// These are conversation settings, not every key in useSettingsStore: autoTitle
// belongs to ConversationSidebar's sort menu; markUnsure to TranscriptView.
export const STUDIO_SETTINGS_SECTIONS = [
  { id: 'maxTokens', title: 'Max reply length' },
  { id: 'maxToolCalls', title: 'Tools per reply' },
  { id: 'summarize', title: 'Summarize older messages' },
  { id: 'microphone', title: 'Microphone' },
  { id: 'mapTiles', title: 'Map tiles' },
] as const

export const TOOL_CALL_STOPS = [
  { value: 0, label: 'Server default' },
  ...[5, 10, 25, 50, 100].map(value => ({ value, label: `${value} tool calls` })),
]

export function replyLengthStops(maxCtx: number): (number | null)[] {
  const cap = Number.isSafeInteger(maxCtx) && maxCtx > 0 ? maxCtx : 8192
  const stops: (number | null)[] = []
  for (let value = 512; value < cap; value *= 2) stops.push(value)
  return [...stops, null]
}

export function replyStopLabel(value: number | null): string {
  return value == null ? 'Model maximum' : `${formatReplyTokens(value)} tokens`
}

export function formatReplyTokens(value: number): string {
  if (value < 1024) return String(value)
  const k = value / 1024
  return `${Number.isInteger(k) ? k : k.toFixed(1)}K`
}

export function studioSettingsLayout(maxCtx: number) {
  return {
    sections: STUDIO_SETTINGS_SECTIONS,
    replyStops: replyLengthStops(maxCtx).map(value => ({
      value, label: replyStopLabel(value), shortLabel: value == null ? 'Max' : formatReplyTokens(value),
    })),
    toolStops: TOOL_CALL_STOPS,
  }
}
