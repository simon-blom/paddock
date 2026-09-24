// SettingsPanel.vue and the native settings page use this same presentation.
// These are conversation settings, not every key in useSettingsStore: autoTitle
// belongs to ConversationSidebar's sort menu; markUnsure to TranscriptView.
import { MAX_REPLY_LIMIT } from './reply-limit'

export const STUDIO_SETTINGS_SECTIONS = [
  { id: 'maxTokens', title: 'Reply limit' },
  { id: 'maxToolCalls', title: 'Tools per reply' },
  { id: 'summarize', title: 'Summarize older messages' },
  { id: 'microphone', title: 'Microphone' },
  { id: 'mapTiles', title: 'Map tiles' },
] as const

export const TOOL_CALL_STOPS = [
  { value: 0, label: 'Server default' },
  ...[5, 10, 25, 50, 100].map(value => ({ value, label: `${value} tool calls` })),
]

export function studioSettingsLayout() {
  return {
    sections: STUDIO_SETTINGS_SECTIONS,
    replyLimit: { maximum: MAX_REPLY_LIMIT },
    toolStops: TOOL_CALL_STOPS,
  }
}
