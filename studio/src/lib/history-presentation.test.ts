import { describe, expect, it } from 'vitest'
import web from '../components/chat/ConversationSidebar.vue?raw'
import native from '../../../apps/macos/Sources/PaddockUI/StudioHistoryView.swift?raw'
import nativeActions from '../../../apps/macos/Sources/PaddockUI/StudioHistoryActions.swift?raw'

describe('history title and action placement', () => {
  it('keeps the same ordered actions on web and native', () => {
    const labels = ['Rename', 'Pin', 'Generate title', 'Show full title', 'Delete']
    const webMenu = web.slice(web.indexOf('<MenuContent :label="`Actions for'))
    const nativeMenu = native.slice(native.indexOf('private var actions:'))
    for (const source of [webMenu, nativeMenu]) {
      const positions = labels.map(label => source.indexOf(label))
      expect(positions.every(position => position >= 0)).toBe(true)
      expect(positions).toEqual([...positions].sort((a, b) => a - b))
    }
  })
  it('exposes stored full titles without generating summaries or hydrating chats', () => {
    expect(web).toContain('showFullTitle(c.title)')
    expect(web).toContain('<Tooltip :label="c.title">')
    expect(native).toContain('StudioFullChatTitle(title: row.title)')
    expect(native).toContain('.help(row.title).accessibilityLabel(row.title)')
    expect(nativeActions).toContain('Text(verbatim: title).textSelection(.enabled)')
    expect(nativeActions).not.toContain('StudioWorkspace')
  })
})
