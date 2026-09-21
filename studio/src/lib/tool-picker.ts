import type { ToolSelection, ToolPick } from '@/types/chat'

/** Presentation/interaction rules shared by the web composer and Swift's
 * presentation bridge. All mode is represented by its own row, not by ticking
 * every server. Personal connectors remain explicitly armed per conversation. */
export interface PickerState { selection: ToolSelection; connectorIds: string[] }
export interface PickerGroup { label: string; connectorId?: string; tools?: string[] }
export type PickerAction = { kind: 'all' } | { kind: 'clear' } | { kind: 'group'; label: string } | { kind: 'tool'; label: string; tool: string }

export function pickerGroupState(state: PickerState, group: PickerGroup): 'all' | 'some' | 'none' {
  const s = state.selection
  if (s.mode === 'all') return group.connectorId && state.connectorIds.includes(group.connectorId) ? 'all' : 'none'
  if (s.picks.some(p => p.label === group.label && p.tool == null)) return 'all'
  return s.picks.some(p => p.label === group.label) ? 'some' : 'none'
}

export function pickerToolChecked(state: PickerState, group: PickerGroup, name: string): boolean {
  const s = state.selection
  if (s.mode === 'all') return !!group.connectorId && state.connectorIds.includes(group.connectorId)
  return s.picks.some(p => p.label === group.label && (p.tool == null || p.tool === name))
}

export function changeToolPicker(state: PickerState, group: PickerGroup | undefined, personal: PickerGroup[], action: PickerAction): PickerState {
  if (action.kind === 'all') return { ...state, selection: { mode: 'all' } }
  if (action.kind === 'clear') return { ...state, selection: { mode: 'custom', picks: [] } }
  if (!group || group.label !== action.label) throw new Error('This tool group is no longer available')
  if (action.kind === 'group' && state.selection.mode === 'all' && group.connectorId) {
    const ids = new Set(state.connectorIds)
    if (ids.has(group.connectorId)) ids.delete(group.connectorId); else ids.add(group.connectorId)
    return { ...state, connectorIds: [...ids] }
  }
  let picks: ToolPick[] = state.selection.mode === 'custom' ? state.selection.picks.map(p => ({ ...p }))
    : personal.filter(g => g.connectorId && state.connectorIds.includes(g.connectorId)).map(g => ({ label: g.label }))
  if (action.kind === 'group') {
    const was = pickerGroupState(state, group)
    picks = picks.filter(p => p.label !== group.label)
    if (was !== 'all') picks.push({ label: group.label })
  } else {
    if (!group.tools?.includes(action.tool)) throw new Error("Reload this server's tool listing before choosing a tool")
    if (picks.some(p => p.label === group.label && p.tool == null)) {
      picks = picks.filter(p => p.label !== group.label)
      picks.push(...group.tools.filter(t => t !== action.tool).map(tool => ({ label: group.label, tool })))
    } else if (picks.some(p => p.label === group.label && p.tool === action.tool)) {
      picks = picks.filter(p => !(p.label === group.label && p.tool === action.tool))
    } else {
      picks.push({ label: group.label, tool: action.tool })
      if (group.tools.every(t => picks.some(p => p.label === group.label && p.tool === t))) {
        picks = picks.filter(p => p.label !== group.label)
        picks.push({ label: group.label })
      }
    }
  }
  return { ...state, selection: { mode: 'custom', picks } }
}

/** Same multi-word, separator-insensitive, in-order name match in both UIs. */
export function toolTermHits(term: string, name: string, description?: string): boolean {
  const n = name.toLowerCase().replace(/[-_.\s]/g, ''), t = term.toLowerCase().replace(/[-_.\s]/g, '')
  if (!t || n.includes(t) || description?.toLowerCase().includes(term.toLowerCase())) return true
  let i = 0
  for (const ch of n) { if (ch === t[i]) i++; if (i === t.length) return true }
  return false
}
