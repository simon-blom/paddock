import { describe, expect, it } from 'vitest'
import { changeToolPicker, pickerGroupState, pickerToolChecked, toolTermHits, type PickerState } from './tool-picker'

const server = { label: 'artifacts', tools: ['read', 'write'] }
const connector = { label: 'github', connectorId: 'personal', tools: ['create_issue', 'get_issue'] }
const personal = [connector]
const all: PickerState = { selection: { mode: 'all' }, connectorIds: ['personal'] }

describe('web/native tool picker interaction contract', () => {
  it('represents server coverage only in the All row, while connectors stay opt-in', () => {
    expect(pickerGroupState(all, server)).toBe('none')
    expect(pickerToolChecked(all, server, 'read')).toBe(false)
    expect(pickerGroupState(all, connector)).toBe('all')
    expect(pickerToolChecked(all, connector, 'create_issue')).toBe(true)
    const off = changeToolPicker(all, connector, personal, { kind: 'group', label: connector.label })
    expect(off).toEqual({ selection: { mode: 'all' }, connectorIds: [] })
    expect(all.connectorIds).toEqual(['personal'])
  })
  it('switches from All to exactly the picked group plus armed personal connectors', () => {
    const next = changeToolPicker(all, server, personal, { kind: 'group', label: 'artifacts' })
    expect(next.selection).toEqual({ mode: 'custom', picks: [{ label: 'github' }, { label: 'artifacts' }] })
    const onlyRead = changeToolPicker(all, server, personal, { kind: 'tool', label: 'artifacts', tool: 'read' })
    expect(onlyRead.selection).toEqual({ mode: 'custom', picks: [{ label: 'github' }, { label: 'artifacts', tool: 'read' }] })
    expect(pickerGroupState(onlyRead, server)).toBe('some')
  })
  it('expands a whole pick when unchecking a tool and collapses complete picks back to the group', () => {
    const one = changeToolPicker(all, connector, personal, { kind: 'tool', label: 'github', tool: 'create_issue' })
    expect(one.selection).toEqual({ mode: 'custom', picks: [{ label: 'github', tool: 'get_issue' }] })
    const both = changeToolPicker(one, connector, personal, { kind: 'tool', label: 'github', tool: 'create_issue' })
    expect(both.selection).toEqual({ mode: 'custom', picks: [{ label: 'github' }] })
    expect(pickerToolChecked(both, connector, 'future_tool')).toBe(true)
  })
  it('does not erase opt-in connector identities when moving between All and custom', () => {
    const empty = changeToolPicker(all, undefined, personal, { kind: 'clear' })
    expect(empty.connectorIds).toEqual(['personal'])
    expect(pickerGroupState(empty, connector)).toBe('none')
    expect(changeToolPicker(empty, undefined, personal, { kind: 'all' })).toEqual(all)
  })
  it('refuses a vanished group or tool rather than broadening the selection', () => {
    expect(() => changeToolPicker(all, undefined, personal, { kind: 'group', label: 'missing' })).toThrow()
    expect(() => changeToolPicker(all, server, personal, { kind: 'tool', label: 'artifacts', tool: 'missing' })).toThrow()
    expect(() => changeToolPicker(all, { label: 'offline' }, personal, { kind: 'tool', label: 'offline', tool: 'read' })).toThrow()
  })
  it('uses separator-insensitive and fuzzy names alongside per-word description matches', () => {
    expect(toolTermHits('crisu', 'create_issue')).toBe(true)
    expect(toolTermHits('createissue', 'create_issue')).toBe(true)
    expect(['create', 'repository'].every(t => toolTermHits(t, 'create_issue', 'Add an issue to a repository'))).toBe(true)
    expect(['create', 'calendar'].every(t => toolTermHits(t, 'create_issue', 'Add an issue to a repository'))).toBe(false)
  })
})
