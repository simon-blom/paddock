import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Editor } from '@tiptap/core'
import { Schema } from '@tiptap/pm/model'
import { EditorState, Selection, TextSelection } from '@tiptap/pm/state'
import { DecorationSet } from '@tiptap/pm/view'
import { Dictation, appendDictated, setGhost } from './dictation'
import fixtures from '../../../apps/macos/Tests/PaddockUITests/Fixtures/dictation.json'

const schema = new Schema({ nodes: {
  doc: { content: 'paragraph+' }, paragraph: { content: 'text*' }, text: {},
} })
afterEach(() => vi.unstubAllGlobals())

describe('web/native inline dictation contract', () => {
  for (const row of fixtures) {
    it(row.name, () => {
      // Exercise the real extension/state/decoration. Only creating its inert
      // span is stubbed: no browser or replacement implementation of spacing.
      let span: { className: string; textContent: string } | undefined
      vi.stubGlobal('document', { createElement: () => (span = { className: '', textContent: '' }) })
      const plugins = Dictation.config.addProseMirrorPlugins!.call({} as never)
      const plugin = plugins[0]!
      let state = EditorState.create({ schema, plugins, doc: schema.node('doc', null,
        row.draft.split('\n').map(line => schema.node('paragraph', null, line ? schema.text(line) : undefined))) })
      state = state.apply(state.tr.setSelection(TextSelection.create(state.doc, 1)))
      const original = state.doc
      const selection = state.selection.toJSON()
      let inserted = ''
      const editor = {
        isDestroyed: false,
        get state() { return state },
        view: { dispatch: (tr: Parameters<typeof state.apply>[0]) => { state = state.apply(tr) } },
        commands: { insertContentAt: (at: number, text: string, options: unknown) => {
          expect(at).toBe(Selection.atEnd(state.doc).to)
          expect(options).toEqual({ updateSelection: false })
          inserted = text
        } },
      } as unknown as Editor
      setGhost(editor, row.provisional)
      const decorations = plugin.props.decorations!.call(plugin, state)
      expect(state.doc).toBe(original)
      expect(state.selection.toJSON()).toEqual(selection)
      expect(span?.textContent ?? '').toBe(row.ghost)
      if (row.ghost) {
        expect(span?.className).toBe('dictation-ghost')
        const widget = (decorations as DecorationSet).find()[0]!
        expect(widget.from).toBe(Selection.atEnd(state.doc).to)
        expect(widget.spec.side).toBe(1)
      } else expect(decorations).toBeNull()
      appendDictated(editor, row.provisional)
      expect(row.draft + inserted).toBe(row.committed)
      setGhost(editor, '')
      expect(plugin.props.decorations!.call(plugin, state)).toBeNull()
      expect(state.doc).toBe(original)
    })
  }
})
