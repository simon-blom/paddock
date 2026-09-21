import { describe, expect, it } from 'vitest'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/tree-parity.json'
import { activeMessages, migrate, stepSibling, deleteSubtree } from './tree'
import type { Conversation } from '@/types/chat'

// The same golden cases run in the independent Swift implementation. Sharing
// tests/contracts does not put JavaScript in the native runtime.
describe('native/web persisted-tree contract', () => {
  for (const fixture of fixtures) it(fixture.name, () => {
    const c = structuredClone(fixture.document) as unknown as Conversation
    for (const op of fixture.operations) {
      if (op.kind === 'migrate') {
        const changed = migrate(c)
        if ('changed' in op) expect(changed).toBe(op.changed)
      } else if (op.kind === 'sibling' && 'id' in op && 'delta' in op) {
        expect(stepSibling(c, op.id!, op.delta!)).toBe(true)
      } else if (op.kind === 'delete' && 'id' in op && 'removed' in op) {
        expect(deleteSubtree(c, op.id!).sort()).toEqual(op.removed!.slice().sort())
      }
      expect(activeMessages(c).map(m => m.id)).toEqual(op.path)
      expect(c.leafId ?? null).toBe(op.leaf)
    }
  })
})
