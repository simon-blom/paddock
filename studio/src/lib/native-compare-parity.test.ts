import { describe, expect, it } from 'vitest'
import type { Message } from '@/types/chat'
import { cloudModelIdentity } from '@/stores/models'
import { fastestCompareLane } from './compare-presentation'
import identities from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/compare-identity.json'
import races from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/compare-races.json'

describe('Swift and Web Compare presentation', () => {
  for (const row of identities) it(`maker/name: ${row.id} / ${row.kind ?? ''}`, () => {
    const actual = cloudModelIdentity(row.id, row.display, row.kind, row.provider)
    expect(actual.name).toBe(row.name)
    expect(actual.vendor ?? null).toBe(row.vendor)
  })
  for (const row of races) it(row.name, () => {
    expect(fastestCompareLane(row.messages as Message[]) ?? null).toBe(row.expected)
  })
})
