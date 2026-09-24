import { describe, expect, it } from 'vitest'
import { parseReplyLimit, resolveReplyLimit } from './reply-limit'
import { replyReserve } from './tokens'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/reply-limit.json'

describe('exact reply-limit editor', () => {
  for (const value of ['', '0', '-1', '+100', '1.5', '1e4', '5,000', '1048577', '9'.repeat(100)]) {
    it(`rejects ${JSON.stringify(value)}`, () => expect(parseReplyLimit(value)).toBeNull())
  }
  for (const value of ['1', '5000', '1048576', ' 512 \n']) {
    it(`preserves ${JSON.stringify(value)}`, () => expect(parseReplyLimit(value)).toBe(Number(value)))
  }
})
describe('native/web per-model effective limits', () => {
  it('never reserves a whole small context or a huge custom ceiling for a reply', () => {
    expect(replyReserve(null, 4096)).toBe(1024)
    expect(replyReserve(32768, 4096)).toBe(1024)
    expect(replyReserve(512, 4096)).toBe(512)
    expect(replyReserve(null, 131072, 2048)).toBe(2048)
    expect(replyReserve(32768, 131072)).toBe(4096)
  })
  for (const row of fixtures) it(row.name, () => {
    expect(resolveReplyLimit(
      row.requested ?? null, row.cloud, row.context, row.prompt ?? 0, row.ceiling, row.exact ?? 0,
    )).toBe(row.expected)
  })
})
