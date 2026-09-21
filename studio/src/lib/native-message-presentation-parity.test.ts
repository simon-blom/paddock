import { describe, expect, it } from 'vitest'
import type { RunMeta, Usage } from '@/types/chat'
import { answerMetrics, answerMetricsHint, runDetailSections } from './message-presentation'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/message-presentation.json'

describe('Swift and Web Studio response details contract', () => {
  for (const row of fixtures) {
    it(row.name, () => {
      const usage = row.usage as Usage | undefined
      const run = row.run as RunMeta | undefined
      expect(answerMetrics(usage, row.realtime)).toBe(row.footer)
      expect(answerMetricsHint(usage)).toBe(row.hint)
      expect(runDetailSections({ run, usage })).toEqual(row.sections)
    })
  }
})
