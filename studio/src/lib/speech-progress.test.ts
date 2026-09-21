import { describe, expect, it } from 'vitest'
import { SpeechProgress } from './speech-progress'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/speech-progress.json'
type Step = { drain?: boolean; event?: Record<string, unknown>; accepted?: boolean; pending: number; waiting: boolean; speaking?: string }

describe('same native/web speech-event sequences', () => {
  for (const fixture of fixtures) it(fixture.name, () => {
    const p = new SpeechProgress()
    for (const step of fixture.steps as Step[]) {
      if (step.drain) p.beginDrain()
      if (step.event) expect(p.receive(step.event)).toBe(step.accepted ?? true)
      expect(p.pending.size).toBe(step.pending)
      expect(p.waiting).toBe(step.waiting)
      expect(p.speaking).toBe(step.speaking)
    }
  })
})
