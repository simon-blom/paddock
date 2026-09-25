import fixture from '../../../apps/macos/Sources/PaddockConversationCore/Resources/reads-example.json'
import { fromWire, validate, type ReadQuestion, type Samples } from './reads'

/** One bundled request shared with native Reads. Keep descriptive, derived IDs:
 * they are prompt text, and the model reads them better than arbitrary labels. */
export function readExample(): { state: string; questions: ReadQuestion[]; samples: Samples } {
  const parsed = fromWire(fixture)
  if (
    parsed.errors.length ||
    !parsed.state ||
    parsed.samples === undefined ||
    !validate(parsed.questions).ok
  ) {
    throw new Error('The Reads example could not be loaded.')
  }
  // A fresh editable draft, not an imported set with manually pinned IDs.
  for (const question of parsed.questions) question.idTouched = false
  return { state: parsed.state, questions: parsed.questions, samples: parsed.samples }
}
