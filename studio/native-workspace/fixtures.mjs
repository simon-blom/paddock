import { mkdir, writeFile } from 'node:fs/promises'
import { pdfFixture, docxFixture } from '../renderer-lab/fixtures.mjs'

// Reuse the existing standards-based fixtures, without generating the large
// benchmark suite just to run the functional gate.
const output = process.argv[2]
if (!output) throw new Error('Supply a generated fixture directory')
await mkdir(output, { recursive: true })
await Promise.all([
  writeFile(`${output}/sample.pdf`, pdfFixture()),
  writeFile(`${output}/page-selection.pdf`, pdfFixture(6)),
  writeFile(`${output}/sample.docx`, docxFixture()),
])
