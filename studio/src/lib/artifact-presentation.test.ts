import { describe, expect, it } from 'vitest'
import { parseCsv } from './artifact-presentation'
import fixtures from '../../../apps/macos/Tests/PaddockUITests/Fixtures/artifact-csv.json'

describe('Web and native artifact CSV parity', () => {
  for (const row of fixtures) it(row.name, () => expect(parseCsv(row.source)).toEqual(row.rows))
})
