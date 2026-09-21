import { test } from 'node:test'
import assert from 'node:assert/strict'
import { runInNewContext } from 'node:vm'
import { crc32, pdfFixture, docxFixture } from './fixtures.mjs'
import { DISPOSAL_SHIM } from './compat.mjs'

test('PDF xref entries resolve to the declared objects', () => {
  const bytes = pdfFixture(); const text = bytes.toString('ascii')
  const start = Number(text.match(/startxref\n(\d+)/)[1])
  assert.equal(text.slice(start, start + 4), 'xref')
  const lines = text.slice(start).split('\n')
  assert.equal(lines[1], '0 8')
  for (let index = 1; index < 8; index++) {
    const offset = Number(lines[index + 2].slice(0, 10))
    assert.equal(text.slice(offset, offset + 7), `${index} 0 obj`)
  }
  assert.equal(text.match(/Labrador/g).length, 2)
})
test('DOCX stored ZIP members have valid CRC and central-directory offsets', () => {
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926)
  const bytes = docxFixture(); const end = bytes.length - 22
  assert.equal(bytes.readUInt32LE(end), 0x06054b50)
  const entries = bytes.readUInt16LE(end + 10)
  let cursor = bytes.readUInt32LE(end + 16); const names = []
  for (let index = 0; index < entries; index++) {
    assert.equal(bytes.readUInt32LE(cursor), 0x02014b50)
    const offset = bytes.readUInt32LE(cursor + 42)
    const size = bytes.readUInt32LE(cursor + 24)
    const nameSize = bytes.readUInt16LE(cursor + 28)
    const name = bytes.subarray(cursor + 46, cursor + 46 + nameSize).toString()
    names.push(name)
    assert.equal(bytes.readUInt32LE(offset), 0x04034b50)
    assert.equal(crc32(bytes.subarray(offset + 30 + nameSize, offset + 30 + nameSize + size)), bytes.readUInt32LE(cursor + 16))
    cursor += 46 + nameSize
  }
  assert.equal(cursor, end)
  assert.ok(names.includes('word/document.xml'))
  assert.ok(names.includes('word/media/test.png'))
})
test('large PDFs keep page trees, binary stream lengths and xrefs consistent', () => {
  for (const [pages, images] of [[1000, false], [3, true]]) {
    const bytes = pdfFixture(pages, images); const text = bytes.toString('latin1')
    assert.equal((text.match(/\/Type \/Page /g) ?? []).length, pages)
    assert.ok(text.includes(`/Count ${pages}`))
    assert.equal((text.match(/\/Subtype \/Image/g) ?? []).length, images ? pages : 0)
    const xref = Number(text.match(/startxref\n(\d+)/)[1]); const lines = text.slice(xref).split('\n')
    const count = Number(lines[1].split(' ')[1])
    for (let i = 1; i < count; i++) assert.ok(text.slice(Number(lines[i + 2].slice(0, 10))).startsWith(`${i} 0 obj\n`))
    for (const match of text.matchAll(/\/Length (\d+) >>\nstream\n/g)) {
      const end = match.index + match[0].length + Number(match[1]); assert.equal(text.slice(end, end + 10), '\nendstream')
    }
  }
  assert.throws(() => pdfFixture(1001), RangeError)
  assert.throws(() => pdfFixture(1.5), RangeError)
})
test('large DOCX uses explicit, deterministic page breaks and bounded sizes', () => {
  const text = docxFixture(200).toString('utf8')
  assert.equal((text.match(/w:br w:type="page"/g) ?? []).length, 199)
  assert.ok(text.includes('Scale document page 200'))
  assert.throws(() => docxFixture(201), RangeError)
})
test('disposal shim survives a bundle-local hoisted Symbol and preserves native symbols', () => {
  const shimmedSymbol = { for: Symbol.for }
  const context = { Symbol: shimmedSymbol }
  runInNewContext(`(function(){ ${DISPOSAL_SHIM}; var Symbol; })()`, context)
  assert.equal(shimmedSymbol.dispose, Symbol.for('Symbol.dispose'))
  assert.equal(shimmedSymbol.asyncDispose, Symbol.for('Symbol.asyncDispose'))
  const native = Symbol('native disposal')
  const nativeContext = { Symbol: { for: Symbol.for, dispose: native, asyncDispose: native } }
  runInNewContext(DISPOSAL_SHIM, nativeContext)
  assert.equal(nativeContext.Symbol.dispose, native)
  assert.equal(nativeContext.__labDisposeShim, undefined)
})
