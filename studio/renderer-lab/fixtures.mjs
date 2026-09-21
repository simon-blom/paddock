import { mkdir, writeFile } from 'node:fs/promises'
import { deflateSync } from 'node:zlib'

export function crc32(bytes) {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0)
  }
  return (crc ^ 0xffffffff) >>> 0
}
function pngChunk(type, bytes) {
  const name = Buffer.from(type)
  const length = Buffer.alloc(4); length.writeUInt32BE(bytes.length)
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([name, bytes])))
  return Buffer.concat([length, name, bytes, crc])
}
function png() {
  const header = Buffer.alloc(13)
  header.writeUInt32BE(96, 0); header.writeUInt32BE(48, 4); header[8] = 8; header[9] = 2
  const pixels = Buffer.alloc(48 * (1 + 96 * 3))
  for (let y = 0; y < 48; y++) for (let x = 0; x < 96; x++) {
    const offset = y * 289 + 1 + x * 3
    pixels[offset] = x < 48 ? 25 : 195
    pixels[offset + 1] = y < 24 ? 90 : 180
    pixels[offset + 2] = 140
  }
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), pngChunk('IHDR', header), pngChunk('IDAT', deflateSync(pixels)), pngChunk('IEND', Buffer.alloc(0))])
}
/** Minimal standards-based fixture writer, not a replacement document engine. */
export function pdfFixture(pages = 2, images = false) {
  if (!Number.isInteger(pages) || pages < 1 || pages > 1000) throw new RangeError('PDF page count must be 1-1000')
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    `<< /Type /Pages /Kids [${Array.from({ length: pages }, (_, i) => `${4 + i * 2} 0 R`).join(' ')}] /Count ${pages} >>`,
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
  ]
  const streamObject = (bytes, extra = '') => Buffer.concat([Buffer.from(`<< ${extra} /Length ${bytes.length} >>\nstream\n`), bytes, Buffer.from('\nendstream')])
  for (let page = 1; page <= pages; page++) {
    const stream = `0.12 0.12 0.12 rg BT /F1 24 Tf 54 720 Td (Rendering Lab - page ${page}) Tj 0 -40 Td /F1 14 Tf (Synthetic PDF. Search for Labrador.) Tj 0 -26 Td (Select this sentence and copy it.) Tj ET\n0.2 0.5 0.7 rg 54 510 220 95 re f\n0.1 0.1 0.1 RG 54 390 400 80 re S\nBT /F1 12 Tf 66 435 Td (Table: Engine | Status) Tj 0 -24 Td (Lector PDFium | Bundled WASM) Tj ET\n`
    const imageId = 4 + pages * 2 + page - 1
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> ${images ? `/XObject << /Im0 ${imageId} 0 R >>` : ''} >> /Contents ${5 + (page - 1) * 2} 0 R >>`)
    objects.push(streamObject(Buffer.from(stream + (images ? 'q 300 0 0 300 54 54 cm /Im0 Do Q\n' : ''))))
  }
  if (images) for (let page = 0; page < pages; page++) {
    // Unique deterministic image per page, not a shared-image cache shortcut.
    const pixels = Buffer.alloc(512 * 512 * 3)
    let seed = page + 1
    for (let i = 0; i < pixels.length; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; pixels[i] = seed >>> 24 }
    objects.push(streamObject(deflateSync(pixels), '/Type /XObject /Subtype /Image /Width 512 /Height 512 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode'))
  }
  const chunks = [Buffer.from('%PDF-1.7\n')]; const offsets = []; let offset = chunks[0].length
  objects.forEach((object, i) => {
    offsets.push(offset)
    const bytes = Buffer.concat([Buffer.from(`${i + 1} 0 obj\n`), Buffer.from(object), Buffer.from('\nendobj\n')])
    chunks.push(bytes); offset += bytes.length
  })
  chunks.push(Buffer.from(`xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.map(n => `${String(n).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${offset}\n%%EOF\n`))
  return Buffer.concat(chunks)
}
export function zipStore(files) {
  const local = []; const central = []; let offset = 0
  for (const [path, content] of Object.entries(files)) {
    const name = Buffer.from(path); const data = Buffer.from(content); const crc = crc32(data)
    const head = Buffer.alloc(30)
    head.writeUInt32LE(0x04034b50, 0); head.writeUInt16LE(20, 4)
    head.writeUInt32LE(crc, 14); head.writeUInt32LE(data.length, 18); head.writeUInt32LE(data.length, 22); head.writeUInt16LE(name.length, 26)
    const dir = Buffer.alloc(46)
    dir.writeUInt32LE(0x02014b50, 0); dir.writeUInt16LE(20, 4); dir.writeUInt16LE(20, 6)
    dir.writeUInt32LE(crc, 16); dir.writeUInt32LE(data.length, 20); dir.writeUInt32LE(data.length, 24); dir.writeUInt16LE(name.length, 28); dir.writeUInt32LE(offset, 42)
    local.push(head, name, data); central.push(dir, name); offset += head.length + name.length + data.length
  }
  const index = Buffer.concat(central); const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(central.length / 2, 8); end.writeUInt16LE(central.length / 2, 10)
  end.writeUInt32LE(index.length, 12); end.writeUInt32LE(offset, 16)
  return Buffer.concat([...local, index, end])
}
export function docxFixture(pages = 2) {
  if (!Number.isInteger(pages) || pages < 2 || pages > 200) throw new RangeError('DOCX page count must be 2-200')
  const w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  const rel = 'http://schemas.openxmlformats.org/package/2006/relationships'
  const office = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
  const paragraph = text => `<w:p><w:r><w:t>${text}</w:t></w:r></w:p>`
  return zipStore({
    '[Content_Types].xml': '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>',
    '_rels/.rels': `<Relationships xmlns="${rel}"><Relationship Id="rId1" Type="${office}/officeDocument" Target="word/document.xml"/></Relationships>`,
    'word/_rels/document.xml.rels': `<Relationships xmlns="${rel}"><Relationship Id="rId1" Type="${office}/styles" Target="styles.xml"/><Relationship Id="rId2" Type="${office}/image" Target="media/test.png"/></Relationships>`,
    'word/styles.xml': `<w:styles xmlns:w="${w}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults></w:styles>`,
    'word/document.xml': `<w:document xmlns:w="${w}" xmlns:r="${office}" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>
      ${paragraph('Rendering Lab - synthetic DOCX')}${paragraph('Selectable text, a table, an embedded image and tracked changes.')}
      <w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="8"/><w:bottom w:val="single" w:sz="8"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="3600"/><w:gridCol w:w="3600"/></w:tblGrid><w:tr><w:tc>${paragraph('Engine')}</w:tc><w:tc>${paragraph('Scriptor WASM')}</w:tc></w:tr><w:tr><w:tc>${paragraph('Fixture')}</w:tc><w:tc>${paragraph('Two pages')}</w:tc></w:tr></w:tbl>
      <w:p><w:r><w:drawing><wp:inline><wp:extent cx="1828800" cy="914400"/><wp:docPr id="1" name="Synthetic color grid"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="1" name="test.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="1828800" cy="914400"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
      <w:p><w:del w:id="1" w:author="Lab Reviewer" w:date="2026-01-01T00:00:00Z"><w:r><w:delText>Old wording</w:delText></w:r></w:del><w:ins w:id="2" w:author="Lab Reviewer" w:date="2026-01-01T00:00:00Z"><w:r><w:t>Revised wording</w:t></w:r></w:ins></w:p>
      <w:p><w:r><w:br w:type="page"/></w:r></w:p>${paragraph('Second page - layout sentinel')}${paragraph('Labrador appears here for selection testing.')}
      ${Array.from({ length: pages - 2 }, (_, i) => `<w:p><w:r><w:br w:type="page"/></w:r></w:p>${paragraph(`Scale document page ${i + 3}`)}${Array.from({ length: 12 }, (_, line) => paragraph(`Labrador paragraph ${line + 1}. Deterministic text for pagination and selection.`)).join('')}`).join('')}
      <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>
    </w:body></w:document>`,
    'word/media/test.png': png(),
  })
}
export async function generateFixtures(output) {
  await mkdir(`${output}/fixtures`, { recursive: true })
  await Promise.all([
    writeFile(`${output}/fixtures/sample.pdf`, pdfFixture()),
    writeFile(`${output}/fixtures/sample.docx`, docxFixture()),
    ...[100, 500, 1000].map(pages => writeFile(`${output}/fixtures/pdf-${pages}.pdf`, pdfFixture(pages))),
    writeFile(`${output}/fixtures/pdf-images-100.pdf`, pdfFixture(100, true)),
    ...[50, 200].map(pages => writeFile(`${output}/fixtures/docx-${pages}.docx`, docxFixture(pages))),
  ])
}
