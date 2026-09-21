import { pagesParam } from '@/lib/attachments'
import { isPdfPart } from '@/lib/docrun'
import type { ContentPart } from '@/types/chat'
import type { AttachmentChoice } from './protocol'

/** Validate against the staged file, not just the bridge's numeric schema.
 * Keep the original reference and use the web composer's wire formatter.
 * No extraction, page rendering or mutation of the preview copy happens here. */
export function selectedAttachment(value: ContentPart | undefined, choice: AttachmentChoice): ContentPart {
  if (!value || !('attachmentId' in value) || value.attachmentId !== choice.id) throw new Error('An attachment is not ready; the draft has been kept')
  const pdf = value.type === 'file' && isPdfPart(value)
  const tiff = value.type === 'image' && (value.mime === 'image/tiff' || /\.tiff?$/i.test(value.name ?? ''))
  if ((choice.from !== undefined || choice.to !== undefined) && !pdf && !tiff) throw new Error('Page selection is only available for PDF and TIFF files')
  if (choice.text && !pdf) throw new Error('Text-only reading is only available for PDF files')
  const count = value.type === 'file' ? value.pages : undefined
  if (count && ((choice.from ?? 1) > count || (choice.to ?? 1) > count)) throw new Error(`This file has only ${count} pages`)
  if (value.type === 'file') return { ...value, pdfMode: choice.text ? 'text' : undefined, pageRange: pagesParam(choice) }
  if (value.type === 'image') return { ...value, detail: choice.detail ?? 'auto', pageRange: pagesParam(choice) }
  return { ...value }
}
