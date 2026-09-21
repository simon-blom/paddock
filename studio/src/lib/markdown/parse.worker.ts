import { MarkdownParser, type ParseInput } from './parse'
const parser = new MarkdownParser()
self.onmessage = (event: MessageEvent<{ id: number; owner: string; input: ParseInput; acknowledge?: boolean }>) => {
  const { id, owner, input } = event.data
  if (event.data.acknowledge) self.postMessage({ id, started: true })
  try { self.postMessage({ id, value: parser.parse(owner, input) }) }
  catch (error) { self.postMessage({ id, error: error instanceof Error ? error.message : String(error) }) }
}
