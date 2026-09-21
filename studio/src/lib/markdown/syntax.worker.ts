import { SyntaxRenderer, type HighlightInput } from './syntax'
const renderer = new SyntaxRenderer()
self.onmessage = async (event: MessageEvent<{ id: number; input: HighlightInput }>) => {
  const { id, input } = event.data
  try { self.postMessage({ id, value: await renderer.render(input) }) }
  catch (error) { self.postMessage({ id, error: error instanceof Error ? error.message : String(error) }) }
}
