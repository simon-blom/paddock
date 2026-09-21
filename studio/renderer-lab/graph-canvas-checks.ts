import Graph from 'graphology'
import Sigma from 'sigma'
import { drawDiscNodeLabel } from 'sigma/rendering'
import { budgetGraphCanvases, disposeGraphRenderer } from '@/lib/graph/canvas-budget'
import { createHoverRefresh } from '@/lib/graph/hover-refresh'
import { GRAPH_COLORS_LIGHT, fadeColor } from '@/lib/graph/session'
import NodePulseProgram from '@/lib/graph/node-pulse'
import { assert, check, delay } from './report'

/** Functional capture only, never a scale/performance run. Exercise the real
 * Sigma captors and painted canvases rather than mocking its event boundary. */
export async function graphCanvasChecks() {
  await check('Incremental hover matches full refresh and pulse redraw avoids processing', async () => {
    const hosts = [0, 1].map(() => {
      const host = document.createElement('div')
      host.style.cssText = 'position:fixed;left:80px;top:160px;width:420px;height:260px;z-index:100;background:white'
      document.body.append(host); return host
    })
    const graph = new Graph()
    for (let i = 0; i < 8; i++) graph.addNode(String(i), { x: i % 4, y: Math.floor(i / 4), size: 10,
      label: `Node ${i}`, color: '#0369a1', originalColor: '#0369a1' })
    for (let i = 0; i < 7; i++) graph.addEdge(String(i), String(i + 1), { size: 2, color: '#444444', originalColor: '#444444' })
    let node: string | null = null, nodes = new Set<string>(), edges = new Set<string>()
    const settings = { renderEdgeLabels: false, labelRenderedSizeThreshold: 0,
      nodeProgramClasses: { pulse: NodePulseProgram },
      nodeReducer: (key: string, data: Record<string, any>) => {
        const result = { ...data }
        if (node && !nodes.has(key)) { result.color = fadeColor(String(data.originalColor), false); result.label = '' }
        else if (node) { result.forceLabel = true; if (key === node) result.labelColor = GRAPH_COLORS_LIGHT.label }
        return result
      },
      edgeReducer: (key: string, data: Record<string, any>) => ({ ...data, color: node && !edges.has(key) ? fadeColor(String(data.originalColor), false) : data.color }),
    }
    let layoutRunning = false
    const reference = new Sigma(graph, hosts[0], settings), candidate = new Sigma(graph, hosts[1], {
      ...settings, defaultDrawNodeLabel: (context, data, options) => { if (!layoutRunning) drawDiscNodeLabel(context, data, options) },
    })
    const hover = createHoverRefresh(candidate, graph)
    const pixels = (s: Sigma) => ['nodes', 'edges', 'labels'].map(key => s.getCanvases()[key].toDataURL())
    let processes = 0; candidate.on('beforeProcess', () => { processes++ })
    const move = (key: string | null) => {
      node = key; nodes = new Set(key ? [key, ...graph.neighbors(key)] : []); edges = new Set(key ? graph.edges(key) : [])
      hover.update({ node, nodes, edges })
    }
    const equal = async () => {
      reference.refresh(); const expected = pixels(reference)
      // A previously scheduled graph-mutation render can precede the hover
      // frame. Compare the settled hover, not that intermediate older frame.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
      const actual = await new Promise<string[]>(resolve => {
        candidate.once('afterRender', () => resolve(pixels(candidate))); candidate.scheduleRender()
      })
      const different = ['nodes', 'edges', 'labels'].filter((_, i) => actual[i] !== expected[i])
      assert(!different.length, `Partial hover pixels differ at ${node ?? 'none'}: ${different.join(',')}`)
    }
    try {
      move('0'); await equal()
      for (const key of ['1', '5', '7', '0']) { move(null); move(key); await equal() }
      assert(processes === 0, 'Color/forced-label transitions unnecessarily rebuilt the graph index')
      // A full position/type pass while hovering rebuilds the label grid.
      // Empty labels must keep membership so exit can restore labels by repaint.
      graph.setNodeAttribute('2', 'x', 2.1)
      graph.setNodeAttribute('3', 'type', 'pulse')
      move('3'); await equal()
      move(null); await equal()
      for (const enabled of [false, true]) {
        hover.invalidate()
        reference.setSetting('renderLabels', enabled); candidate.setSetting('renderLabels', enabled)
        move(enabled ? '5' : '1'); await equal()
      }
      move(null); await equal()
      const beforeLabels = processes
      for (const running of [true, false]) {
        layoutRunning = running
        reference.setSetting('renderLabels', !running)
        candidate.scheduleRender(); await equal()
      }
      assert(processes === beforeLabels, 'Layout label visibility reprocessed graph data')
      const before = processes
      const oldPixels = await new Promise<string[]>(resolve => {
        candidate.once('afterRender', () => resolve(pixels(candidate))); candidate.scheduleRender()
      })
      NodePulseProgram.currentTime = .5
      const newPixels = await new Promise<string[]>(resolve => {
        candidate.once('afterRender', () => resolve(pixels(candidate))); candidate.scheduleRender()
      })
      assert(processes === before, 'Shader clock redraw reprocessed graph data')
      assert(oldPixels[0] !== newPixels[0], 'Pulse uniform did not change painted node pixels')
      return 'Hover, geometry/type changes and labels are pixel-identical; layout label visibility and pulse pixels update without rebuilding indexes'
    } finally {
      NodePulseProgram.currentTime = 0; hover.dispose()
      disposeGraphRenderer(reference); disposeGraphRenderer(candidate); hosts.forEach(host => host.remove())
    }
  })
  await check('Graph backing budget preserves pixels and pointer coordinates', async () => {
    const host = document.createElement('div')
    host.style.cssText = 'position:fixed;left:80px;top:160px;width:420px;height:260px;z-index:100;background:white'
    document.body.append(host)
    const graph = new Graph()
    graph.addNode('a', { x: 0, y: 0, size: 12, label: 'Alpha', color: '#0369a1' })
    graph.addNode('b', { x: 1, y: 1, size: 12, label: 'Beta', color: '#be123c' })
    graph.addEdge('a', 'b', { size: 3, label: 'Edge label', color: '#444444', forceLabel: true })
    const renderer = new Sigma(graph, host, { renderEdgeLabels: false, labelRenderedSizeThreshold: 0 })
    try {
      const layers = renderer.getCanvases()
      const rect = layers.mouse.getBoundingClientRect()
      // Immediate capture after synchronous refresh, before WebGL's ordinary
      // drawing-buffer discard. No screenshots in the performance profiles.
      const pixels = () => {
        renderer.refresh()
        return ['nodes', 'edges', 'labels', 'hovers', 'hoverNodes'].map(key => layers[key].toDataURL())
      }
      const original = pixels()
      budgetGraphCanvases(renderer)
      assert(JSON.stringify(pixels()) === JSON.stringify(original), 'Visible graph pixels changed with compact blank canvases')
      assert(layers.mouse.width === 1 && layers.mouse.height === 1 && layers.edgeLabels.width === 1, 'Blank bitmaps are not compact')
      assert(layers.mouse.getBoundingClientRect().width === rect.width, 'Pointer hit box changed')
      let clicks = 0; let moves = 0
      renderer.on('clickNode', ({ node }) => { if (node === 'a') clicks++ })
      renderer.on('moveBody', () => { moves++ })
      const click = () => {
        const point = renderer.graphToViewport(graph.getNodeAttributes('a') as { x: number; y: number })
        const box = layers.mouse.getBoundingClientRect()
        const clientX = box.left + point.x; const clientY = box.top + point.y
        assert(document.elementFromPoint(clientX, clientY) === layers.mouse, 'Compact pointer surface lost CSS hit testing')
        layers.mouse.dispatchEvent(new MouseEvent('mousemove', { clientX, clientY, bubbles: true }))
        layers.mouse.dispatchEvent(new MouseEvent('click', { clientX, clientY, bubbles: true }))
      }
      click(); assert(clicks === 1 && moves > 0, 'Compact canvas lost node picking or pointer movement')
      renderer.setSetting('renderEdgeLabels', true); renderer.refresh()
      assert(layers.edgeLabels.width === layers.nodes.width && layers.edgeLabels.height === layers.nodes.height, 'Enabled edge-label resolution not restored')
      const painted = () => layers.edgeLabels.getContext('2d')!.getImageData(0, 0, layers.edgeLabels.width, layers.edgeLabels.height).data.some((v, i) => i % 4 === 3 && v > 0)
      assert(painted(), 'Restored edge-label canvas did not paint text')
      host.style.width = '360px'; host.style.height = '240px'
      renderer.resize(); renderer.refresh()
      assert(layers.mouse.width === 1 && layers.edgeLabels.width === layers.nodes.width && painted(), 'Resize broke backing policy or label painting')
      // Separate clicks, not Sigma's intentional 300ms double-click gesture.
      await delay(renderer.getSetting('doubleClickTimeout') + 20)
      click(); assert(Number(clicks) === 2, 'Resized pointer coordinates broke picking')
      renderer.setSetting('renderEdgeLabels', false); renderer.refresh()
      assert(layers.edgeLabels.width === 1, 'Disabling edge labels did not compact the bitmap')
      // Actual wheel captor, not a direct camera method. It must still receive
      // the event through the full CSS hit surface despite a one-pixel bitmap.
      const ratio = renderer.getCamera().ratio
      layers.mouse.dispatchEvent(new WheelEvent('wheel', { clientX: rect.left + 200, clientY: rect.top + 100, deltaY: -100, bubbles: true, cancelable: true }))
      await delay(350)
      assert(renderer.getCamera().ratio < ratio, 'Compact canvas lost wheel zoom')
      return '5 painted layers pixel-identical; full CSS hit area, node picking/movement before and after resize, wheel zoom, and enabled edge-label DPR/text verified'
    } finally {
      const canvases = Object.values(renderer.getCanvases())
      const contexts = ['nodes', 'edges', 'hoverNodes'].map(key => {
        const canvas = renderer.getCanvases()[key]
        return canvas.getContext('webgl2') ?? canvas.getContext('webgl')
      })
      let shrunkBeforeLoss = false
      renderer.on('kill', () => {
        shrunkBeforeLoss = contexts.every(gl => gl && !gl.isContextLost() && gl.drawingBufferWidth <= 1 && gl.drawingBufferHeight <= 1)
      })
      try { disposeGraphRenderer(renderer) } finally { host.remove() }
      assert(shrunkBeforeLoss, 'WebGL drawing buffers were not shrunk before context loss')
      assert(canvases.every(canvas => !canvas.isConnected && canvas.width === 0 && canvas.height === 0), 'Graph disposal retained a bitmap')
    }
  })
}
