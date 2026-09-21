import type Sigma from 'sigma'

/** Sigma's pointer surface receives events in CSS coordinates; it never paints.
 * Disabled edge labels also need no full-resolution bitmap. Keep the DOM/CSS
 * boxes and captors intact, and restore label resolution if enabled later.
 * Use public hooks: resize can reset every bitmap even when labels are off. */
export function budgetGraphCanvases(renderer: Sigma) {
  const sync = () => {
    const { mouse, edgeLabels, nodes } = renderer.getCanvases()
    // A blank 1px bitmap alone still gets a full-CSS-size compositing layer
    // in WebKit. Opacity zero keeps pointer hit-testing but needs no painting.
    if (mouse.style.opacity !== '0') mouse.style.opacity = '0'
    if (mouse.width !== 1 || mouse.height !== 1) { mouse.width = 1; mouse.height = 1 }
    const enabled = renderer.getSetting('renderEdgeLabels')
    if (edgeLabels.hidden !== !enabled) edgeLabels.hidden = !enabled
    const width = enabled ? nodes.width : 1
    const height = enabled ? nodes.height : 1
    if (edgeLabels.width !== width || edgeLabels.height !== height) {
      edgeLabels.width = width; edgeLabels.height = height
      if (enabled) {
        const { width: cssWidth, height: cssHeight } = renderer.getDimensions()
        edgeLabels.getContext('2d')?.setTransform(width / cssWidth, 0, 0, height / cssHeight, 0, 0)
      }
    }
  }
  renderer.on('resize', sync)
  renderer.on('beforeClear', sync)
  sync()
  // Sigma.kill removes these listeners along with its own handlers.
}

/** Shrink drawing buffers before kill() loses the WebGL contexts. WebKit's
 * didUpdateCanvasSizeProperties skips framebuffer reshape on a lost context;
 * zero DOM dimensions after kill are not evidence of a released GL backing.
 * No frame can interleave this synchronous disposal. Sigma still owns listener,
 * program and context teardown, even if a canvas reset unexpectedly throws. */
export function disposeGraphRenderer(renderer: Sigma | null) {
  if (!renderer) return
  const canvases = Object.values(renderer.getCanvases())
  try { for (const canvas of canvases) { canvas.width = 0; canvas.height = 0 } }
  finally { renderer.kill() }
}
