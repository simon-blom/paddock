import { describe, expect, it, vi } from 'vitest'
import type Sigma from 'sigma'
import { budgetGraphCanvases, disposeGraphRenderer } from './canvas-budget'

function harness() {
  let enabled = false
  const events = new Map<string, () => void>()
  const transform = vi.fn()
  const canvas = () => ({ width: 1722, height: 1220, hidden: false, style: { width: '861px', height: '610px' }, getContext: () => ({ setTransform: transform }) })
  const layers = { mouse: canvas(), edgeLabels: canvas(), nodes: canvas(), labels: canvas(), hovers: canvas() }
  const renderer = { on: (event: string, callback: () => void) => events.set(event, callback),
    getCanvases: () => layers, getSetting: () => enabled, getDimensions: () => ({ width: 861, height: 610 }) } as unknown as Sigma
  return { renderer, layers, transform, emit: (event: string) => events.get(event)!(), enable: (value: boolean) => { enabled = value } }
}

describe('graph canvas backing budget', () => {
  it('shrinks only blank bitmaps, preserving CSS hit boxes and painted layers', () => {
    const h = harness(); budgetGraphCanvases(h.renderer)
    for (const key of ['mouse', 'edgeLabels'] as const) {
      expect([h.layers[key].width, h.layers[key].height]).toEqual([1, 1])
      expect(h.layers[key].style).toMatchObject({ width: '861px', height: '610px' })
    }
    for (const key of ['nodes', 'labels', 'hovers'] as const) expect(h.layers[key].width).toBe(1722)
    expect(h.transform).not.toHaveBeenCalled()
  })
  it('restores enabled edge labels at renderer resolution and resets the 2D transform once', () => {
    const h = harness(); budgetGraphCanvases(h.renderer)
    h.enable(true); h.emit('beforeClear')
    expect([h.layers.edgeLabels.width, h.layers.edgeLabels.height]).toEqual([1722, 1220])
    expect(h.transform).toHaveBeenCalledExactlyOnceWith(2, 0, 0, 2, 0, 0)
    h.emit('beforeClear'); expect(h.transform).toHaveBeenCalledTimes(1)
    expect(h.layers.edgeLabels.hidden).toBe(false)
    h.enable(false); h.emit('beforeClear'); expect(h.layers.edgeLabels.width).toBe(1)
    expect(h.layers.edgeLabels.hidden).toBe(true)
  })
  it('reapplies after Sigma resizes every bitmap without changing visible resolution', () => {
    const h = harness(); budgetGraphCanvases(h.renderer)
    for (const layer of Object.values(h.layers)) { layer.width = 861; layer.height = 610 }
    h.emit('resize')
    expect(h.layers.mouse.width).toBe(1); expect(h.layers.edgeLabels.width).toBe(1)
    h.enable(true); h.emit('beforeClear')
    expect([h.layers.edgeLabels.width, h.layers.edgeLabels.height]).toEqual([861, 610])
    expect(h.transform).toHaveBeenCalledExactlyOnceWith(1, 0, 0, 1, 0, 0)
  })
  it('shrinks every bitmap before Sigma loses its contexts, including a kill error path', () => {
    const h = harness()
    const kill = vi.fn(() => {
      for (const canvas of Object.values(h.layers)) expect([canvas.width, canvas.height]).toEqual([0, 0])
      throw new Error('test disposal failure')
    })
    h.renderer.kill = kill
    expect(() => disposeGraphRenderer(h.renderer)).toThrow('test disposal failure')
    expect(kill).toHaveBeenCalledTimes(1)
    for (const canvas of Object.values(h.layers)) expect([canvas.width, canvas.height]).toEqual([0, 0])
    expect(() => disposeGraphRenderer(null)).not.toThrow()
  })
  it('still kills Sigma if resetting a canvas throws', () => {
    const h = harness(); h.renderer.kill = vi.fn()
    Object.defineProperty(h.layers.mouse, 'width', { set: () => { throw new Error('reset failed') } })
    expect(() => disposeGraphRenderer(h.renderer)).toThrow('reset failed')
    expect(h.renderer.kill).toHaveBeenCalledTimes(1)
  })
  it('does not rewrite visibility on every clear', () => {
    const h = harness(); budgetGraphCanvases(h.renderer)
    const setter = vi.fn()
    Object.defineProperty(h.layers.edgeLabels, 'hidden', { get: () => true, set: setter })
    for (let i = 0; i < 100; i++) h.emit('beforeClear')
    expect(setter).not.toHaveBeenCalled()
  })
})
