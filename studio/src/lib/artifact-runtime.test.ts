import { describe, expect, it } from 'vitest'
import type { CatalogArtifact, CatalogModel } from './api'
import { artifactCapabilities, artifactContextCap, artifactPlatform, automaticWeights, companionAllowed, embeddedVision } from './artifact-runtime'
import { ctxFits } from './ctx-ladder'
import { archBlocked, archBlockReason } from './arch-floor'

const weights: CatalogArtifact = {
  id: 'mlx-4bit', kind: 'weights', format: 'safetensors', label: 'MLX-community 4-bit',
  files: [], installed: false, total_size: 0,
  runtime: { backends: ['metal'], capability: ['chat', 'tools', 'reasoning'], companions: [], kv_cache_dtype: 'auto' },
}
const family: CatalogModel = {
  id: 'qwen3.8-27b', display: 'Qwen 3.8 27B',
  capability: ['chat', 'tools', 'reasoning', 'vision', 'speculative'],
  artifacts: [weights], installed: false, total_size: 0,
}
const tower: CatalogArtifact = { ...weights, id: 'vision', kind: 'vision', format: 'gguf', runtime: undefined }

describe('artifact runtime contracts', () => {
  it('respects native loader context limits without NVIDIA memory telemetry', () => {
    const bounded = { ...weights, runtime: { memory: { max_ctx: 4096, max_batch: 4 } } }
    expect(artifactContextCap(bounded)).toBe(4096)
    expect(artifactContextCap(bounded, 32768)).toBe(4096)
    expect(artifactContextCap(bounded, 2048)).toBe(2048)
    expect(artifactContextCap(weights, 32768)).toBe(32768)
    expect(ctxFits({ vramCap: 0, modelCap: artifactContextCap(bounded) })).toEqual([4096])
    expect(ctxFits({ vramCap: 0, modelCap: 2048 })).toEqual([2048])
  })

  it('ignores legacy qualification labels but still excludes unsupported exports', () => {
    const unqualified: CatalogArtifact = { ...weights, installed: true, default: true, runtime: { qualification: 'unqualified' } }
    expect(automaticWeights([unqualified, { ...weights, backend_supported: false }, weights])).toEqual([unqualified, weights])
    expect(automaticWeights([unqualified])).toEqual([unqualified])
  })

  it('does not inherit vision or speculation from the family into native MLX', () => {
    expect(artifactCapabilities(family, weights)).toEqual(['chat', 'tools', 'reasoning'])
    expect(artifactCapabilities(family, { ...weights, runtime: undefined })).toEqual(family.capability)
  })

  it('distinguishes omitted companion lists from explicitly empty lists', () => {
    expect(companionAllowed(weights, tower)).toBe(false)
    expect(companionAllowed({ ...weights, runtime: undefined }, tower)).toBe(true)
    expect(companionAllowed({ ...weights, runtime: { companions: ['vision'] } }, tower)).toBe(true)
    expect(companionAllowed(undefined, { ...tower, backend_supported: false })).toBe(false)
  })

  it('blocks a Metal-only export even on the newest CUDA card', () => {
    const wrongBackend = { ...weights, backend_supported: false }
    expect(archBlocked(wrongBackend, [12, 0])).toBe(true)
    expect(archBlockReason(wrongBackend, undefined)).toBe('Needs macOS on Apple Silicon (Metal runner)')
    expect(archBlocked({ ...weights, backend_supported: true }, undefined)).toBe(false)
  })

  it('labels the runner platform without guessing the browser OS', () => {
    expect(artifactPlatform(weights)).toBe('macOS · Apple Silicon · Metal')
    expect(artifactPlatform({ runtime: { backends: ['cuda', 'metal'] } })).toBeNull()
    expect(artifactPlatform({ runtime: undefined })).toBeNull()
    expect(archBlocked({ ...weights, backend_supported: true }, undefined)).toBe(false)
  })

  it('identifies embedded vision without attaching the family GGUF tower', () => {
    const vision: CatalogArtifact = { ...weights, runtime: { ...weights.runtime, embedded_vision: true, capability: ['chat', 'vision'] } }
    expect(embeddedVision(vision)).toBe(true)
    expect(artifactCapabilities(family, vision)).toEqual(['chat', 'vision'])
    expect(companionAllowed(vision, tower)).toBe(false)
    expect(embeddedVision(weights)).toBe(false)
    expect(embeddedVision(undefined)).toBe(false)
  })

  it('preserves the existing compute-capability floor for CUDA exports', () => {
    expect(archBlocked({ min_cc: [12, 0] }, [8, 6])).toBe(true)
    expect(archBlockReason({ min_cc: [12, 0] }, [8, 6])).toBe('Needs a Blackwell GPU')
    expect(archBlocked({ min_cc: [12, 0] }, [12, 0])).toBe(false)
  })
})
