import { describe, expect, it } from 'vitest'
import { backendAdvancedFields, defaultSpeculation, nativeKvOption } from './backend-settings'
import type { CatalogArtifact, CatalogModel } from './api'

const model = { capability: ['chat', 'speculative'] } as CatalogModel
const weights = (default_spec: string, capability = model.capability) => ({ runtime: { default_spec, capability } }) as CatalogArtifact

describe('backend-aware model settings', () => {
  it('honors Qwen/Splash adaptive and Bonsai off without inventing a UI default', () => {
    expect(defaultSpeculation(model, weights('adaptive'))).toBe('adaptive')
    expect(defaultSpeculation(model, weights('off'))).toBe('off')
    expect(defaultSpeculation(model, weights('on', ['chat']))).toBe('off')
    expect(defaultSpeculation(model, undefined, { default: false } as CatalogArtifact)).toBe('off')
    expect(defaultSpeculation(model, undefined)).toBe('on')
  })
  it('labels F32, F16 and checkpoint-native caches without guessing from file format', () => {
    expect(nativeKvOption('f32').label).toBe('F32 · 32-bit')
    expect(nativeKvOption('f16').label).toBe('F16 · 16-bit')
    expect(nativeKvOption('auto').label).toBe('Checkpoint-native precision')
  })
  it('filters CUDA-only controls and keeps the actual Metal precision in Advanced', () => {
    const fields = ['device', 'kv_cache_dtype', 'kernel_pack', 'fp8_native', 'max_image_tokens', 'moe_offload', 'max_ctx'].map(key => ({ key, choices: ['cuda'] }))
    const metal = backendAdvancedFields(fields, 'metal', ['f32'])
    expect(metal.map(f => f.key)).toEqual(['device', 'kv_cache_dtype', 'max_ctx'])
    expect(metal[0]!.choices).toEqual(['metal'])
    expect(metal[1]!.choices).toEqual(['f32'])
    expect(backendAdvancedFields(fields, 'cuda', ['f16'])).toHaveLength(fields.length)
    expect(fields[0]!.choices).toEqual(['cuda'])
  })
})
