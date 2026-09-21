import type { CatalogArtifact, CatalogModel } from './api'
import { artifactCapabilities } from './artifact-runtime'

export function defaultSpeculation(model: CatalogModel | undefined, weights: CatalogArtifact | undefined, drafter?: CatalogArtifact): string {
  if (model && !artifactCapabilities(model, weights).includes('speculative')) return 'off'
  return weights?.runtime?.default_spec ?? (drafter && !drafter.default ? 'off' : 'on')
}

export function nativeKvOption(dtype: string) {
  const precision: Record<string, string> = { auto: 'Checkpoint-native precision', f16: 'F16 · 16-bit', bf16: 'BF16 · 16-bit', f32: 'F32 · 32-bit' }
  return { value: dtype, label: precision[dtype] ?? dtype, hint: 'Required by this checkpoint and backend' }
}

export function backendAdvancedFields<T extends { key: string; choices?: string[]; hint?: string }>(fields: T[], backend: string, kv: string[]): T[] {
  const unsupported = new Set(['kernel_pack', 'fp8_native', 'moe_offload', 'max_image_tokens'])
  return fields.filter(f => backend !== 'metal' || !unsupported.has(f.key)).map(f => {
    if (f.key === 'device') return { ...f, choices: [backend], hint: 'The backend provided by this manager' }
    if (f.key === 'kv_cache_dtype' && backend === 'metal') return { ...f, choices: kv, hint: 'Checkpoint-native KV precision' }
    return f
  })
}
