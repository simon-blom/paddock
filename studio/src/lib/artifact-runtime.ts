import type { CatalogArtifact, CatalogModel } from './api'

/** Runner requirements, not the OS of the browser displaying a remote Studio. */
export function artifactPlatform(artifact: Pick<CatalogArtifact, 'runtime'>): string | null {
  const backends = artifact.runtime?.backends
  return backends?.length === 1 && backends[0] === 'metal'
    ? 'macOS · Apple Silicon · Metal'
    : null
}

export function embeddedVision(weights?: CatalogArtifact): boolean {
  return weights?.runtime?.embedded_vision === true
}

/** A loader limit remains binding even when device memory telemetry is absent. */
export function artifactContextCap(weights: CatalogArtifact | undefined, estimatedCap = 0): number {
  const caps = [weights?.runtime?.memory?.max_ctx ?? 0, estimatedCap].filter(c => c > 0)
  return caps.length ? Math.min(...caps) : 0
}

/** Availability is a backend contract, not a test-completion label. */
export function automaticWeights(artifacts: readonly CatalogArtifact[]): CatalogArtifact[] {
  return artifacts.filter(a => a.backend_supported !== false)
}

/** Export restrictions replace family capabilities; they do not extend them. */
export function artifactCapabilities(model: CatalogModel, weights?: CatalogArtifact): string[] {
  return weights?.runtime?.capability ?? model.capability
}

/** An omitted list inherits companions; an empty list deliberately allows none. */
export function companionAllowed(weights: CatalogArtifact | undefined, companion: CatalogArtifact): boolean {
  const allowed = weights?.runtime?.companions
  return companion.backend_supported !== false && (allowed === undefined || allowed.includes(companion.id))
}
