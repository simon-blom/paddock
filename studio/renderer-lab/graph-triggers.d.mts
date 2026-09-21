import type Graph from 'graphology'
import type Sigma from 'sigma'
export interface GraphTriggerRow {
  kind: 'refresh' | 'process' | 'interaction'
  start: number
  ms: number
  causes: string[]
  full?: boolean
  reindex?: boolean
  scheduled?: boolean
  nodes?: number | null
  edges?: number | null
}
export function traceGraphTriggers(prototype: object, graph: Graph, record: (row: GraphTriggerRow) => void,
  now?: () => number, discovered?: (renderer: Sigma) => void): () => void
