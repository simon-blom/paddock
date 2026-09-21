import type Graph from 'graphology';
export interface LayoutStats {
  phase: 'refine' | 'complete';
  iterations: number;
  reason?: 'iterations' | 'time-budget';
  levels?: number;
  components?: number;
  expansion?: number;
  /** Exact graph-coordinate overlap count, or null for unaudited dense cells.
   * This does not establish label/viewport readability or layout quality. */
  remainingOverlaps?: number | null;
  nodes?: number;
  edges?: number;
  inputBytes?: number;
  elapsedMs?: number;
  candidateChecks?: number;
  dispersedNodes?: number;
  overlapsBeforeLastPass?: number;
  crowdedNodes?: number;
}
export interface LayoutOptions {
  kind?: 'fa2' | 'd3';
  maxMs?: number;
  onProgress?: (stats: LayoutStats) => void;
  onComplete?: (stats: LayoutStats) => void;
  onError?: (error: Error) => void;
  /** Automatic cancellation after a topology change; never completion. */
  onCancel?: () => void;
  workerFactory?: () => Worker;
}
export class LayoutController {
  constructor(graph: Graph, options?: LayoutOptions);
  stats: LayoutStats | null;
  isRunning: boolean;
  start(): void;
  stop(): void;
  kill(): void;
}
