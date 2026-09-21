import { createFa2Solver } from './fa2.js';
import { forceSimulation, forceLink, forceManyBody, forceCenter } from 'd3-force';
import { coarsen, refine } from './hierarchy.js';
import { separate } from './collision.js';
import { packComponents } from './components.js';

export const LIMITS = Object.freeze({ nodes: 200000, edges: 1000000, inputBytes: 64*1024*1024 });
export function validate(input) {
  const { positions, radii, edges, weights, fixed } = input;
  if (!(positions instanceof Float32Array) || !(radii instanceof Float32Array) || !(edges instanceof Uint32Array) || !(weights instanceof Float32Array) || !(fixed instanceof Uint8Array)) throw new Error('Invalid layout buffers');
  const n = radii.length;
  if (n > LIMITS.nodes || edges.length/2 > LIMITS.edges || positions.length !== n*2 || edges.length%2 || weights.length !== edges.length/2 || fixed.length !== n) throw new Error('Layout size or buffer shape exceeds limits');
  const bytes = positions.byteLength+radii.byteLength+edges.byteLength+weights.byteLength+fixed.byteLength;
  if (bytes > LIMITS.inputBytes) throw new Error('Layout input exceeds byte budget');
  for (const x of positions) if (!Number.isFinite(x) || Math.abs(x) > 1e8) throw new Error('Invalid layout coordinate');
  for (const r of radii) if (!Number.isFinite(r) || r <= 0 || r > 10000) throw new Error('Invalid layout radius');
  for (const v of edges) if (v >= n) throw new Error('Layout edge endpoint out of range');
  for (const w of weights) if (!Number.isFinite(w) || w < 0 || w > 1e6) throw new Error('Invalid layout edge weight');
  return bytes;
}

function solver(level, kind) {
  const n = level.radii.length;
  if (kind === 'd3') {
    const nodes = Array.from({length:n}, (_, i) => ({ index:i, x:level.positions[2*i], y:level.positions[2*i+1], ...(level.fixed[i] ? {fx:level.positions[2*i], fy:level.positions[2*i+1]} : {}) }));
    const links = Array.from({length:level.edges.length/2}, (_, i) => ({source:level.edges[2*i], target:level.edges[2*i+1]}));
    const simulation = forceSimulation(nodes).stop().randomSource(() => .61803398875)
      .force('links', forceLink(links).distance(30)).force('charge', forceManyBody().strength(-100).theta(.9));
    // Center force would move pinned nodes too; omit it in that case.
    if (!level.fixed.some(Boolean)) simulation.force('center', forceCenter(0,0));
    return () => { simulation.tick(); for (let i = 0; i < n; i++) { level.positions[2*i] = nodes[i].x; level.positions[2*i+1] = nodes[i].y; } };
  }
  return createFa2Solver(level);
}

// Async at iteration/pass boundaries: the worker can receive cancellation and
// acknowledgments. A JS solver iteration is indivisible; the disposable worker
// can be terminated by the controller without affecting the database.
export async function layout(input, options = {}, publish = () => {}, cancelled = () => false) {
  const inputBytes = validate(input), start = performance.now();
  if (options.kind !== undefined && !['fa2','d3'].includes(options.kind)) throw new Error('Unknown layout kind');
  if (options.maxMs !== undefined && !Number.isFinite(options.maxMs)) throw new Error('Invalid layout time budget');
  const maxMs = Math.max(100, Math.min(30000, options.maxMs ?? 5000));
  const forceDeadline = start + maxMs*.6;
  const levels = [input];
  const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));
  const check = () => { if (cancelled()) throw new DOMException('Layout cancelled', 'AbortError'); };
  if (!input.fixed.some(Boolean)) while (levels.at(-1).radii.length > 128 && levels.length < 16) {
    const next = coarsen(levels.at(-1));
    if (!next) break;
    levels.push(next); await yieldTask(); check();
  }
  const coarse = levels.at(-1);
  // Stable noncoincident seed for new multilevel layouts. Single-level jobs
  // preserve incoming positions, including pinned vertices.
  if (levels.length > 1) {
    let spacing = 4;
    for (const r of coarse.radii) spacing = Math.max(spacing, 2*r+4);
    const side = Math.ceil(Math.sqrt(coarse.radii.length));
    for (let i = 0; i < coarse.radii.length; i++) { coarse.positions[2*i] = (i%side-side/2)*spacing; coarse.positions[2*i+1] = (Math.floor(i/side)-side/2)*spacing; }
  }
  let iterations = 0, lastPublish = 0, reason = 'iterations';
  let collisions = {candidateChecks:0, dispersedNodes:0, overlapsBeforeLastPass:0, crowdedNodes:0};
  for (let l = levels.length-1; l >= 0; l--) {
    check();
    const level = levels[l];
    if (l < levels.length-1) refine(level, levels[l+1]);
    // Disperse coincident buckets before BH tree construction, including for
    // single-level inputs. A pinned coincidence remains reported, not moved.
    separate(level.positions, level.radii, level.fixed, {passes:2});
    if (performance.now() >= forceDeadline) { reason = 'time-budget'; continue; }
    const step = solver(level, options.kind ?? 'fa2');
    const allocatedMs = maxMs*.5/levels.length;
    const levelStart = performance.now();
    for (let k = 0; k < (l === levels.length-1 ? 80 : 30); k++) {
      check();
      if (performance.now() >= forceDeadline) { reason = 'time-budget'; break; }
      step(); iterations++;
      for (const v of level.positions) if (!Number.isFinite(v) || Math.abs(v)>1e8) throw new Error('Layout produced invalid coordinates');
      const now = performance.now();
      if (l === 0 && now-lastPublish > 100) { publish(level.positions, {phase:'refine', iterations}); lastPublish = now; }
      await yieldTask();
      if (now-levelStart > allocatedMs) { reason = 'time-budget'; break; }
    }
  }
  const components = packComponents(input);
  for (let pass = 0; pass < 24; pass++) {
    check();
    const stats = separate(input.positions, input.radii, input.fixed, {passes:1});
    collisions = {...stats, candidateChecks:collisions.candidateChecks+stats.candidateChecks, dispersedNodes:collisions.dispersedNodes+stats.dispersedNodes};
    await yieldTask();
    if (!stats.overlapsBeforeLastPass && !stats.crowdedNodes) break;
    if (performance.now()-start > maxMs) { reason = 'time-budget'; break; }
  }
  // A small final uniform expansion preserves angles/relative distances while
  // resolving residual overlaps that iterative nudges approach asymptotically.
  // Cap expansion: pathological coincidences/pinned inputs remain explicit.
  let audit = separate(input.positions,input.radii,input.fixed,{passes:1,measureOnly:true});
  const expansion = !input.fixed.some(Boolean) && !audit.crowdedNodes && Number.isFinite(audit.requiredScale)
    ? Math.min(4,audit.requiredScale > 1 ? audit.requiredScale*1.01 : 1) : 1;
  if (expansion > 1) {
    for (let i = 0; i < input.positions.length; i++) input.positions[i] *= expansion;
    packComponents(input);
    audit = separate(input.positions,input.radii,input.fixed,{passes:1,measureOnly:true});
  }
  check();
  for (const value of input.positions) if (!Number.isFinite(value) || Math.abs(value)>1e8) throw new Error('Layout produced invalid final coordinates');
  return {positions:input.positions, stats:{phase:'complete', reason, iterations, levels:levels.length, components, expansion,
    nodes:input.radii.length, edges:input.edges.length/2, inputBytes, elapsedMs:performance.now()-start, ...collisions,
    requiredScale: undefined, remainingOverlaps:audit.crowdedNodes ? null : audit.overlapsBeforeLastPass, crowdedNodes:audit.crowdedNodes}};
}
