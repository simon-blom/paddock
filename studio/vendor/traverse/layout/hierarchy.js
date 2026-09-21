// Edge matching and contraction, O(V+E) per level, at most 16 levels. Vertex
// arrays shrink geometrically; sparse edge arrays need not. Parallel edges
// become weights; original edges/IDs stay
// in the caller. Fixed-node graphs use the single-level path.
export function coarsen(level) {
  const n = level.radii.length;
  const mates = new Int32Array(n).fill(-1);
  for (let e = 0; e < level.edges.length; e += 2) {
    const a = level.edges[e], b = level.edges[e+1];
    if (a !== b && mates[a] === -1 && mates[b] === -1) { mates[a] = b; mates[b] = a; }
  }
  const parents = new Uint32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (mates[i] >= 0 && mates[i] < i) continue;
    parents[i] = count;
    if (mates[i] >= 0) parents[mates[i]] = count;
    count++;
  }
  if (count > n * .8) return null;
  const positions = new Float32Array(count*2), radii = new Float32Array(count), mass = new Uint32Array(count);
  for (let i = 0; i < n; i++) {
    const p = parents[i]; mass[p]++;
    positions[p*2] += level.positions[i*2]; positions[p*2+1] += level.positions[i*2+1];
    radii[p] += level.radii[i] ** 2;
  }
  for (let i = 0; i < count; i++) { positions[2*i] /= mass[i]; positions[2*i+1] /= mass[i]; radii[i] = Math.sqrt(radii[i]); }
  const weights = new Map();
  for (let e = 0; e < level.edges.length; e += 2) {
    let a = parents[level.edges[e]], b = parents[level.edges[e+1]];
    if (a === b) continue;
    if (a > b) [a,b] = [b,a];
    // n <= 200k, so the integer key is exactly representable in a double.
    const k = a*count+b;
    weights.set(k, (weights.get(k) ?? 0) + level.weights[e/2]);
  }
  const edges = new Uint32Array(weights.size*2), edgeWeights = new Float32Array(weights.size);
  let e = 0;
  for (const [k,w] of weights) { edges[e*2] = Math.floor(k/count); edges[e*2+1] = k%count; edgeWeights[e++] = w; }
  level.parents = parents;
  return { positions, radii, edges, weights: edgeWeights, fixed: new Uint8Array(count) };
}

export function refine(fine, coarse) {
  for (let i = 0; i < fine.radii.length; i++) {
    const p = fine.parents[i];
    const angle = i * 2.399963229728653;
    const radius = fine.radii[i] + 2;
    fine.positions[2*i] = coarse.positions[2*p] + Math.cos(angle)*radius;
    fine.positions[2*i+1] = coarse.positions[2*p+1] + Math.sin(angle)*radius;
  }
}
