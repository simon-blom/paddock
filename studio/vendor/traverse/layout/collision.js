// Original bounded spatial collision pass. There is deliberately no set of
// node pairs: crowded buckets are dispersed in linear work before refinement.
// A pathological input may exhaust the pass budget; report that, never claim
// convergence or silently drop nodes. Fixed nodes are never displaced.
export function separate(positions, radii, fixed, { passes = 8, gap = 2, measureOnly = false } = {}) {
  const n = radii.length;
  let maxRadius = 1;
  for (const r of radii) maxRadius = Math.max(maxRadius, r);
  const cellSize = maxRadius * 2 + gap;
  const key = (x, y) => `${x},${y}`;
  let checks = 0, dispersed = 0, overlaps = 0, crowded = 0, requiredScale = 1;
  for (let pass = 0; pass < passes; pass++) {
    const cells = new Map();
    for (let i = 0; i < n; i++) {
      const k = key(Math.floor(positions[2*i] / cellSize), Math.floor(positions[2*i+1] / cellSize));
      let cell = cells.get(k);
      if (!cell) cells.set(k, cell = []);
      cell.push(i);
    }
    overlaps = 0; crowded = 0;
    // A fixed-size grid alone isn't a complexity bound. Avoid comparing every
    // pair in dense cells; deterministically seed a local lattice instead.
    for (const cell of cells.values()) if (cell.length > 32) {
      crowded += cell.length;
      if (measureOnly) continue;
      let x = 0, y = 0;
      for (const i of cell) { x += positions[2*i]; y += positions[2*i+1]; }
      x /= cell.length; y /= cell.length;
      const side = Math.ceil(Math.sqrt(cell.length));
      for (let rank = 0; rank < cell.length; rank++) {
        const i = cell[rank];
        if (fixed[i]) continue;
        positions[2*i] = x + (rank % side - (side-1)/2) * cellSize;
        positions[2*i+1] = y + (Math.floor(rank/side) - (side-1)/2) * cellSize;
        dispersed++;
      }
    }
    // Rebuild next pass after lattice seeding: stale bucket coordinates must
    // not be used for exact overlap checks.
    if (crowded) { if (measureOnly) break; continue; }
    requiredScale = 1;
    const dx = measureOnly ? null : new Float32Array(n), dy = measureOnly ? null : new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = positions[2*i], y = positions[2*i+1];
      const cx = Math.floor(x / cellSize), cy = Math.floor(y / cellSize);
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
        const cell = cells.get(key(cx+ox, cy+oy));
        if (!cell) continue;
        for (const j of cell) {
          if (j <= i) continue;
          checks++;
          let vx = positions[2*j]-x, vy = positions[2*j+1]-y;
          const min = radii[i]+radii[j]+gap;
          let distance = Math.hypot(vx, vy);
          if (distance >= min * .999) continue;
          overlaps++;
          requiredScale = Math.max(requiredScale, distance > 1e-6 ? min/distance : Infinity);
          if (measureOnly) continue;
          if (fixed[i] && fixed[j]) continue;
          if (distance < 1e-6) {
            const angle = ((Math.imul(i+1, 1664525) ^ Math.imul(j+1, 1013904223)) >>> 0) / 4294967296 * Math.PI*2;
            distance = 1e-6; vx = Math.cos(angle)*distance; vy = Math.sin(angle)*distance;
          }
          const push = (min-distance) / distance;
          const a = fixed[j] ? 1 : .5, b = fixed[i] ? 1 : .5;
          if (!fixed[i]) { dx[i] -= vx*push*a; dy[i] -= vy*push*a; }
          if (!fixed[j]) { dx[j] += vx*push*b; dy[j] += vy*push*b; }
        }
      }
    }
    if (measureOnly) break;
    for (let i = 0; i < n; i++) {
      // Damp simultaneous corrections when many neighbors collide.
      const length = Math.hypot(dx[i], dy[i]);
      const scale = length > cellSize ? cellSize / length : 1;
      positions[2*i] += dx[i]*scale; positions[2*i+1] += dy[i]*scale;
    }
    if (!overlaps) break;
  }
  return { candidateChecks: checks, dispersedNodes: dispersed, overlapsBeforeLastPass: overlaps, crowdedNodes: crowded, requiredScale };
}
