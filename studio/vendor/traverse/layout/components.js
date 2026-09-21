// Pack disconnected components without altering their internal geometry.
// Union/find + shelf packing: O(V+E+K log K), no component pair matrix.
export function packComponents({positions, radii, edges, fixed}, gap = 12) {
  const n = radii.length;
  if (!n || fixed.some(Boolean)) return 0;
  const parent = Uint32Array.from({length:n}, (_,i) => i), rank = new Uint8Array(n);
  const root = i => { while (i !== parent[i]) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let e = 0; e < edges.length; e += 2) {
    let a = root(edges[e]), b = root(edges[e+1]);
    if (a === b) continue;
    if (rank[a] < rank[b]) [a,b] = [b,a];
    parent[b] = a; if (rank[a] === rank[b]) rank[a]++;
  }
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const r = root(i); parent[i] = r;
    let box = groups.get(r);
    if (!box) groups.set(r,box={minX:Infinity,minY:Infinity,maxX:-Infinity,maxY:-Infinity});
    box.minX = Math.min(box.minX,positions[2*i]-radii[i]); box.maxX = Math.max(box.maxX,positions[2*i]+radii[i]);
    box.minY = Math.min(box.minY,positions[2*i+1]-radii[i]); box.maxY = Math.max(box.maxY,positions[2*i+1]+radii[i]);
  }
  if (groups.size === 1) return 1;
  const boxes = [...groups.values()];
  let area = 0, maxWidth = 0;
  for (const b of boxes) {
    b.width = b.maxX-b.minX+gap; b.height = b.maxY-b.minY+gap;
    area += b.width*b.height; maxWidth = Math.max(maxWidth,b.width);
  }
  boxes.sort((a,b) => b.height-a.height);
  const width = Math.max(maxWidth,Math.sqrt(area)*1.2);
  let x = 0, y = 0, rowHeight = 0;
  for (const b of boxes) {
    if (x && x+b.width > width) { x = 0; y += rowHeight; rowHeight = 0; }
    b.dx = x-b.minX; b.dy = y-b.minY; x += b.width; rowHeight = Math.max(rowHeight,b.height);
  }
  for (let i = 0; i < n; i++) {
    const b = groups.get(parent[i]);
    positions[2*i] += b.dx-width/2; positions[2*i+1] += b.dy-(y+rowHeight)/2;
  }
  return groups.size;
}
