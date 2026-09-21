// Version-pinned adapter to Graphology FA2 0.10.1's iteration primitive.
// Keep its adaptive displacement/convergence state for the entire level.
// Layout tests compare this buffer adapter against the public solver API.
import forceAtlas2 from 'graphology-layout-forceatlas2';
import iterate from 'graphology-layout-forceatlas2/iterate.js';
import defaults from 'graphology-layout-forceatlas2/defaults.js';

export function settingsFor(n) {
  return {...defaults, ...forceAtlas2.inferSettings(n), barnesHutOptimize:true,
    barnesHutTheta:.6, adjustSizes:true, gravity:.1, strongGravityMode:true, scalingRatio:10};
}

export function createFa2Solver(level) {
  const n = level.radii.length, nodes = new Float32Array(n*10);
  const edges = new Float32Array(level.weights.length*3);
  for (let i = 0; i < n; i++) {
    nodes[10*i] = level.positions[2*i]; nodes[10*i+1] = level.positions[2*i+1];
    nodes[10*i+6] = 1; nodes[10*i+7] = 1;
    nodes[10*i+8] = level.radii[i]; nodes[10*i+9] = level.fixed[i];
  }
  for (let e = 0; e < level.weights.length; e++) {
    const a = level.edges[2*e]*10, b = level.edges[2*e+1]*10, weight = level.weights[e];
    edges[3*e] = a; edges[3*e+1] = b; edges[3*e+2] = weight;
    nodes[a+6] += weight; nodes[b+6] += weight;
  }
  const settings = settingsFor(n);
  return () => {
    iterate(settings, nodes, edges);
    for (let i = 0; i < n; i++) {
      level.positions[2*i] = nodes[10*i]; level.positions[2*i+1] = nodes[10*i+1];
    }
  };
}
