/** Read-only analysis of WebKit Inspector v2/v3 snapshots (4-field nodes).
 * BFS reports one shortest root path, not a dominator/retained-size estimate.
 * Native malloc/graphics bytes are not included in JS node self-size sums.
 */
import { readFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'
import { resolve } from 'node:path'

export function summarizeHeap(snapshot, pattern = /HTML.*Element|Text$|Worker|Graph|Sigma/) {
  const { nodes, edges, nodeClassNames: classes, edgeTypes, edgeNames } = snapshot
  if (![2, 3].includes(snapshot.version) || snapshot.type !== 'Inspector'
    || nodes.length % 4 || edges.length % 4) throw new Error('Unsupported WebKit heap format')
  const index = new Map()
  const counts = new Map()
  let selfBytes = 0
  for (let i = 0; i < nodes.length; i += 4) {
    index.set(nodes[i], i)
    const name = classes[nodes[i + 2]]
    const row = counts.get(name) ?? { name, count: 0, selfBytes: 0 }
    row.count++; row.selfBytes += nodes[i + 1]; selfBytes += nodes[i + 1]
    counts.set(name, row)
  }
  const describe = id => {
    const i = index.get(id)
    return { id, class: classes[nodes[i + 2]], selfBytes: nodes[i + 1], flags: nodes[i + 3] }
  }
  const outgoing = new Map()
  const graphShape = new Map()
  for (let i = 0; i < edges.length; i += 4) {
    const row = outgoing.get(edges[i]) ?? []
    row.push(i); outgoing.set(edges[i], row)
    if (edgeTypes[edges[i + 2]] === 'Property') {
      const bit = { source: 1, target: 2, in: 4, out: 8 }[edgeNames[edges[i + 3]]] ?? 0
      if (bit) graphShape.set(edges[i], (graphShape.get(edges[i]) ?? 0) | bit)
    }
  }
  const visited = new Map([[0, -1]])
  const queue = [0]
  for (let cursor = 0; cursor < queue.length; cursor++) {
    for (const e of outgoing.get(queue[cursor]) ?? []) {
      const to = edges[e + 1]
      if (visited.has(to)) continue
      visited.set(to, e); queue.push(to)
    }
  }
  const path = id => {
    const result = []
    for (let cursor = id; cursor !== 0;) {
      const e = visited.get(cursor)
      if (e === undefined) return [{ ...describe(cursor), unreachableFromRoot: true }, ...result]
      const type = edgeTypes[edges[e + 2]]
      result.unshift({ ...describe(cursor), via: type, name: ['Property', 'Variable'].includes(type) ? edgeNames[edges[e + 3]] : edges[e + 3] })
      cursor = edges[e]
    }
    return [describe(0), ...result]
  }
  const selected = [...counts.values()].filter(r => pattern.test(r.name)).sort((a, b) => b.count - a.count)
  let reachableSelfBytes = 0
  for (const id of visited.keys()) reachableSelfBytes += nodes[index.get(id) + 1]
  const examples = selected.map(row => {
    // One example per matching class; not a claim that every object has the
    // same owner. Keep the raw snapshot for paths to all other instances.
    const id = [...index.keys()].find(id => describe(id).class === row.name)
    return { ...row, path: path(id) }
  })
  // Minified constructor names change per build. Identify Graphology-shaped
  // records by their fields as well; this is not a retained-size calculation.
  const graphRecords = new Map()
  for (const [id, shape] of graphShape) {
    const role = (shape & 3) === 3 ? 'edge-source-target' : (shape & 12) === 12 ? 'node-in-out' : null
    if (!role) continue
    const node = describe(id); const key = `${role}/${node.class}`
    const row = graphRecords.get(key) ?? { role, class: node.class, count: 0, selfBytes: 0, recordedRootReachable: 0, examplePath: path(id) }
    row.count++; row.selfBytes += node.selfBytes; row.recordedRootReachable += visited.has(id) ? 1 : 0
    graphRecords.set(key, row)
  }
  return { version: snapshot.version, nodeCount: nodes.length / 4, edgeCount: edges.length / 4, selfBytes,
    recordedRootReachableNodes: visited.size, recordedRootReachableSelfBytes: reachableSelfBytes,
    limitation: 'Missing root edges are not proof of dead or leaked objects; self sizes are not physical footprint or retained size.',
    graphRecords: [...graphRecords.values()],
    classes: [...counts.values()].sort((a, b) => b.selfBytes - a.selfBytes).slice(0, 30), selected: examples }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const snapshot = JSON.parse(await readFile(process.argv[2], 'utf8'))
  console.log(JSON.stringify(summarizeHeap(snapshot, process.argv[3] ? new RegExp(process.argv[3]) : undefined), null, 2))
}
