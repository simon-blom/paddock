// One lifecycle owner shared by Traverse Studio and Paddock. Graphology and
// Sigma stay in the view; only compact layout input enters the disposable worker.
export class LayoutController {
  constructor(graph, options = {}) {
    this.graph = graph; this.options = options; this.worker = null;
    this.frame = null; this.pending = null; this.keys = []; this.disposed = false;
    this.stats = null; this.isRunning = false;
    this.watchdog = null;
  }
  start() {
    if (this.disposed) throw new Error('Layout controller is disposed');
    this.stop();
    this.stats = null;
    const graph = this.graph;
    if (graph.order > 200000 || graph.size > 1000000) throw new Error('Graph exceeds interactive layout budget');
    this.keys = graph.nodes();
    const ids = new Map(this.keys.map((key,i) => [key,i]));
    const n = this.keys.length;
    const positions = new Float32Array(n*2), radii = new Float32Array(n), fixed = new Uint8Array(n);
    graph.forEachNode((key,a) => {
      const i = ids.get(key);
      positions[2*i] = a.x ?? 0; positions[2*i+1] = a.y ?? 0; radii[i] = a.size ?? 5; fixed[i] = a.fixed ? 1 : 0;
    });
    const edges = new Uint32Array(graph.size*2), weights = new Float32Array(graph.size);
    let e = 0;
    graph.forEachEdge((_key,a,source,target) => { edges[2*e] = ids.get(source); edges[2*e+1] = ids.get(target); weights[e++] = a.weight ?? 1; });
    const worker = this.options.workerFactory ? this.options.workerFactory() : new Worker(new URL('./worker.js', import.meta.url), {type:'module'});
    this.worker = worker; this.isRunning = true;
    this.onTopology = () => { this.stop(); this.options.onCancel?.(); };
    for (const event of ['nodeAdded','nodeDropped','edgeAdded','edgeDropped','cleared']) graph.on(event, this.onTopology);
    const fail = (error) => {
      if (this.worker !== worker) return;
      this.stop(); this.options.onError?.(error);
    };
    worker.onerror = () => fail(new Error('Layout worker failed'));
    worker.onmessageerror = () => fail(new Error('Invalid layout worker message'));
    const maxMs = Number.isFinite(this.options.maxMs) ? Math.max(100,Math.min(30000,this.options.maxMs)) : 5000;
    this.watchdog = setTimeout(() => fail(new Error('Layout exceeded its execution budget')),maxMs+5000);
    worker.onmessage = ({data}) => {
      if (this.worker !== worker) return;
      if (data?.kind === 'error') { fail(new Error(data.message)); return; }
      if (!['progress','complete'].includes(data?.kind) || !data.stats) { fail(new Error('Invalid layout worker message')); return; }
      if (!(data.positions instanceof Float32Array) || data.positions.length !== n*2) { fail(new Error('Invalid layout result')); return; }
      for (const value of data.positions) if (!Number.isFinite(value)) { fail(new Error('Nonfinite layout result')); return; }
      this.pending = data;
      if (this.frame !== null) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = null;
        if (this.worker !== worker || !this.pending) return;
        const update = this.pending; this.pending = null;
        // One batched event, not two node-attribute events per node per tick.
        graph.updateEachNodeAttributes((key,a) => {
          const i = ids.get(key);
          return a.fixed ? a : {...a, x:update.positions[2*i], y:update.positions[2*i+1]};
        }, {attributes:['x','y']});
        this.stats = update.stats;
        this.options.onProgress?.(update.stats);
        if (update.kind === 'complete') { this.stop(); this.options.onComplete?.(update.stats); }
        else worker.postMessage({kind:'ack'});
      });
    };
    try { worker.postMessage({kind:'start', input:{positions,radii,edges,weights,fixed}, options:{kind:this.options.kind, maxMs:this.options.maxMs}}, [positions.buffer,radii.buffer,edges.buffer,weights.buffer,fixed.buffer]); }
    catch (error) { fail(error); }
  }
  stop() {
    if (this.watchdog !== null) clearTimeout(this.watchdog);
    this.watchdog = null;
    this.worker?.terminate(); this.worker = null; this.isRunning = false;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null; this.pending = null;
    if (this.onTopology) for (const event of ['nodeAdded','nodeDropped','edgeAdded','edgeDropped','cleared']) this.graph.removeListener(event, this.onTopology);
    this.onTopology = null;
    // Cancellation never runs completion, collision cleanup, or camera resets.
  }
  kill() { this.stop(); this.disposed = true; this.keys = []; }
}
