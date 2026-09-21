import { layout } from './engine.js';
let active = false, awaitingAck = false;
self.onmessage = async ({data}) => {
  if (data.kind === 'ack') { awaitingAck = false; return; }
  if (data.kind !== 'start' || active) return;
  active = true;
  try {
    const result = await layout(data.input, data.options, (positions, stats) => {
      if (awaitingAck) return;
      awaitingAck = true;
      const copy = positions.slice();
      self.postMessage({kind:'progress', positions:copy, stats}, [copy.buffer]);
    });
    // One terminal frame is allowed alongside the sole outstanding preview.
    self.postMessage({kind:'complete', ...result}, [result.positions.buffer]);
  } catch (error) {
    self.postMessage({kind:'error', message:error instanceof Error ? error.message : String(error)});
  }
};
