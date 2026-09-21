// Bounded admission shared by every database operation. Cancelling a promise
// does NOT make a synchronous WASM call stop: retain its physical slot until
// the worker replies. The database must never be killed to cancel a query.
export function payloadBytes(value) {
  const seen = new Set(), stack = [value];
  let bytes = 0;
  while (stack.length) {
    const v = stack.pop();
    if (typeof v === 'string') bytes += v.length*3;
    else if (v && typeof v === 'object' && !seen.has(v)) {
      seen.add(v);
      if (ArrayBuffer.isView(v)) stack.push(v.buffer);
      else if (v instanceof ArrayBuffer) bytes += v.byteLength;
      else {
        if (!Array.isArray(v) && Object.getPrototypeOf(v) !== Object.prototype && Object.getPrototypeOf(v) !== null) throw new TypeError('Worker payload must contain JSON-like values or byte buffers');
        for (const key in v) if (Object.hasOwn(v,key)) {
          bytes += key.length*3+16;
          if (stack.length+seen.size >= 100000 || bytes > 256*1024*1024) throw new RangeError('Worker payload exceeds admission budget');
          stack.push(v[key]);
        }
      }
    } else bytes += 8;
    if (bytes > 256*1024*1024 || seen.size > 100000) throw new RangeError('Worker payload exceeds admission budget');
  }
  return bytes;
}

export class RequestQueue {
  constructor(send, {maxRequests=32, maxBytes=256*1024*1024} = {}) {
    this.send=send; this.maxRequests=maxRequests; this.maxBytes=maxBytes;
    this.entries=new Map(); this.waiting=[]; this.active=null; this.bytes=0; this.next=0; this.closed=false;
  }
  call(kind,payload,options={}) {
    return new Promise((resolve,reject) => {
      if (this.closed) { reject(new DOMException('Database is closed','InvalidStateError')); return; }
      const signal=options.signal;
      if (signal?.aborted) { reject(new DOMException('Aborted before dispatch','AbortError')); return; }
      let bytes;
      try { bytes=payloadBytes(payload); } catch (e) { reject(e); return; }
      if (this.entries.size >= this.maxRequests || this.bytes+bytes > this.maxBytes) { reject(new RangeError('Database worker queue is full')); return; }
      // Preserve call-time values and the admitted size while waiting. A
      // caller mutating its params later cannot enlarge a queued request.
      try { payload=structuredClone(payload); } catch (e) { reject(e); return; }
      if (signal?.aborted) { reject(new DOMException('Aborted before dispatch','AbortError')); return; }
      const id=++this.next;
      const entry={id,kind,payload,resolve,reject,bytes,signal,onAbort:null,aborted:false};
      entry.onAbort=()=> {
        if (!this.entries.has(id)) return;
        entry.aborted=true;
        const running=this.active===entry;
        const error=new DOMException(running ? 'Caller cancelled; execution may finish before its deadline' : 'Aborted before dispatch','AbortError');
        // Callers of mutating operations must not interpret AbortError as a
        // rollback guarantee. The worker keeps all state until normal close.
        error.executionMayHaveCompleted=running;
        reject(error);
        signal?.removeEventListener('abort',entry.onAbort);
        if (!running) { this.waiting=this.waiting.filter(e=>e!==entry); this.release(entry); }
      };
      signal?.addEventListener('abort',entry.onAbort,{once:true});
      this.entries.set(id,entry); this.bytes+=bytes; this.waiting.push(entry); this.drain();
    });
  }
  drain() {
    if (this.closed || this.active || !this.waiting.length) return;
    const entry=this.waiting.shift(); this.active=entry;
    try {
      this.send({id:entry.id,kind:entry.kind,payload:entry.payload});
      entry.payload=null; // postMessage has captured it; keep only the slot/charge.
    }
    catch (error) { entry.reject(error); this.release(entry); this.active=null; this.drain(); }
  }
  reply({id,ok,result,error}) {
    const entry=this.active;
    if (!entry || entry.id!==id) return;
    if (!entry.aborted) {
      if (ok) entry.resolve(result);
      else { const e=new Error(error?.message ?? 'WASM operation failed'); e.code=error?.code; entry.reject(e); }
    }
    this.release(entry); this.active=null; this.drain();
  }
  release(entry) {
    entry.signal?.removeEventListener('abort',entry.onAbort);
    this.entries.delete(entry.id); this.bytes-=entry.bytes; entry.payload=null;
  }
  close(error=new DOMException('Database closed','AbortError')) {
    if (this.closed) return;
    this.closed=true;
    for (const entry of this.entries.values()) { entry.reject(error); this.release(entry); }
    this.waiting=[]; this.active=null;
  }
  get stats() { return {active:this.active ? 1:0, queued:this.waiting.length, bytes:this.bytes, closed:this.closed}; }
}
