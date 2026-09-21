// Bundled module worker + WASM probe, independent of any renderer's fallback.
const wasm = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])
WebAssembly.instantiate(wasm).then(() => {
  postMessage({ module: import.meta.url, wasm: true })
}).catch(error => postMessage({ error: String(error) }))
