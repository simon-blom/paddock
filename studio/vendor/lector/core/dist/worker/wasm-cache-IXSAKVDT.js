import "./chunk-UAWBPTDW.js";

// Paddock uses shipped assets and a bounded worker-lifetime compiled-module
// cache. No browser CacheStorage; terminating the worker releases this cache.
const modules = new Map();
async function loadWasmCached(wasmUrl, imports) {
  let pending = modules.get(wasmUrl);
  if (!pending) {
    pending = (async () => {
      const response = await fetch(wasmUrl);
      if (!response.ok) throw new Error("PDF engine download failed: " + response.status);
      if (typeof WebAssembly.compileStreaming === "function") {
        try { return await WebAssembly.compileStreaming(response.clone()); } catch { /* MIME fallback */ }
      }
      return await WebAssembly.compile(await response.arrayBuffer());
    })();
    if (modules.size >= 4) modules.delete(modules.keys().next().value);
    modules.set(wasmUrl, pending);
  }
  let module;
  try { module = await pending; } catch (error) {
    if (modules.get(wasmUrl) === pending) modules.delete(wasmUrl);
    throw error;
  }
  return { module, instance: await WebAssembly.instantiate(module, imports) };
}
function createInstantiateWasmHook(wasmUrl) {
  return (imports, receiveInstance) => {
    loadWasmCached(wasmUrl, imports).then(({ instance, module }) => {
      receiveInstance(instance, module);
    }).catch((err) => { throw err; });
    return {};
  };
}
export { createInstantiateWasmHook, loadWasmCached };
