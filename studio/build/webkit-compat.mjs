// Inject into page and worker chunks before disposal classes evaluate. Qualify
// globals because graphlib has a hoisted local Symbol. This is a standards shim,
// not eval, and requires no private WebKit preferences or security exceptions.
export const DISPOSAL_SHIM = `for (const key of ['dispose','asyncDispose']) { if (!globalThis.Symbol[key]) { globalThis.Object.defineProperty(globalThis.Symbol,key,{value:globalThis.Symbol.for('Symbol.'+key)}); globalThis.__labDisposeShim = true; } }`
