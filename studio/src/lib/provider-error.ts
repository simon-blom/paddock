/** Display-only repair of old tool-loop errors cut inside JSON. New relays
 * normalize the provider envelope before bounding it. Never show raw metadata. */
export function providerErrorMessage(raw: string): string {
  const text=raw.replace(/^provider error:\s*/, '').trim().slice(0,128*1024)
  if (!text.startsWith('{')) return text
  const field=(key:string):string|undefined=>{
    const found=new RegExp('"'+key+'"\\s*:\\s*("(?:[^"\\\\]|\\\\.)*")').exec(text)
    try { return found ? JSON.parse(found[1]) : undefined } catch { return undefined }
  }
  let error: Record<string,unknown> | undefined
  try { const v=JSON.parse(text); error=v.error ?? v } catch { /* legacy cut */ }
  const metadata=error?.metadata as Record<string,unknown> | undefined
  const provider=metadata?.provider_name ?? field('provider_name')
  const code=error?.code ?? Number(/"(?:code|status)"\s*:\s*(\d{3})/.exec(text)?.[1])
  const detail=String(metadata?.raw ?? field('raw') ?? '')
  if (code===429 && (detail.includes('rate-limited upstream') || metadata?.provider_error_code==='engine_overloaded' || field('provider_error_code')==='engine_overloaded')) {
    return `${String(provider ?? 'The upstream provider').slice(0,80)} is temporarily rate-limiting this model. Retry shortly, or choose another provider or model. This is a provider capacity limit, not a problem with your Mac.`
  }
  const message=typeof error==='string' ? error : error?.message ?? field('message')
  return typeof message==='string' && !message.trim().startsWith('{')
    ? message.slice(0,600) : 'The provider returned incomplete error information. Retry shortly or choose another model.'
}
