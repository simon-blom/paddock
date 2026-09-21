import { expect,it } from 'vitest'
import { providerErrorMessage } from './provider-error'
it('repairs legacy truncated provider JSON without routing or account data',()=>{
  const raw='provider error: '+JSON.stringify({error:{message:'Provider returned error',code:429,metadata:{
    raw:'moonshotai/kimi-k3 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations',
    provider_name:'DeepInfra',is_byok:false,provider_error_code:'engine_overloaded',limit_source:'upstream_provider_shared_pool',
    private:'PRIVATE_ACCOUNT'}}}).slice(0,400)
  const text=providerErrorMessage(raw)
  expect(text).toContain('DeepInfra')
  expect(text).toContain('Retry shortly')
  expect(text).not.toContain('{')
  expect(text).not.toContain('PRIVATE_')
})
it('keeps plain errors and never displays incomplete JSON',()=>{
  expect(providerErrorMessage('Connection interrupted')).toBe('Connection interrupted')
  expect(providerErrorMessage('provider error: {"error":{"message":"half')).not.toContain('{')
})
