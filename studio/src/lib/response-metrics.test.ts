import { expect, it } from 'vitest'
import { measuredUsage, engineFields } from './response-metrics'
import type { Usage } from '@/types/chat'

const legacy: Usage = { promptTokens:2719, completionTokens:1386, reasoningTokens:99,
  ms:56898.261834, answerMs:3464.010292, reasoningMs:49009.056542, tps:400.11428 }
it('repairs the actual artifact-turn 400 tok/s regression without editing history',()=>{
  const u=measuredUsage(legacy)
  expect(u.tps).toBeCloseTo(26.09921555,6)
  expect(u.timingSource).toBe('end-to-end')
  expect(u.answerMs).toBeUndefined()
  expect(u.reasoningMs).toBeUndefined()
  expect(legacy.tps).toBe(400.11428)
})
it('uses all-round engine time, independent of final answer delivery',()=>{
  const t={source:'engine',version:1,queue_ms:10,prefill_ms:4000,decode_ms:50000}
  const u=measuredUsage({...legacy,...engineFields(t)})
  expect(u.tps).toBe(29.7)
  expect(u.timingSource).toBe('engine')
  expect(engineFields(t,legacy).timingSource).toBe('end-to-end')
  expect(engineFields(t,u).decodeMs).toBe(100000)
})
it('does not invent zero-time, missing or invalid throughput',()=>{
  for(const ms of [undefined,0,-1,NaN,Infinity]) expect(measuredUsage({promptTokens:0,completionTokens:10,ms}).tps).toBeUndefined()
  expect(engineFields({source:'engine',version:1,queue_ms:0,prefill_ms:0,decode_ms:NaN}).timingSource).toBe('end-to-end')
})
