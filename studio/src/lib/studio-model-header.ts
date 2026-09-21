// One presentation for AppHeader.vue and the native title bar. In particular,
// show the conversation's target, not the fleet's potentially different seat.
import { takesTurns, useModelsStore } from '@/stores/models'
import { useChatStore } from '@/stores/chat'
import { friendlyModelName, isVisionModel } from '@/lib/model-caps'
import { fleetLabel, fleetVendor, speculationBadge } from '@/lib/model-name'
import { effectiveModelId } from '@/lib/select-model'

export function studioModelHeader() {
  const models = useModelsStore(), chat = useChatStore()
  const currentModel = effectiveModelId()
  const pickerOptions = models.models
    .filter(m => takesTurns(m.kind) && m.status === 'ok')
    .map(m => ({
      value: m.id, label: m.display ?? friendlyModelName(m.id),
      hint: m.cloud ? m.cloud.endpointName : String(m.port),
      vendor: m.vendor ?? '', available: true,
      title: m.cloud ? `${m.id} · ${m.cloud.endpointName}` : `${m.id} · port ${m.port}`,
    }))
  // Stopped targets must remain visible. Choosing a reachable option fixes
  // the target; silently displaying the next available runner does not.
  if (currentModel && !pickerOptions.some(o => o.value === currentModel)) {
    const m = models.models.find(m => m.id === currentModel)
    pickerOptions.unshift({
      value: currentModel, label: fleetLabel(currentModel),
      hint: m?.cloud ? m.cloud.endpointName : 'not running',
      vendor: fleetVendor(currentModel) ?? '', available: false,
      title: `${currentModel} - not running`,
    })
  }
  const compareLanes = (chat.active?.compareModels ?? []).map(id => ({
    id, label: fleetLabel(id), vendor: fleetVendor(id) ?? '',
    spec: speculationBadge(models.models.find(m => m.id === id)?.spec),
  }))
  const encoder = models.models.some(m => takesTurns(m.kind))
    ? undefined : models.models.find(m => m.kind === 'encoder')
  return {
    currentModel, pickerOptions, compareLanes, comparing: compareLanes.length >= 2,
    specLabel: speculationBadge(currentModel ? models.specFor(currentModel) : undefined),
    isVision: !!currentModel && (models.models.find(m => m.id === currentModel)?.vision ?? isVisionModel(currentModel)),
    soleEncoder: encoder ? encoder.display ?? friendlyModelName(encoder.id) : null,
  }
}
