import { describe, expect, it } from 'vitest'
import web from '../components/manage/ServerForm.vue?raw'
import native from '../../../apps/macos/Sources/PaddockUI/EndpointSettingsView.swift?raw'
import workload from '../../../apps/macos/Sources/PaddockUI/EndpointModelWorkload.swift?raw'
import editor from '../../../apps/macos/Sources/PaddockUI/EndpointEditorCatalog.swift?raw'
import draft from '../../../apps/macos/Sources/PaddockUI/EndpointEditor.swift?raw'
import start from '../../../apps/macos/Sources/PaddockUI/StartModelView.swift?raw'
import shell from '../../../apps/macos/Sources/PaddockUI/WorkspaceView.swift?raw'
import advanced from '../../../apps/macos/Sources/PaddockUI/EndpointAdvancedSettings.swift?raw'
import offload from '../../../apps/macos/Sources/PaddockUI/EndpointKVOffload.swift?raw'
import residency from '../../../apps/macos/Sources/PaddockUI/EndpointResidencySettings.swift?raw'
import memory from '../../../apps/macos/Sources/PaddockUI/EndpointMemorySettings.swift?raw'
import search from '../../../apps/macos/Sources/PaddockUI/WebSearchSettingsView.swift?raw'
import mcp from '../../../apps/macos/Sources/PaddockUI/EndpointMCPSection.swift?raw'
import runnerPolicy from '../../../crates/paddock-engine/src/spec_policy.rs?raw'

describe('native endpoint Simple form follows the web reference', () => {
  it('elects native ports automatically and puts manual addresses under Advanced', () => {
    expect(draft).toContain('automaticPort = true')
    expect(draft).toContain('port: automaticPort ? nil : UInt16(newPort)')
    expect(start).toMatch(/EndpointSettingsView\(\s*editor:\s*editor/)
    expect(native).toContain('access(advanced: true)')
    expect(native.indexOf('if advanced {')).toBeLessThan(native.indexOf('EndpointFormField("Port")'))
    expect(shell).not.toContain('Port \\(job.port)')
  })
  it('keeps common sections in web order with native Metal memory and KV controls', () => {
    const simple = web.slice(web.indexOf('<template v-if="mode === \'simple\'">'), web.indexOf('<template v-else-if="mode === \'advanced\'">'))
    const headings = [...simple.matchAll(/<p class="sf__card-hd">([^<]+)<\/p>/g)].map(m => m[1]!.replace(/&amp;/g, '&'))
    const nativeSimple = native.slice(native.indexOf('private var simple:'))
      .replace('EndpointKVOffloadSettings(editor: editor)', offload)
      .replace('EndpointResidencySettings(editor: editor)', residency)
    const nativeHeadings = [...nativeSimple.matchAll(/EndpointFormCard\("([^"]+)"\)/g)].map(m => m[1])
    expect(nativeHeadings).toEqual(headings)
    expect(advanced).toContain('editor.visibleRuntimeOptions.filter')
    expect(memory).toContain('EndpointFormCard("Memory budget")')
    expect(memory).toContain('recommendedMaxWorkingSetSize')
    expect(memory).not.toContain('memoryLimit = "32"')
    expect(memory).toContain('not currently free memory')
    expect(native).not.toContain('EndpointConfigurationSummary')
  })
  it('offers the same per-runner residency choices without promising unsupported families', () => {
    expect(web).toContain("catModel.value?.family === 'whisper'")
    expect(web).toContain('v-if="canResidency"')
    expect(residency).toContain('if editor.residencySupported')
    for (const label of ['Load model', 'At runner startup', 'On first request', 'Unload after inactivity']) {
      expect(web).toContain(label)
      expect(residency).toContain(label)
    }
    expect(web).toContain("k === 'residency' && residencyLive.value")
    expect(native).toContain('editor.runtimeState?.residencyLive == true')
  })
  it('keeps tools grouped with intrinsic provider pills and readable connector identities', () => {
    expect(search).toContain('EndpointFormCard("Web search")')
    expect(search).toContain('SettingsChoiceLayout')
    expect(search).not.toContain('LazyVGrid')
    expect(search).not.toContain('Add connectors in Studio')
    expect(mcp).toContain('EndpointFormCard("MCP servers")')
    expect(mcp).toContain('.toggleStyle(.switch)')
    expect(mcp).toContain('.truncationMode(.middle).help(row.url)')
    expect(mcp).toContain('row.system || model.saving')
    expect(web).toContain('On for every model'.toLowerCase())
  })
  it('uses the same workload and speculation values', () => {
    for (const [label, batch] of [['Just me', 1], ['Coding agents', 4], ['A team / an app', 16]]) {
      expect(web).toContain(`label: '${label}', batch: ${batch}`)
      expect(editor).toContain(`("${label}", ${batch})`)
    }
    for (const [value, label] of [['on', 'On'], ['off', 'Off'], ['adaptive', 'Adaptive']]) {
      expect(web).toContain(`value: '${value}', label: '${label}'`)
      expect(editor).toContain(`("${value}", "${label}")`)
    }
    expect(workload).toContain('Conversation memory')
    expect(workload).toContain('Context per conversation')
    expect(runnerPolicy).toContain('"adaptive" | "auto" => Ok(SpecPolicy::Auto)')
    expect(editor).toContain('case "auto", "adaptive": "adaptive"')
    expect(editor).toContain('"ladder", "legacy": "on"')
  })
  it('has one Save action and asks when a running model would restart', () => {
    expect(native).toContain('Button(editor.isCreating ? "Start Model" : (editor.saving ? "Saving…" : "Save"))')
    expect(native).toMatch(/if editor\.pid != nil \{\s*confirmation = \.restart\s*\}/)
    expect(native).toContain('Button("Save for later")')
    expect(native).toContain('Button(editor.dirty ? "Save & restart" : "Restart to apply")')
    expect(native).not.toContain('Button("Save for next start")')
    expect(native).not.toContain('Saved keys stay private in Rust')
    expect(native).toMatch(/SecureField\(\s*editor\.endpoint\.settings\?\.hasApiKey == true \? "\*{6}"/)
  })
})
