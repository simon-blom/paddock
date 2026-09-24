import { describe, expect, it } from 'vitest'
import settingsSource from '../components/settings/SettingsPanel.vue?raw'
import sidebarSource from '../components/chat/ConversationSidebar.vue?raw'
import transcriptSource from '../components/chat/TranscriptView.vue?raw'
import nativeSettings from '../../../apps/macos/Sources/PaddockUI/StudioPreferencesView.swift?raw'
import nativeOverview from '../../../apps/macos/Sources/PaddockUI/OverviewView.swift?raw'
import nativeToolbar from '../../../apps/macos/Sources/PaddockUI/WorkspaceToolbar.swift?raw'
import nativeCloudCatalog from '../../../apps/macos/Sources/PaddockUI/CloudCatalogView.swift?raw'
import nativeCloudDetail from '../../../apps/macos/Sources/PaddockUI/CloudModelDetailView.swift?raw'
import nativeConnections from '../../../apps/macos/Sources/PaddockUI/ConnectionsView.swift?raw'
import nativePresets from '../../../apps/macos/Sources/PaddockUI/StudioLibraryView.swift?raw'
import nativeFooter from '../../../apps/macos/Sources/PaddockUI/StudioSidebarFooter.swift?raw'
import nativeDetail from '../../../apps/macos/Sources/PaddockUI/EndpointDetailView.swift?raw'
import nativeEdit from '../../../apps/macos/Sources/PaddockUI/EndpointEditView.swift?raw'
import fixture from './studio-settings-layout.fixture.json'
import { STUDIO_SETTINGS_SECTIONS, studioSettingsLayout } from './studio-settings-layout'

describe('web/native information placement contract', () => {
  it('does not put manual refresh controls in native headers or lists', () => {
    for (const source of [nativeToolbar, nativeCloudCatalog, nativeCloudDetail, nativeConnections, nativePresets]) {
      expect(source).not.toContain('arrow.clockwise')
      expect(source).not.toMatch(/Button\("Refresh/)
    }
    expect(nativeToolbar).toContain('.accessibilityLabel("GPU metrics")')
  })
  it('uses the same ordered sections as the actual web settings template', () => {
    const headings = [...settingsSource.matchAll(/<h2>([^<]+)<\/h2>/g)].map(m => m[1])
    expect(headings).toEqual(STUDIO_SETTINGS_SECTIONS.map(s => s.title))
    expect(settingsSource).not.toMatch(/v-model="settings\.(autoTitle|markUnsure)"/)
  })
  it('keeps the native decoding fixture identical to the shared presentation', () => {
    expect(studioSettingsLayout()).toEqual(fixture)
  })
  it('uses a model-independent exact editor, never a rounded slider label', () => {
    expect(studioSettingsLayout().replyLimit.maximum).toBe(1048576)
    expect(settingsSource).toContain('ReplyLimitControl v-model="settings.maxTokens"')
    expect(nativeSettings).toContain('StudioReplyLimitControl(draft: $model.reply)')
    expect(nativeSettings).not.toContain('replyIndex')
    expect(settingsSource).not.toContain('replyLengthStops')
  })
  it('keeps This Mac free of qualification badges and redundant platform facts', () => {
    expect(nativeOverview).not.toContain('readiness.title')
    expect(nativeOverview).not.toContain('FactRow(title: "Backend"')
    expect(nativeOverview).not.toContain('FactRow(title: "Operating system"')
  })
  it('keeps contextual switches on the web surfaces the native client must follow', () => {
    expect(sidebarSource).toContain('@select="settings.autoTitle = !settings.autoTitle"')
    expect(transcriptSource).toContain('@click="settings.markUnsure = !settings.markUnsure"')
    expect(nativeSettings).not.toMatch(/\$model\.(autoTitle|markUnsure)/)
    expect(nativeSettings).toContain('ForEach(layout.sections)')
  })
  it('keeps native appearance in the sidebar footer and editing off the endpoint detail page', () => {
    expect(nativeOverview).not.toContain('"Appearance"')
    expect(nativeFooter).toContain('Picker("Appearance"')
    expect(nativeToolbar).not.toContain('Picker("Appearance"')
    const detailView = nativeDetail.split('struct EndpointSettingsView')[0]
    expect(detailView).not.toContain('EndpointSettingsView(')
    expect(detailView).toContain('EndpointLogsView(')
    expect(detailView).toContain('EndpointSummaryView(')
    expect(nativeEdit).toContain('EndpointSettingsView(')
  })
})
