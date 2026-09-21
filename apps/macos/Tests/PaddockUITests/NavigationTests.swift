import Testing

@testable import PaddockUI

@Suite("Studio and Manager navigation")
@MainActor struct NavigationTests {
  @Test func firstLaunchIsStudioWithNoSidebar() {
    let navigation = WorkspaceNavigation()
    #expect(navigation.mode == .studio)
    #expect(navigation.studio == .newChat)
    #expect(!navigation.sidebarVisible)
    #expect(!StudioDraft().hasContent)
  }

  @Test func modeSwitchingKeepsIndependentSidebarAndDestinationState() {
    var navigation = WorkspaceNavigation()
    navigation.studio = .newChat
    navigation.mode = .manager
    #expect(navigation.sidebarVisible)
    #expect(navigation.manager == .runners)
    navigation.manager = .downloads
    navigation.sidebarVisible = false
    navigation.mode = .studio
    #expect(!navigation.sidebarVisible)
    #expect(navigation.studio == .newChat)
    navigation.sidebarVisible = true
    navigation.mode = .manager
    #expect(!navigation.sidebarVisible)
    #expect(navigation.manager == .downloads)
    navigation.mode = .studio
    #expect(navigation.sidebarVisible)
  }

  @Test func acceptedStartRoutesToManagerWithoutOverwritingStudioDestination() {
    var navigation = WorkspaceNavigation()
    navigation.studio = .chats
    navigation.showManager(.runners)
    #expect(navigation.mode == .manager)
    #expect(navigation.manager == .runners)
    #expect(navigation.sidebarVisible)
    navigation.mode = .studio
    #expect(navigation.studio == .chats)
    #expect(!navigation.sidebarVisible)
  }

  @Test func draftContentRequiresExplicitDiscardIncludingWhitespace() {
    var draft = StudioDraft()
    draft.message = "\n "
    #expect(draft.hasContent)
    draft.message = "An unfinished question"
    #expect(draft.hasContent)
    draft = StudioDraft()
    #expect(!draft.hasContent)
  }

  @Test func sidebarStaysPresentForItsFooterDestinations() {
    var navigation = WorkspaceNavigation()
    navigation.sidebarVisible = true
    for destination in StudioDestination.allCases {
      navigation.studio = destination
      #expect(navigation.showsSidebar)
    }
  }

  @Test func footerNavigationKeepsHistoryAndItsFoldState() {
    var navigation = WorkspaceNavigation()
    navigation.showStudioChats()
    #expect(navigation.showsSidebar)
    navigation.studio = .prompts
    #expect(navigation.showsSidebar)
    navigation.studio = .settings
    #expect(navigation.showsSidebar)
    navigation.studio = .chats
    #expect(navigation.showsSidebar)
    navigation.toggleSidebar()
    navigation.studio = .prompts
    navigation.studio = .chats
    #expect(!navigation.showsSidebar)
  }

  @Test func togglingSidebarDoesNotReplaceSettingsOrTheCurrentThread() {
    var navigation = WorkspaceNavigation()
    navigation.studio = .settings
    navigation.toggleSidebar()
    #expect(navigation.studio == .settings && navigation.showsSidebar)
    navigation.showManager(.connectors)
    #expect(navigation.showsSidebar && navigation.manager == .connectors)
    navigation.mode = .studio
    #expect(navigation.studio == .settings && navigation.showsSidebar)
  }

  @Test func studioSettingsStayInStudioAndSurviveAManagerSwitch() {
    var navigation = WorkspaceNavigation()
    navigation.studio = .settings
    #expect(navigation.mode == .studio)
    navigation.showManager(.runners)
    #expect(navigation.manager == .runners)
    navigation.mode = .studio
    #expect(navigation.studio == .settings)
  }

  @Test func instancesAndCatalogHaveDistinctMenuTitles() {
    let navigation = WorkspaceNavigation()
    #expect(navigation.manager == .runners)
    #expect(ManagerDestination.runners.rawValue == "Instances")
    #expect(ManagerDestination.models.rawValue == "Catalog")
  }

  @Test func visibleHistoryFitsInsideTheNativeMinimumWindowWidth() {
    var navigation = WorkspaceNavigation()
    #expect(WorkspaceView.minimumWidth(for: navigation) == 900)
    navigation.sidebarVisible = true
    #expect(WorkspaceView.minimumWidth(for: navigation) == 900)
    navigation.studio = .settings
    #expect(WorkspaceView.minimumWidth(for: navigation) == 900)
    navigation.showManager(.runners)
    #expect(WorkspaceView.minimumWidth(for: navigation) == 900)
  }
}
