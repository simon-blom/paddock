import SwiftUI

/// AppKit owns the shortcut even with the sidebar folded or WebKit focused.
/// The shared ChatView intentionally delegates application shortcuts to us.
public struct StudioHistoryCommands: Commands {
  @Bindable private var model: WorkspaceModel
  public init(model: WorkspaceModel) { self.model = model }
  public var body: some Commands {
    CommandGroup(after: .textEditing) {
      Button("Search Chats") { model.searchConversations() }
        .keyboardShortcut("k", modifiers: .command)
        .disabled(model.navigation.mode != .studio || model.desktopTransition || model.quitting)
    }
  }
}
