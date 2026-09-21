import SwiftUI

/// Manual developer fixtures. The shipping renderer cannot be switched to web.
public struct StudioRendererCommands: Commands {
  @Bindable private var model: WorkspaceModel

  public init(model: WorkspaceModel) { self.model = model }

  public var body: some Commands {
    #if DEBUG
      CommandMenu("Renderer") {
        Button("Markdown Samples…") { model.showsRendererSamples = true }
          .disabled(model.desktopTransition || model.quitting)
      }
    #endif
  }
}
