import AppKit
import Foundation
import PaddockClient
import PaddockStudio

/// All OS entry points use this narrow routing vocabulary. No URLs, scripts,
/// credentials or arbitrary management commands can arrive via notifications.
public enum DesktopAction: Equatable, Sendable {
  case studio, manager, settings, newChat, startModel, startSpeechModel
  case endpoint(port: UInt16)
  case chat(port: UInt16)
  case conversation(String)
}

public struct DesktopRequest: Identifiable, Equatable {
  public let id = UUID()
  public let action: DesktopAction
  public init(_ action: DesktopAction) { self.action = action }
}

extension WorkspaceModel {
  public func isDesktopDestinationVisible(_ action: DesktopAction) -> Bool {
    guard NSApp.isActive, NSApp.keyWindow?.title == "Paddock" else { return false }
    switch action {
    case .settings: return navigation.mode == .manager
    case .manager, .endpoint:
      return navigation.mode == .manager && navigation.manager == .runners
    case .startModel:
      return navigation.mode == .manager && navigation.manager == .models
        && navigation.libraryPurpose == .all
    case .startSpeechModel:
      return navigation.mode == .manager && navigation.manager == .models
        && navigation.libraryPurpose == .speech
    case .conversation(let id):
      return navigation.mode == .studio && navigation.studio == .newChat
        && chat.conversation?.id == id
    case .chat: return navigation.mode == .manager && navigation.manager == .runners
    default: return navigation.mode == .studio && navigation.studio == .newChat
    }
  }
  public var desktopRows: [EndpointRow] {
    guard let snapshot else { return [] }
    return EndpointRow.rows(snapshot: snapshot, latestJob: latestJob)
  }

  public var managedRunners: [RunnerInfo] {
    // Only runners present in this core's supervised inventory. Never enumerate
    // or terminate arbitrary processes, cloud endpoints or foreign listeners.
    snapshot?.runners ?? []
  }

  var desktopNavigationBlocked: Bool {
    draft.hasContent || chat.busy || chat.uploading || chat.hasAttachments
      || chat.hasMessageEdit
      || chat.state?.unsavedEdits == true
      || chat.hasArtifactEdits
  }

  public func request(_ action: DesktopAction) {
    guard !desktopTransition else { return }
    desktopRequest = DesktopRequest(action)
  }

  func prepareDesktopChat() async throws {
    await chat.start()
    let deadline = ContinuousClock.now.advanced(by: .seconds(20))
    while !chat.ready || chat.restoring {
      try Task.checkCancellation()
      if let error = chat.error { throw ManagerError.core(error) }
      guard ContinuousClock.now < deadline else {
        throw ManagerError.core("Studio did not finish opening. Your draft has been kept.")
      }
      try await Task.sleep(for: .milliseconds(50))
    }
  }

  public func handleDesktopRequest(_ request: DesktopRequest) async {
    guard !desktopTransition else { return }
    switch request.action {
    case .settings:
      navigation.showSettings()
      return
    case .manager:
      navigation.showManager(.runners)
      return
    case .endpoint(let port):
      selectedEndpointPort = port
      navigation.showManager(.runners)
      return
    case .startModel:
      navigation.showModelLibrary(purpose: .all)
      return
    case .startSpeechModel:
      navigation.showModelLibrary(purpose: .speech)
      return
    case .studio:
      navigation.mode = .studio
      navigation.studio = .newChat
      return
    default: break
    }
    navigation.mode = .studio
    navigation.studio = .newChat
    if case .conversation(let id) = request.action, id == chat.conversation?.id { return }
    guard !desktopNavigationBlocked else {
      desktopError =
        "Your current work is still open. Send or clear the draft, finish uploads, and save artifact edits before opening another conversation. Nothing was discarded."
      return
    }
    desktopTransition = true
    defer { desktopTransition = false }
    do {
      try await prepareDesktopChat()
      try Task.checkCancellation()
      // Loading may have restored work; recheck before any destructive routing.
      guard !desktopNavigationBlocked else {
        throw ManagerError.core("Finish the current Studio work first. Nothing was discarded.")
      }
      switch request.action {
      case .conversation(let id): try await chat.command("open", ["id": .string(id)])
      case .newChat: try await chat.command("newChat")
      case .chat(let port):
        try await chat.command("refresh")
        guard
          let model = chat.state?.models.first(where: {
            $0.port == port && $0.chat && $0.status == "ok"
          })
        else {
          throw ManagerError.core("This model is not ready for chat. Check Settings > Instances.")
        }
        try await chat.command("newChat")
        try await chat.command("models", ["ids": .array([.string(model.id)])])
      default: break
      }
    } catch { desktopError = error.localizedDescription }
  }

  /// Snapshot the exact identities the person approved. A replacement on the
  /// same port is never stopped. Jobs remain owned by Rust even on cancellation.
  public func stopForQuit(_ runners: [RunnerInfo]) async -> Bool {
    guard !operationInProgress else { return false }
    for runner in runners {
      await refresh()
      guard state == .ready else {
        desktopError = "Could not refresh model status. Paddock stayed open."
        return false
      }
      guard let current = snapshot?.runners.first(where: { $0.port == runner.port }) else {
        continue
      }
      guard current.pid == runner.pid else {
        desktopError =
          "A model was replaced while quitting. Review the running models; the replacement was not stopped."
        return false
      }
      guard await submit(.stop(port: runner.port, pid: runner.pid), duringQuit: true),
        let id = lastReceipt?.id
      else {
        desktopError = commandError ?? "The stop request was not accepted."
        return false
      }
      let deadline = ContinuousClock.now.advanced(by: .seconds(90))
      while ContinuousClock.now < deadline {
        if let job = snapshot?.jobs?.first(where: { $0.id == id }), !job.isActive {
          guard job.state == "succeeded" else {
            desktopError = job.message
            return false
          }
          break
        }
        do { try await Task.sleep(for: .milliseconds(500)) } catch { return false }
        await refresh()
      }
      guard let job = snapshot?.jobs?.first(where: { $0.id == id }), job.state == "succeeded" else {
        desktopError = "Stopping took longer than expected. Paddock stayed open to monitor it."
        return false
      }
    }
    await refresh()
    guard state == .ready, managedRunners.isEmpty else {
      desktopError = "There are still running models. Paddock stayed open; review their status."
      return false
    }
    return true
  }
}
