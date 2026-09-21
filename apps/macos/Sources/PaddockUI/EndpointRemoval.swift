import PaddockClient
import SwiftUI

/// Capture the exact configuration reviewed by the user, never just its name.
struct EndpointRemovalReview: Equatable {
  let port: UInt16
  let title: String
  let revision: String

  init?(_ row: EndpointRow) {
    guard row.runner == nil, row.configured?.running == false, row.job?.isActive != true,
      let revision = row.configured?.revision
    else { return nil }
    port = row.port
    title = row.title
    self.revision = revision
  }
}

extension WorkspaceModel {
  func canRemoveEndpoint(_ review: EndpointRemovalReview) -> Bool {
    guard canSubmit, let snapshot,
      !(endpointEditor?.endpoint.port == review.port && endpointEditor?.dirty == true)
    else { return false }
    return EndpointRow.rows(snapshot: snapshot, latestJob: latestJob)
      .first { $0.port == review.port }.flatMap(EndpointRemovalReview.init) == review
  }

  @discardableResult
  func removeEndpoint(_ review: EndpointRemovalReview) async -> Bool {
    guard canRemoveEndpoint(review) else {
      desktopError =
        "The configuration changed, is in use, or has unsaved edits. Review it before removing."
      return false
    }
    // The core also checks the revision and process state atomically. Its job
    // receipt/error uses the same app-wide feedback as Start and Stop.
    return await submit(.remove(port: review.port, revision: review.revision))
  }
}

private struct EndpointRemovalConfirmation: ViewModifier {
  @Bindable var workspace: WorkspaceModel
  @Binding var review: EndpointRemovalReview?

  func body(content: Content) -> some View {
    content.confirmationDialog(
      "Remove \(review?.title ?? "model") configuration?",
      isPresented: Binding(get: { review != nil }, set: { if !$0 { review = nil } }),
      titleVisibility: .visible, presenting: review
    ) { target in
      Button("Remove configuration", role: .destructive) {
        review = nil
        Task { await workspace.removeEndpoint(target) }
      }.disabled(!workspace.canRemoveEndpoint(target))
        .accessibilityIdentifier("confirm-remove-endpoint-\(target.port)")
      Button("Cancel", role: .cancel) { review = nil }
    } message: { target in
      Text(
        "Port \(String(target.port)). Deletes its settings and endpoint API key. Model files and chats are kept. This cannot be undone."
      )
    }
  }
}

extension View {
  func endpointRemovalConfirmation(
    workspace: WorkspaceModel, review: Binding<EndpointRemovalReview?>
  ) -> some View {
    modifier(EndpointRemovalConfirmation(workspace: workspace, review: review))
  }
}
