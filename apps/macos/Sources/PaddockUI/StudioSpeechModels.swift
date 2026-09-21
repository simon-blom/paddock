import Foundation
import Observation
import PaddockClient
import PaddockStudio

/// Native lifecycle transport for the shared SpeechModels menu. WebContent
/// can read speech inventory, but never receives privileged start/stop routes.
@MainActor @Observable
final class StudioSpeechModels {
  private(set) var port: UInt16?
  private(set) var pending: ManagementJob?
  private(set) var checking = false
  private(set) var error: String?
  private(set) var submitting = false
  var busy: Bool { submitting || pending != nil }
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored var canAct: () -> Bool = { true }
  @ObservationIgnored var onChange: (() async -> Void)?
  @ObservationIgnored var onNetworkReview: ((ConfiguredEndpoint) -> Void)?

  init(client: any ManagerLoading) { self.client = client }

  func act(_ shown: StudioState.Audio.SpeechModel, start: Bool) {
    guard !busy, canAct(), start ? shown.canStart : shown.canStop else { return }
    submitting = true
    port = shown.port
    error = nil
    task = Task {
      do {
        let snapshot = try await client.snapshot()
        guard let endpoint = snapshot.servers?.first(where: { $0.port == shown.port }),
          endpoint.model == shown.model, endpoint.capability?.contains("transcription") == true,
          snapshot.jobs?.contains(where: { $0.isActive }) != true
        else {
          throw ManagerError.core("Speech endpoint changed. Reopen the menu to refresh its status.")
        }
        let runner = snapshot.runners.first { $0.port == shown.port }
        let command: ModelCommand
        if start {
          guard runner == nil, !endpoint.running, let revision = endpoint.revision else {
            throw ManagerError.core(
              "This speech model is already running or starting. Refresh its status.")
          }
          guard endpoint.localOnly == true else {
            onNetworkReview?(endpoint)
            throw ManagerError.core(
              "Review this model's network access in Settings > Instances before starting it.")
          }
          command = .start(port: endpoint.port, revision: revision)
        } else {
          guard let runner else {
            throw ManagerError.core("This speech model is no longer running.")
          }
          command = .stop(port: endpoint.port, pid: runner.pid)
        }
        pending = try await client.submit(command)
        submitting = false
        await poll()
      } catch {
        submitting = false
        self.error = error.localizedDescription
        if pending == nil { port = nil }
        await onChange?()
      }
    }
  }

  func retryStatus() {
    guard pending != nil, !checking else { return }
    error = nil
    task = Task { await poll() }
  }

  private func poll() async {
    guard let receipt = pending else { return }
    checking = true
    defer { checking = false }
    do {
      guard receipt.port == port else { throw ManagerError.core("Unexpected operation receipt") }
      var job = receipt
      while job.isActive {
        try await Task.sleep(for: .milliseconds(350))
        job = try await client.submit(.poll(id: receipt.id))
        guard job.id == receipt.id, job.port == receipt.port, job.action == receipt.action else {
          throw ManagerError.core("Unexpected operation receipt")
        }
        pending = job
      }
      error = job.state == "succeeded" ? nil : job.message
      pending = nil
      port = nil
      await onChange?()
    } catch {
      self.error =
        "The operation was accepted, but its status could not be read. Retry status; do not start or stop it again."
    }
  }

  func settle() async { await task?.value }
}
