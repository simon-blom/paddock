import PaddockClient
import SwiftUI

struct DownloadsView: View {
  @Bindable var downloads: DownloadsModel
  let canStart: Bool
  let onStart: (String, String) -> Void

  var body: some View {
    VStack(spacing: 0) {
      VStack(alignment: .leading, spacing: 20) {
        PageHeading(title: "Downloads") {}
        if let error = downloads.error {
          HStack(alignment: .top) {
            Label(error, systemImage: "exclamationmark.circle").textSelection(.enabled)
            Spacer()
            Button("Try again") { Task { await downloads.refresh() } }
              .buttonStyle(FlatButtonStyle()).disabled(downloads.busy)
          }.font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
        }
      }.padding(32)
      if downloads.visible.isEmpty {
        Spacer(minLength: 0)
      } else {
        PaddockScrollView {
          LazyVStack(spacing: 0) {
            ForEach(downloads.visible) { job in
              row(job)
              WorkspaceRule()
            }
          }.padding(.horizontal, 32).padding(.bottom, 32)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity)
      .overlay {
        if downloads.visible.isEmpty {
          if !downloads.loaded && downloads.error == nil {
            ProgressView("Loading downloads…")
              .frame(maxWidth: .infinity, maxHeight: .infinity).allowsHitTesting(false)
          } else {
            DownloadsEmptyState(unavailable: !downloads.loaded && downloads.error != nil)
              .allowsHitTesting(false)
          }
        }
      }
      .background(PaddockStyle.canvas).task { await downloads.refresh() }
      .accessibilityIdentifier("downloads-list")
  }

  private func row(_ job: ModelDownload) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(alignment: .top, spacing: 16) {
        VStack(alignment: .leading, spacing: 6) {
          Text(verbatim: job.display).font(.system(size: 14, weight: .medium))
          Text(job.title).font(.system(size: 12)).foregroundStyle(.secondary)
        }
        Spacer()
        if job.active {
          Button("Pause", systemImage: "pause") {
            Task { await downloads.perform(.pause(id: job.id)) }
          }
          .buttonStyle(FlatButtonStyle()).disabled(downloads.busy || job.cancelling == true)
          .accessibilityIdentifier("download-pause-\(job.id)")
        } else if job.resumable {
          Button(job.status.state == "error" ? "Retry" : "Resume", systemImage: "arrow.clockwise") {
            Task { await downloads.perform(.resume(id: job.id)) }
          }.buttonStyle(FlatButtonStyle()).disabled(downloads.busy)
            .accessibilityIdentifier("download-resume-\(job.id)")
        } else if job.complete, let artifact = job.weights {
          Button("Configure instance", systemImage: "plus") { onStart(job.model, artifact) }
            .buttonStyle(FlatButtonStyle(primary: true)).disabled(!canStart)
            .accessibilityIdentifier("download-start-\(job.id)")
        }
      }
      if !job.complete {
        ProgressView(value: job.progress).tint(PaddockStyle.accent)
          .accessibilityLabel("\(job.display) download progress")
        HStack {
          Text("\(DisplayFormat.bytes(job.downloaded)) of \(DisplayFormat.bytes(job.total))")
          Spacer()
          Text(job.progress, format: .percent.precision(.fractionLength(0)))
        }.font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
      }
      if let message = job.status.message {
        Text(verbatim: message).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
          .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
      } else if job.active, let phase = job.phase {
        if let transfer = downloads.transferStatus(for: job) {
          Text(transfer).font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
            .accessibilityIdentifier("download-speed-\(job.id)")
        }
        Text(verbatim: phase).font(.system(size: 11)).foregroundStyle(.secondary)
          .lineLimit(2).truncationMode(.middle).help(phase)
      }
    }.padding(.vertical, 20).accessibilityElement(children: .contain)
  }
}

/// Center the visible icon/text group as a single unit. ContentUnavailableView
/// owns additional internal spacing; an explicit stack makes our geometry exact.
struct DownloadsEmptyState: View {
  var unavailable = false
  var body: some View {
    VStack(spacing: 14) {
      Image(systemName: unavailable ? "exclamationmark.circle" : "arrow.down.circle")
        .font(.system(size: 30, weight: .light)).foregroundStyle(.secondary)
      Text(unavailable ? "Downloads unavailable" : "No downloads yet")
        .font(.system(size: 18, weight: .medium)).foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }.padding(.horizontal, 32)
      .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
      .accessibilityIdentifier("downloads-empty")
  }
}
