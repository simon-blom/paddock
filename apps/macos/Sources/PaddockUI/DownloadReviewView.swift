import PaddockClient
import SwiftUI

struct DownloadReviewView: View {
  let downloads: DownloadsModel
  let model: String
  let artifact: String
  let onAccepted: () -> Void
  @Environment(\.dismiss) private var dismiss
  @State private var plan: DownloadPlan?
  @State private var error: String?
  @State private var submitting = false

  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      Text("Download model").font(.system(size: 21, weight: .semibold))
      if let plan {
        Text(verbatim: plan.display).font(.system(size: 14, weight: .medium))
        VStack(spacing: 12) {
          ForEach(plan.pieces) { piece in
            HStack(alignment: .top) {
              Text(verbatim: piece.displayLabel).frame(maxWidth: .infinity, alignment: .leading)
              Text(DisplayFormat.bytes(piece.size)).monospacedDigit()
            }.font(.system(size: 12))
          }
        }.padding(16).background(
          PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
        Text(
          "\(plan.fileCount) files · \(DisplayFormat.bytes(plan.total)) total · \(DisplayFormat.bytes(plan.free)) free"
        )
        .font(.system(size: 12)).monospacedDigit().foregroundStyle(.secondary)
        if !plan.fits {
          Label(
            plan.free == nil
              ? "Available disk space could not be checked."
              : "Not enough free disk space, including 1 GiB of headroom.",
            systemImage: "exclamationmark.circle"
          )
          .font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
        }
      } else if error == nil {
        ProgressView("Checking files and storage…").frame(height: 100)
      }
      if let error {
        Text(error).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution).textSelection(
          .enabled)
      }
      HStack {
        Button("Cancel") { dismiss() }.buttonStyle(FlatButtonStyle()).disabled(submitting)
          .keyboardShortcut(.cancelAction)
        Spacer()
        Button(plan.map { "Download up to \(DisplayFormat.bytes($0.remaining))" } ?? "Download") {
          guard let plan else { return }
          submitting = true
          Task {
            if await downloads.perform(.pull(plan)) {
              dismiss()
              onAccepted()
            } else {
              error = downloads.error ?? "The download was not accepted. Try again."
            }
            submitting = false
          }
        }.buttonStyle(FlatButtonStyle(primary: true))
          .disabled(plan?.fits != true || submitting || downloads.busy)
          .accessibilityIdentifier("confirm-model-download")
      }
    }.padding(28).frame(width: 480).background(PaddockStyle.canvas)
      .interactiveDismissDisabled(submitting).presentationBackground(PaddockStyle.canvas)
      .task {
        do { plan = try await downloads.plan(model: model, artifact: artifact) } catch {
          self.error = error.localizedDescription
        }
      }
  }
}
