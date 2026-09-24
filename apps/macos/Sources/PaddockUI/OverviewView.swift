import AppKit
import PaddockClient
import SwiftUI

/// A quiet settings page. No fabricated memory gauges or performance numbers.
struct OverviewView: View {
  let snapshot: ManagerSnapshot
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 28) {
        PageHeading(title: "This Mac") {
          EmptyView()
        }
        HStack(spacing: 16) {
          Image(systemName: "laptopcomputer").font(.system(size: 32, weight: .light))
            .foregroundStyle(.secondary).frame(width: 46)
          VStack(alignment: .leading, spacing: 6) {
            Text(snapshot.readiness.card ?? "Apple silicon")
              .font(.system(size: 17, weight: .medium))
          }
          Spacer()
          if let warning = snapshot.readiness.warning {
            StatusBadge(title: warning, tone: .caution)
          }
        }.padding(.vertical, 8)
        WorkspaceRule()
        VStack(alignment: .leading, spacing: 20) {
          Text("Storage").font(.system(size: 14, weight: .semibold))
          FactRow(
            title: "Downloaded models",
            value: String(
              snapshot.catalog.models.filter {
                $0.hasInstalledWeights(on: snapshot.readiness.backend)
              }.count))
          FactRow(
            title: "Available space",
            value: DisplayFormat.bytes(snapshot.identity.registry?.diskFree))
          FactRow(
            title: "Volume capacity",
            value: DisplayFormat.bytes(snapshot.identity.registry?.diskTotal))
          HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
              Text("Model folder")
              Text(snapshot.identity.registry?.modelsDir ?? "Location unavailable")
                .font(.system(size: 12, design: .monospaced)).foregroundStyle(.secondary)
                .textSelection(.enabled).lineLimit(3).truncationMode(.middle)
            }
            Spacer()
            if let path = snapshot.identity.registry?.modelsDir {
              Button("Show in Finder", systemImage: "folder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
              }.buttonStyle(FlatButtonStyle())
            }
          }
        }.font(.system(size: 13))
        WorkspaceRule()
        VStack(alignment: .leading, spacing: 20) {
          Text("Application").font(.system(size: 14, weight: .semibold))
          FactRow(title: "Paddock version", value: snapshot.identity.version)
          FactRow(title: "Running endpoints", value: String(snapshot.runners.count))
        }.font(.system(size: 13))
      }.padding(.horizontal, 32).padding(.top, 20).padding(.bottom, 32)
        .frame(maxWidth: 820, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
    }.background(PaddockStyle.canvas)
  }
}
