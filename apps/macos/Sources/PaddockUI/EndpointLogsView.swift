import SwiftUI

struct EndpointLogsView: View {
  @Bindable var model: EndpointLogsModel
  let port: UInt16
  @Environment(\.scenePhase) private var phase
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Logs").font(.system(size: 16, weight: .semibold))
        Spacer()
        Text(status).font(.system(size: 11)).foregroundStyle(.secondary)
      }
      HStack(spacing: 10) {
        Dropdown(title: "Minimum log level", value: levelName(model.minimum), fillsWidth: true) {
          ForEach(["all", "INFO", "WARN", "ERROR"], id: \.self) { value in
            Button(levelName(value)) { model.minimum = value }
          }
        }.frame(width: 110)
        TextField("Filter lines…", text: $model.query)
          .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel("Filter log lines")
        Text("\(model.visible.count) of \(model.buffer.lines.count)")
          .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
      }
      ZStack(alignment: .bottomTrailing) {
        EndpointLogText(lines: model.visible, jump: model.jump, following: $model.following)
          .frame(height: 320)
        if model.visible.isEmpty {
          Text(
            model.buffer.lines.isEmpty
              ? "Waiting for log lines in this app's data folder…" : "Nothing matches the filter."
          )
          .font(.system(size: 12)).foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        if !model.following {
          Button("Latest", systemImage: "arrow.down") { model.latest() }
            .buttonStyle(FlatButtonStyle()).padding(12)
        }
      }
      .background(
        PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.card).stroke(
          PaddockStyle.border, lineWidth: 1))
      if let error = model.error {
        Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
          .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
      }
    }
    .task(id: "\(port)-\(phase == .active)") {
      if phase == .active { await model.follow(port: port) }
    }
  }
  private var status: String {
    switch model.state {
    case "live": "Live"
    case "waiting": "No local log yet"
    case "paused": "Paused"
    default: "Connecting…"
    }
  }
  private func levelName(_ value: String) -> String {
    switch value {
    case "INFO": "Info+"
    case "WARN": "Warn+"
    case "ERROR": "Errors"
    default: "All"
    }
  }
}
