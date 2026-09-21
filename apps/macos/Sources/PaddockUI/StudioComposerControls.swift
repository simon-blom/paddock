import PaddockStudio
import SwiftUI

struct StudioReasoningControls: View {
  @Bindable var chat: StudioWorkspace
  private var params: [String: StudioValue] { chat.state?.settings["params"]?.object ?? [:] }
  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      StudioPopoverHeading(title: "Thinking").padding(.horizontal, 9).padding(.vertical, 8)
      StudioReasoningNotice(composer: chat.state?.composer)
        .padding(.horizontal, 9)
      ForEach(chat.state?.composer?.reasoning ?? []) { choice in
        StudioPopoverChoice(
          title: choice.label, selected: chat.state?.composer?.reasoningChoice == choice.value
        ) {
          var p: [String: StudioValue] = ["thinking": .bool(choice.value != "off")]
          if choice.value != "off" && choice.value != "on" {
            p["reasoningEffort"] = .string(choice.value)
          }
          update(p)
        }
      }
      if chat.state?.composer?.thinkingBudget == true {
        WorkspaceRule().padding(.vertical, 8)
        let budget = params["thinkingBudget"]?.number ?? 0
        Dropdown(title: "Thinking budget", value: budget > 0 ? "\(Int(budget)) tokens" : "No limit")
        {
          Button("No limit") { update(["thinkingBudget": .number(0)]) }
          ForEach([1024, 2048, 4096, 8192, 16384], id: \.self) { n in
            Button("\(n) tokens") { update(["thinkingBudget": .number(Double(n))]) }
          }
        }
      }
      if chat.state?.composer?.preserveThinking == true {
        Toggle(
          "Keep earlier reasoning",
          isOn: Binding(
            get: { params["preserveThinking"]?.boolean ?? false },
            set: { update(["preserveThinking": .bool($0)]) })
        )
        .help("Uses additional context on supported models.")
      }
      ComposerError(chat: chat)
    }.font(.system(size: 12)).padding(9).frame(width: 250)
  }
  private func update(_ values: [String: StudioValue]) {
    Task { await chat.perform("settings", ["params": .object(values)]) }
  }
}

struct StudioReasoningNotice: View {
  let composer: StudioState.Composer?
  var body: some View {
    if let notice = composer?.reasoningNotice, !notice.isEmpty {
      Text(notice).font(.system(size: 11)).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true).padding(.bottom, 6)
        .accessibilityIdentifier("reasoning-controls-notice")
    }
  }
}

struct StudioSamplingControls: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.dismiss) private var dismiss
  @State private var values: [String: Double] = [:]
  @State private var changed = Set<String>()
  @State private var seed = ""
  @State private var originalSeed = ""
  @State private var saving = false
  private var seedValid: Bool { seed.isEmpty || Int(seed) != nil }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      StudioPopoverHeading(title: "Sampling")
      VStack(spacing: 12) {
        ForEach(chat.state?.composer?.sampling ?? []) { dial in
          StudioSamplingRow(
            dial: dial,
            value: Binding(
              get: { values[dial.key] ?? dial.value },
              set: {
                values[dial.key] = $0
                changed.insert(dial.key)
              }),
            display: changed.contains(dial.key)
              ? (values[dial.key] ?? dial.value).formatted(
                .number.precision(.fractionLength(0...2))) : dial.display)
        }
      }
      HStack {
        Text("Seed")
        Spacer()
        TextField("Random", text: $seed).frame(width: 120).textFieldStyle(StudioPopoverFieldStyle())
          .accessibilityLabel("Sampling seed")
      }
      if !seedValid { Text("Seed must be a whole number.").foregroundStyle(.secondary) }
      ForEach(chat.state?.composer?.samplingWarnings ?? [], id: \.self) { note in
        Text(note).font(.caption).foregroundStyle(.secondary)
      }
      ComposerError(chat: chat)
      HStack {
        Button("Model defaults") {
          Task {
            await chat.perform("samplerDefaults")
            if chat.error == nil { dismiss() }
          }
        }.buttonStyle(FlatButtonStyle()).help(
          chat.state?.composer?.samplingSource ?? "Model defaults")
        Spacer()
        Button("Apply") {
          var patch = Dictionary(
            uniqueKeysWithValues: changed.map {
              ($0, StudioValue.number(((values[$0] ?? 0) * 100).rounded() / 100))
            })
          if seed != originalSeed { patch["seed"] = Double(seed).map(StudioValue.number) ?? .null }
          saving = true
          Task {
            await chat.perform("settings", ["params": .object(patch)])
            saving = false
            if chat.error == nil { dismiss() }
          }
        }.buttonStyle(FlatButtonStyle(primary: true)).disabled(saving || !seedValid)
      }
    }.font(.system(size: 12)).padding(16).frame(width: 320)
      .task {
        seed = chat.state?.settings["params"]?.object?["seed"]?.number.map { String(Int($0)) } ?? ""
        originalSeed = seed
      }
  }
}

struct StudioContextControls: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.dismiss) private var dismiss
  @State private var limit: Int?
  @State private var summarize = true
  @State private var mode = ""
  @State private var language = ""
  @State private var expectedPreferences: [String: StudioValue] = [:]
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      StudioPopoverHeading(title: "Context and reading")
      if let p = chat.state?.composer, let cap = chat.state?.capabilities {
        Text("≈ \(p.contextUsed.formatted()) / \(cap.context.formatted()) tokens")
          .monospacedDigit()
          .help(
            "Estimate for conversation and draft; pending attachments and tool schemas are not included."
          )
        if p.cost > 0 {
          Text(
            "Reported spend: \(p.cost.formatted(.currency(code: "USD").precision(.fractionLength(2...4))))"
          )
        }
        Divider()
        TextField("Reply limit · model maximum", value: $limit, format: .number)
          .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel("Reply token limit")
          .help("Leave empty for the model maximum. Thinking and answer share the reply budget.")
        Toggle("Summarize older context when needed", isOn: $summarize)
        if !cap.ocrModes.isEmpty {
          Dropdown(title: "Document reading", value: mode.isEmpty ? "Automatic reading mode" : mode)
          {
            Button("Automatic") { mode = "" }
            ForEach(cap.ocrModes, id: \.self) { value in Button(value) { mode = value } }
          }
        }
        if p.audioMode {
          TextField("Audio language · automatic", text: $language).textFieldStyle(
            StudioPopoverFieldStyle())
        }
      }
      ComposerError(chat: chat)
      HStack {
        Spacer()
        Button("Apply") {
          Task {
            let changes: [String: StudioValue] = [
              "maxTokens": limit.map { .number(Double($0)) } ?? .null,
              "summarize": .bool(summarize),
            ]
            await chat.perform(
              "preferencesSave",
              [
                "changes": .object(changes), "expected": .object(expectedPreferences),
              ])
            guard chat.error == nil else { return }
            expectedPreferences = changes
            await chat.perform(
              "settings",
              [
                "ocrMode": .string(mode),
                "audioLanguage": .string(language),
              ])
            if chat.error == nil { dismiss() }
          }
        }.buttonStyle(FlatButtonStyle(primary: true)).disabled(limit.map { $0 < 1 } ?? false)
      }
    }.font(.system(size: 12)).padding(16).frame(width: 320)
      .task {
        let s = chat.state?.settings ?? [:]
        limit = s["maxTokens"]?.number.map(Int.init)
        summarize = s["summarize"]?.boolean ?? true
        expectedPreferences = ["maxTokens": s["maxTokens"] ?? .null, "summarize": .bool(summarize)]
        mode = s["ocrMode"]?.text ?? ""
        language = s["audioLanguage"]?.text ?? ""
      }
  }
}

private struct ComposerError: View {
  let chat: StudioWorkspace
  var body: some View {
    if let error = chat.error {
      Text(error).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
    }
  }
}
