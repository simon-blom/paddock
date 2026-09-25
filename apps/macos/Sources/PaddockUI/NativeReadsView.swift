import AppKit
import PaddockConversationCore
import SwiftUI
import UniformTypeIdentifiers

struct NativeReadsView: View {
  @Bindable var model: NativeReadsModel
  var onStart: () -> Void
  @State private var jsonMode = false
  @State private var importFile = false
  @State private var importJSON = false
  @State private var confirmNew = false
  @State private var confirmDelete = false
  @State private var confirmExample = false
  @State private var pendingSet: NativeReadsModel.SavedSet?
  @State private var pendingSession: NativeReadsModel.Session?
  @State private var confirmDeleteRead = false
  var body: some View {
    GeometryReader { geometry in
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 18) {
          header(stacked: geometry.size.width < 600)
          if let error = model.error {
            Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
              .accessibilityIdentifier("reads-error")
          }
          if model.readers.isEmpty {
            VStack(spacing: 16) {
              if model.loading {
                ProgressView().controlSize(.small)
              } else {
                Image(systemName: "list.bullet.clipboard").font(.system(size: 28)).foregroundStyle(
                  .secondary)
                Text("No model that can read is running").font(.headline)
                Button("Start a model", action: onStart).buttonStyle(FlatButtonStyle(primary: true))
              }
            }.frame(maxWidth: .infinity).padding(.vertical, 60)
          }
          if geometry.size.width >= 1100 {
            HStack(alignment: .top, spacing: 20) {
              editor.frame(maxWidth: .infinity)
              answers.frame(maxWidth: .infinity)
            }
          } else {
            editor
            answers
          }
        }.padding(24).frame(maxWidth: 1400).frame(maxWidth: .infinity)
      }
    }.background(PaddockStyle.canvas).font(.system(size: 13))
      .task {
        repeat {
          await model.refresh()
          do { try await Task.sleep(for: .seconds(5)) } catch { return }
        } while !Task.isCancelled
      }
      .fileImporter(isPresented: $importFile, allowedContentTypes: [.item]) { result in
        if case .success(let url) = result { Task { await model.loadFile(url) } }
      }
      .fileImporter(isPresented: $importJSON, allowedContentTypes: [.json]) { result in
        if case .success(let url) = result { Task { await model.loadFile(url, asJSON: true) } }
      }
      .confirmationDialog("Start a new read?", isPresented: $confirmNew, titleVisibility: .visible)
    {
      Button("Discard current draft", role: .destructive) { model.reset() }
      Button("Cancel", role: .cancel) {}
    }
      .confirmationDialog(
        "Delete this question set?", isPresented: $confirmDelete, titleVisibility: .visible
      ) {
        Button("Delete set", role: .destructive) { Task { await model.remove() } }
        Button("Cancel", role: .cancel) {}
      }
      .confirmationDialog(
        "Replace unsaved questions?",
        isPresented: Binding(
          get: { pendingSet != nil }, set: { if !$0 { pendingSet = nil } }),
        titleVisibility: .visible
      ) {
        if let set = pendingSet {
          Button("Open \(set.name)", role: .destructive) {
            model.open(set)
            pendingSet = nil
          }
        }
        Button("Cancel", role: .cancel) { pendingSet = nil }
      }
      .accessibilityIdentifier("native-reads")
      .confirmationDialog(
        "Open this read?",
        isPresented: Binding(
          get: { pendingSession != nil }, set: { if !$0 { pendingSession = nil } }),
        titleVisibility: .visible
      ) {
        if let session = pendingSession {
          Button("Open \(session.title)") {
            pendingSession = nil
            Task { await model.openSession(session.id) }
          }
        }
        Button("Cancel", role: .cancel) { pendingSession = nil }
      }
      .confirmationDialog(
        "Delete this read and its runs?", isPresented: $confirmDeleteRead, titleVisibility: .visible
      ) {
        Button("Delete read", role: .destructive) { Task { await model.clearHistory() } }
        Button("Cancel", role: .cancel) {}
      }
      .confirmationDialog(
        "Replace the draft with the example?", isPresented: $confirmExample,
        titleVisibility: .visible
      ) {
        Button("Run example") { model.runExample() }
        Button("Cancel", role: .cancel) {}
      }
  }

  private func header(stacked: Bool) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(spacing: 16) {
        Text("Reads").font(.system(size: 25, weight: .semibold)).tracking(-0.5)
          .fixedSize().accessibilityAddTraits(.isHeader)
        Spacer(minLength: 8)
        Button("New read", systemImage: "plus") {
          if model.hasWork { confirmNew = true } else { model.reset() }
        }
        .buttonStyle(FlatButtonStyle()).fixedSize()
        .disabled(model.busy || model.saving || model.importing)
        if !stacked, !model.readers.isEmpty { modelPicker.frame(width: 270) }
      }
      if stacked, !model.readers.isEmpty { modelPicker }
      HStack {
        Dropdown(
          title: "Earlier reads", value: model.activeSession?.value["title"]?.string ?? "History"
        ) {
          ForEach(model.sessions) { session in
            Button(session.title) {
              if model.hasWork {
                pendingSession = session
              } else {
                Task { await model.openSession(session.id) }
              }
            }
          }
        }.disabled(model.busy || model.saving || model.sessions.isEmpty)
          .accessibilityIdentifier("reads-sessions")
        if model.openingSession { ProgressView().controlSize(.small) }
        Spacer()
        if model.historyUnsaved {
          Button("Retry saving") { Task { await model.saveHistory() } }
            .buttonStyle(FlatButtonStyle()).disabled(model.busy || model.saving)
        }
      }
    }.accessibilityIdentifier("reads-header")
  }

  private var modelPicker: some View {
    Dropdown(
      title: "Reading model", value: model.current?.title ?? "Choose model",
      fillsWidth: true, vendor: model.current?.vendor
    ) {
      ForEach(model.readers) { reader in
        Button {
          model.port = reader.port
        } label: {
          ModelProviderMenuLabel(title: reader.title, vendor: reader.vendor)
        }
      }
    }.accessibilityIdentifier("reads-model-picker")
  }

  private var editor: some View {
    VStack(alignment: .leading, spacing: 16) {
      card("State") {
        PaddockTextEditor(text: $model.draft.state).frame(height: 160)
          .onChange(of: model.draft.state) { _, _ in model.stateError = nil }
          .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
          .accessibilityLabel("Text to read")
          .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first, url.isFileURL, !model.importing, !model.busy else {
              return false
            }
            Task { await model.loadFile(url) }
            return true
          }
        HStack {
          Button("Load a file", systemImage: "paperclip") { importFile = true }
            .buttonStyle(FlatButtonStyle()).fixedSize().disabled(model.importing || model.busy)
          if model.importing { ProgressView().controlSize(.small) }
          Text(model.fileName).lineLimit(1).foregroundStyle(.secondary)
          Spacer()
          if !model.draft.state.isEmpty {
            Button("Clear") {
              model.draft.state = ""
              model.fileName = ""
            }.buttonStyle(QuietButtonStyle())
          }
        }
        if let error = model.stateError {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
      }
      card(
        "Questions", detail: "\(model.draft.questions.count) / \(model.current?.maxQuestions ?? 64)"
      ) {
        HStack(spacing: 8) {
          TextField("Set name", text: $model.setName).textFieldStyle(StudioPopoverFieldStyle())
            .accessibilityLabel("Question set name")
          Button("Save") { Task { await model.save() } }.buttonStyle(FlatButtonStyle()).fixedSize()
            .disabled(
              model.saving || model.busy || model.validation != nil || model.setName.isEmpty)
          Menu {
            ForEach(model.sets) { set in
              Button(set.name) { if model.dirty { pendingSet = set } else { model.open(set) } }
            }
            Divider()
            Button("Save as new set") { Task { await model.save(asNew: true) } }
              .disabled(model.validation != nil || model.setName.isEmpty)
            Button("Import JSON…") { importJSON = true }
            Button("Export JSON…") {
              exportText((try? model.draft.orderedJSON()) ?? "", name: "read-questions.json")
            }
            if model.selectedSet != nil {
              Button("Delete set…", role: .destructive) { confirmDelete = true }
            }
          } label: {
            Image(systemName: "ellipsis").frame(width: 28, height: 28)
          }
          .menuStyle(.button).menuIndicator(.hidden).buttonStyle(QuietButtonStyle()).fixedSize()
          .accessibilityLabel("Question set actions").disabled(model.saving || model.busy)
        }
        editorControls
        if jsonMode {
          PaddockTextEditor(text: $model.jsonText).frame(height: 280).accessibilityLabel(
            "Questions JSON")
          Button("Apply JSON") { model.applyJSON(model.jsonText) }.buttonStyle(FlatButtonStyle())
        } else {
          ForEach($model.draft.questions) { $question in
            NativeReadQuestionRow(
              question: $question, onDuplicate: { model.duplicate(question.id) },
              onMove: { model.move(question.id, by: $0) },
              onRemove: { model.draft.questions.removeAll { $0.id == question.id } },
              position: model.draft.questions.firstIndex { $0.id == question.id } ?? 0,
              count: model.draft.questions.count,
              supportedTypes: model.current?.types ?? ["noul", "choice", "score"],
              serverError: model.rowErrors[question.questionID],
              onID: { model.editID(question.id, text: $0) },
              onInstructions: { model.editInstructions(question.id, text: $0) })
          }
          HStack(spacing: 8) {
            ForEach(
              ReadQuestion.Kind.allCases.filter {
                model.current?.types.contains($0.rawValue) != false
              }, id: \.self
            ) { kind in
              Button(kind.title, systemImage: "plus") { model.add(kind) }.buttonStyle(
                FlatButtonStyle()
              )
              .fixedSize()
            }
          }.disabled(model.draft.questions.count >= model.current?.maxQuestions ?? 64)
        }
        if let error = model.validation {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution)
        }
        if let error = model.questionsError {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
        HStack {
          Button("Run", systemImage: "play.fill") { model.run(applyJSON: jsonMode) }.buttonStyle(
            FlatButtonStyle(primary: true)
          )
          .disabled(
            jsonMode
              ? model.current == nil || model.busy || model.importing
                || model.draft.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              : !model.canRun
          ).keyboardShortcut(.return, modifiers: .command)
          if model.busy {
            ProgressView().controlSize(.small)
            Button("Cancel") { model.cancel() }.buttonStyle(QuietButtonStyle())
          }
          Spacer()
        }
      }
      DisclosureGroup("API request") {
        let request =
          (try? model.draft.orderedJSON(model: model.current?.model ?? "")) ?? ""
        Text("POST /v1/systemone\n\n" + request).font(.system(size: 11, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
        Button("Copy request") { copy(request) }.buttonStyle(QuietButtonStyle())
      }
    }
  }

  private var editorControls: some View {
    ViewThatFits(in: .horizontal) {
      HStack(spacing: 16) {
        editorTabs
        Spacer(minLength: 0)
        samplePicker
      }
      VStack(alignment: .leading, spacing: 12) {
        editorTabs
        samplePicker
      }
    }
    .onChange(of: jsonMode) { _, json in
      if json {
        model.beginJSON()
      } else if model.hasUnappliedJSON {
        if !model.applyJSON(model.jsonText) { jsonMode = true }
      }
    }
    .accessibilityIdentifier("reads-editor-controls")
  }

  private var editorTabs: some View {
    Picker("Question editor", selection: $jsonMode) {
      Text("Form").tag(false)
      Text("JSON").tag(true)
    }.pickerStyle(.segmented).labelsHidden().frame(width: 140)
      .accessibilityIdentifier("reads-editor-tabs")
  }

  private var samplePicker: some View {
    HStack(spacing: 8) {
      Text("Reads per question").font(.system(size: 12)).foregroundStyle(.secondary).fixedSize()
      Dropdown(
        title: "Reads per question",
        value: model.draft.samples == 0 ? "Auto" : "\(model.draft.samples)"
      ) {
        Button("Auto") { model.draft.samples = 0 }
        ForEach([1, 2, 4, 8, 16, 32].filter { $0 <= model.current?.maxSamples ?? 32 }, id: \.self) {
          count in
          Button("\(count)") { model.draft.samples = count }
        }
      }.fixedSize()
    }.fixedSize(horizontal: true, vertical: false)
  }

  private var answers: some View {
    card("Answers") {
      if let error = model.historyError {
        Text(error).font(.caption).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
      }
      if let result = model.result {
        HStack {
          Text(
            "\(result.response.diagnostics.timing.totalMilliseconds, specifier: "%.0f") ms · \(result.response.diagnostics.reads) reads"
          )
          .monospacedDigit().foregroundStyle(.secondary)
          Spacer()
          Dropdown(title: "Read history", value: "History") {
            ForEach(model.runs) { run in
              Button(run.at.formatted(date: .abbreviated, time: .standard) + " · " + run.excerpt) {
                model.selectedRun = run.id
              }
            }
            Divider()
            Button("Delete read", role: .destructive) { confirmDeleteRead = true }
              .disabled(model.busy)
          }.fixedSize()
        }
        if result.state == nil {
          Text("The original input was not retained with this older result.")
            .font(.caption).foregroundStyle(.secondary)
        } else if model.stale {
          Text("Edited since this read").font(.caption).foregroundStyle(PaddockStyle.caution)
        } else if model.previousRead {
          Text("Previous read · " + result.excerpt).font(.caption).foregroundStyle(.secondary)
        }
        ForEach(result.questions) { question in
          if let answer = result.response.answers[question.questionID] {
            NativeReadAnswerView(
              question: question, answer: answer,
              diagnostic: result.response.diagnostics.questions.first {
                $0.id == question.questionID
              }, readCount: result.response.diagnostics.reads)
          }
        }
        NativeReadDiagnostics(
          response: result.response, elapsedMilliseconds: result.elapsedMilliseconds)
        DisclosureGroup("Response JSON") {
          Text((try? ReadDraft.json(result.raw)) ?? "").font(.system(size: 11, design: .monospaced))
            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
        Button("Export results…") { export(result.raw, name: "read-results.json") }.buttonStyle(
          FlatButtonStyle())
      } else {
        Text(model.busy ? "Reading…" : "Run a read to see answers.").foregroundStyle(.secondary)
          .padding(.vertical, 24)
        Button("Try an example") {
          if model.hasWork { confirmExample = true } else { model.runExample() }
        }.buttonStyle(FlatButtonStyle())
          .disabled(model.current == nil || model.busy || model.saving || model.importing)
      }
    }
  }
  private func card<Content: View>(
    _ title: String, detail: String? = nil, @ViewBuilder content: () -> Content
  )
    -> some View
  {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text(title).font(.system(size: 13, weight: .semibold)).accessibilityAddTraits(.isHeader)
        Spacer()
        if let detail { Text(detail).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
      }
      content()
    }
    .padding(18).frame(maxWidth: .infinity, alignment: .leading)
    .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
  }
  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
  private func export(_ value: ConversationValue, name: String) {
    do { exportText(try ReadDraft.json(value), name: name) } catch {
      model.error = error.localizedDescription
    }
  }
  private func exportText(_ text: String, name: String) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.json]
    panel.nameFieldStringValue = name
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      do { try text.write(to: url, atomically: true, encoding: .utf8) } catch {
        model.error = error.localizedDescription
      }
    }
  }
}
