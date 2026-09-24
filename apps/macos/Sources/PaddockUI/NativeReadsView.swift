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
  @State private var pendingSet: NativeReadsModel.SavedSet?
  var body: some View {
    GeometryReader { geometry in
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 20) {
          PageHeading(title: "Reads") {
            Button("New read", systemImage: "plus") {
              if model.hasWork { confirmNew = true } else { model.reset() }
            }
            .buttonStyle(FlatButtonStyle()).disabled(model.busy || model.saving)
            if !model.readers.isEmpty {
              Menu {
                ForEach(model.readers) { reader in
                  Button {
                    model.port = reader.port
                  } label: {
                    ModelProviderMenuLabel(title: reader.title, vendor: reader.vendor)
                  }
                }
              } label: {
                HStack(spacing: 8) {
                  ModelProviderLogo(vendor: model.current?.vendor)
                  Text(model.current?.title ?? "Choose model").lineLimit(1)
                  Image(systemName: "chevron.down").font(.system(size: 9))
                }
              }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .accessibilityLabel("Reading model")
            }
          }
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
          } else if geometry.size.width >= 1100 {
            HStack(alignment: .top, spacing: 20) {
              editor.frame(maxWidth: .infinity)
              answers.frame(maxWidth: .infinity)
            }
          } else {
            editor
            answers
          }
        }.padding(28).frame(maxWidth: 1400).frame(maxWidth: .infinity)
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
  }
  private var editor: some View {
    VStack(alignment: .leading, spacing: 18) {
      card("State") {
        PaddockTextEditor(text: $model.draft.state).frame(minHeight: 160, maxHeight: 260)
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
            .buttonStyle(FlatButtonStyle()).disabled(model.importing || model.busy)
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
      }
      card("Questions") {
        HStack {
          TextField("Set name", text: $model.setName).textFieldStyle(StudioPopoverFieldStyle())
          Button("Save") { Task { await model.save() } }.buttonStyle(FlatButtonStyle())
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
            Button("Export JSON…") { export(model.draft.setBody, name: "read-questions.json") }
            if model.selectedSet != nil {
              Button("Delete set…", role: .destructive) { confirmDelete = true }
            }
          } label: {
            Image(systemName: "ellipsis").frame(width: 28, height: 28)
          }
          .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
          .accessibilityLabel("Question set actions").disabled(model.saving || model.busy)
        }
        HStack {
          Picker("Question editor", selection: $jsonMode) {
            Text("Form").tag(false)
            Text("JSON").tag(true)
          }
          .pickerStyle(.segmented).frame(width: 140)
          .onChange(of: jsonMode) { _, json in
            if json {
              model.beginJSON()
            } else if model.hasUnappliedJSON {
              model.applyJSON(model.jsonText)
              if model.error != nil { jsonMode = true }
            }
          }
          Spacer()
          Picker("Reads", selection: $model.draft.samples) {
            Text("Auto").tag(0)
            ForEach(
              [1, 2, 4, 8, 16, 32].filter { $0 <= model.current?.maxSamples ?? 32 }, id: \.self
            ) { Text("\($0)").tag($0) }
          }.fixedSize()
        }
        if jsonMode {
          PaddockTextEditor(text: $model.jsonText).frame(height: 280).accessibilityLabel(
            "Questions JSON")
          Button("Apply JSON") { model.applyJSON(model.jsonText) }.buttonStyle(FlatButtonStyle())
        } else {
          ForEach($model.draft.questions) { $question in
            NativeReadQuestionRow(
              question: $question, onDuplicate: { model.duplicate(question.id) },
              onMove: { model.move(question.id, by: $0) },
              onRemove: { model.draft.questions.removeAll { $0.id == question.id } })
          }
          HStack {
            ForEach(ReadQuestion.Kind.allCases, id: \.self) { kind in
              Button(kind.title, systemImage: "plus") { model.add(kind) }.buttonStyle(
                FlatButtonStyle())
            }
          }.disabled(model.draft.questions.count >= model.current?.maxQuestions ?? 64)
        }
        if let error = model.validation {
          Text(error).font(.caption).foregroundStyle(PaddockStyle.caution)
        }
        HStack {
          Button("Run", systemImage: "play.fill") { model.run() }.buttonStyle(
            FlatButtonStyle(primary: true)
          )
          .disabled(!model.canRun || jsonMode).keyboardShortcut(.return, modifiers: .command)
          if model.busy {
            ProgressView().controlSize(.small)
            Button("Cancel") { model.cancel() }.buttonStyle(QuietButtonStyle())
          }
          Spacer()
          Text("\(model.draft.questions.count) / \(model.current?.maxQuestions ?? 64)")
            .foregroundStyle(.secondary).monospacedDigit()
        }
      }
      DisclosureGroup("API request") {
        let request =
          (try? ReadDraft.json(model.draft.request(model: model.current?.model ?? ""))) ?? ""
        Text("POST /v1/systemone\n\n" + request).font(.system(size: 11, design: .monospaced))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
        Button("Copy request") { copy(request) }.buttonStyle(QuietButtonStyle())
      }
    }
  }
  private var answers: some View {
    card("Answers") {
      if let result = model.result {
        HStack {
          Text(
            "\(result.response.diagnostics.timing.totalMilliseconds, specifier: "%.0f") ms · \(result.response.diagnostics.reads) reads"
          )
          .monospacedDigit().foregroundStyle(.secondary)
          Spacer()
          Menu("History") {
            ForEach(model.runs) { run in
              Button(run.at.formatted(date: .omitted, time: .standard)) {
                model.selectedRun = run.id
              }
            }
          }.fixedSize()
        }
        if model.stale {
          Text("Edited since this read").font(.caption).foregroundStyle(PaddockStyle.caution)
        }
        ForEach(result.questions) { question in
          if let answer = result.response.answers[question.questionID] {
            NativeReadAnswerView(
              question: question, answer: answer,
              diagnostic: result.response.diagnostics.questions.first {
                $0.id == question.questionID
              })
          }
        }
        DisclosureGroup("Response JSON") {
          Text((try? ReadDraft.json(result.raw)) ?? "").font(.system(size: 11, design: .monospaced))
            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
        Button("Export results…") { export(result.raw, name: "read-results.json") }.buttonStyle(
          FlatButtonStyle())
      } else {
        Text(model.busy ? "Reading…" : "Run a read to see answers.").foregroundStyle(.secondary)
          .padding(.vertical, 24)
      }
    }
  }
  private func card<Content: View>(_ title: String, @ViewBuilder content: () -> Content)
    -> some View
  {
    VStack(alignment: .leading, spacing: 14) {
      Text(title).font(.headline)
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
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.json]
    panel.nameFieldStringValue = name
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      do { try ReadDraft.json(value).write(to: url, atomically: true, encoding: .utf8) } catch {
        model.error = error.localizedDescription
      }
    }
  }
}
