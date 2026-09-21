import Foundation
import Observation
import PaddockStudio
import SwiftUI

@MainActor @Observable final class StudioInstructionsModel {
  struct Snapshot: Decodable {
    struct Block: Decodable {
      let label: String
      let text: String
    }
    let conversationId: String
    let body: String
    let blocks: [Block]
  }
  var body = ""
  var error: String?
  private(set) var snapshot: Snapshot?
  private(set) var loading = false
  private(set) var saving = false
  private(set) var presets: StudioPresetPage?
  var query = ""
  var pickedName: String?
  private var generation = 0
  private var searchGeneration = 0
  @ObservationIgnored private var pending: Task<Void, Never>?
  var dirty: Bool { snapshot != nil && body != snapshot?.body }
  var hasWork: Bool { dirty || saving }
  func load(command: StudioCommand) async {
    guard !hasWork, !loading else { return }
    generation += 1
    loading = true
    defer { loading = false }
    do {
      let reply = try await command("instructionsGet", [:])
      snapshot = try studioDecode(Snapshot.self, reply["instructions"])
      body = snapshot?.body ?? ""
      error = nil
      pickedName = nil
    } catch { self.error = error.localizedDescription }
  }
  func search(command: StudioCommand, page: Int = 0) async {
    searchGeneration += 1
    let epoch = searchGeneration
    do {
      let result = try await command(
        "promptList", ["search": .string(query), "page": .number(Double(page))])
      guard epoch == searchGeneration, !Task.isCancelled else { return }
      presets = try studioDecode(StudioPresetPage.self, result["library"])
    } catch { if epoch == searchGeneration { self.error = error.localizedDescription } }
  }
  func pick(_ id: String, command: StudioCommand) async {
    guard !saving, !loading else { return }
    loading = true
    defer { loading = false }
    do {
      let reply = try await command("promptGet", ["id": .string(id)])
      let preset = try studioDecode(StudioPreset.self, reply["prompt"])
      body = preset.body
      pickedName = preset.name
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func discard() {
    guard !saving, !loading else { return }
    generation += 1
    body = ""
    snapshot = nil
    error = nil
    pickedName = nil
  }
  func apply(command: @escaping StudioCommand) {
    guard !saving, !loading, let original = snapshot else { return }
    guard body.utf8.count <= 128 * 1024 else {
      error = "Instructions exceed 128 KiB."
      return
    }
    let value = body
    saving = true
    error = nil
    pending = Task {
      defer { self.saving = false }
      do {
        _ = try await command(
          "instructionsApply",
          [
            "conversationId": .string(original.conversationId), "expected": .string(original.body),
            "body": .string(value),
          ])
        self.snapshot = Snapshot(
          conversationId: original.conversationId, body: value, blocks: original.blocks)
      } catch { self.error = error.localizedDescription }
    }
  }
  func settle() async { await pending?.value }
}

struct StudioInstructionControls: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.studioLibrary) private var library
  var body: some View {
    if let library {
      InstructionEditor(model: library.instructions, library: library, chat: chat)
    } else {
      Text("Open the Studio workspace to edit instructions.").padding()
    }
  }
}

private struct InstructionEditor: View {
  @Bindable var model: StudioInstructionsModel
  let library: StudioLibraryModel
  let chat: StudioWorkspace
  @Environment(\.dismiss) private var dismiss
  @State private var picking = false
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 12) {
        StudioPopoverHeading(title: "Instructions")
        if let error = model.error {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
        if model.loading { ProgressView().controlSize(.small) }
        PaddockTextEditor(text: $model.body).scrollContentBackground(.hidden).frame(height: 155)
          .padding(8)
          .background(
            PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
          )
          .accessibilityLabel("System instructions").disabled(
            model.loading || model.saving || model.snapshot == nil)
        Text(
          model.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "No system prompt"
            : model.pickedName.map { "Loaded '\($0)' · editable copy" } ?? "Custom prompt"
        ).foregroundStyle(.secondary)
        if let blocks = model.snapshot?.blocks, !blocks.isEmpty {
          DisclosureGroup("Also sent by enabled tools") {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
              VStack(alignment: .leading, spacing: 4) {
                Text(block.label).fontWeight(.medium)
                Text(block.text).textSelection(.enabled)
              }.padding(.vertical, 6)
            }
          }
        }
        Button("Use a saved preset", systemImage: "chevron.down") { picking.toggle() }.buttonStyle(
          FlatButtonStyle())
        if picking {
          TextField("Search presets", text: $model.query).textFieldStyle(StudioPopoverFieldStyle())
          if let page = model.presets {
            VStack(alignment: .leading, spacing: 4) {
              ForEach(page.rows) { row in
                Button(row.name) {
                  Task {
                    await model.pick(row.id, command: library.command)
                    if model.error == nil { picking = false }
                  }
                }
                .buttonStyle(FlatButtonStyle()).disabled(model.loading || model.saving)
              }
              if page.rows.isEmpty { Text("No matching presets").foregroundStyle(.secondary) }
              HStack {
                Button("Previous") {
                  Task { await model.search(command: library.command, page: page.page - 1) }
                }.disabled(page.page == 0)
                Spacer()
                Button("Next") {
                  Task { await model.search(command: library.command, page: page.page + 1) }
                }.disabled((page.page + 1) * page.pageSize >= page.matched)
              }.buttonStyle(FlatButtonStyle())
            }
          }
        }
        HStack {
          Button("Manage presets") {
            library.onManage?()
            dismiss()
          }
          Button("Save as preset…") {
            library.create(body: model.body)
            if library.error == nil {
              library.onManage?()
              dismiss()
            } else {
              model.error = library.error
            }
          }.disabled(model.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }.buttonStyle(FlatButtonStyle())
        WorkspaceRule()
        HStack {
          Button("Clear") {
            model.body = ""
            model.pickedName = nil
          }
          Button("Discard draft") {
            model.discard()
            Task { await model.load(command: library.command) }
          }
          Spacer()
          Button("Apply") {
            model.apply(command: library.command)
            Task {
              await model.settle()
              if model.error == nil { dismiss() }
            }
          }.buttonStyle(FlatButtonStyle(primary: true)).disabled(chat.busy || model.snapshot == nil)
        }.buttonStyle(FlatButtonStyle()).disabled(model.saving || model.loading)
      }.font(.system(size: 12)).padding(16)
    }.frame(width: 420).frame(maxHeight: 580)
      .task { await model.load(command: library.command) }
      .task(id: picking ? model.query : nil) {
        guard picking else { return }
        do {
          try await Task.sleep(for: .milliseconds(180))
          try Task.checkCancellation()
        } catch { return }
        await model.search(command: library.command)
      }
  }
}
