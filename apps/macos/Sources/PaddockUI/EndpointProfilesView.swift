import PaddockClient
import SwiftUI

struct EndpointProfilesView: View {
  @Bindable var editor: EndpointEditor
  @State private var profiles: [ModelProfile] = []
  @State private var naming = false
  @State private var name = ""
  @State private var error: String?
  @State private var busy = false
  @State private var removing: ModelProfile?
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Menu("Profiles", systemImage: "slider.horizontal.3") {
          ForEach(profiles) { profile in
            Button(profile.name) { editor.useProfile(profile) }.disabled(
              editor.dirty || editor.saving)
          }
          if !editor.isCreating {
            Button("Save as profile…") {
              name = ""
              naming = true
            }
            .disabled(editor.dirty || editor.saving || editor.endpoint.revision == nil)
          }
          if !profiles.isEmpty {
            Menu("Remove profile") {
              ForEach(profiles) { profile in
                Button(profile.name, role: .destructive) { removing = profile }
              }
            }
          }
        }.menuStyle(.borderlessButton).fixedSize().disabled(busy)
        Spacer()
      }
      if let error { Text(error).font(.caption).foregroundStyle(PaddockStyle.caution) }
    }
    .task(id: "\(editor.modelID):\(editor.artifactID)") { await load() }
    .popover(isPresented: $naming) {
      VStack(alignment: .leading, spacing: 16) {
        StudioPopoverHeading(title: "Save profile")
        TextField("Profile name", text: $name).textFieldStyle(StudioPopoverFieldStyle())
        HStack {
          Button("Cancel") { naming = false }
          Spacer()
          Button("Save") { save() }.disabled(
            name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || name.count > 80 || busy)
        }.buttonStyle(FlatButtonStyle())
      }.padding(18).frame(width: 320).studioPopoverSurface()
    }
    .confirmationDialog(
      "Remove profile?",
      isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
      titleVisibility: .visible
    ) {
      Button("Remove", role: .destructive) {
        guard let removing else { return }
        busy = true
        Task {
          defer {
            busy = false
            self.removing = nil
          }
          do {
            _ = try await editor.settingsClient.inspect(
              .removeProfile(id: removing.id), as: ModelProfiles.self)
            await load()
          } catch { self.error = error.localizedDescription }
        }
      }
      Button("Cancel", role: .cancel) { removing = nil }
    } message: {
      Text("Existing instances and model files are unchanged.")
    }
  }
  private func load() async {
    do {
      let result = try await editor.settingsClient.inspect(
        .profiles(model: editor.modelID, artifact: editor.artifactID), as: ModelProfiles.self)
      try Task.checkCancellation()
      profiles = result.profiles
      error = nil
    } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
  }
  private func save() {
    guard let revision = editor.endpoint.revision, !editor.dirty else { return }
    busy = true
    naming = false
    Task {
      defer { busy = false }
      do {
        _ = try await editor.settingsClient.inspect(
          .saveProfile(port: editor.endpoint.port, revision: revision, name: name),
          as: ModelProfiles.self)
        await load()
      } catch { self.error = error.localizedDescription }
    }
  }
}
