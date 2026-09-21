import PaddockClient
import SwiftUI

struct CloudCatalogView: View {
  @Bindable var model: CloudBrowserModel
  var connections: ConnectionsModel?
  var account: CloudConnection?
  var service: CloudService
  var catalogAvailable = true
  var listHeader: AnyView? = nil
  var listFooter: AnyView? = nil
  var connectionPrompt: AnyView? = nil
  @State private var query = ""
  @State private var filters: Set<CloudFeature> = []
  @State private var selection: String?
  // Newest-published first is the default here, even though the web starts on Trending.
  @AppStorage("openRouterSortOrder") private var order: CloudOrder = .newest
  @FocusState private var searchFocused: Bool
  @FocusState private var listFocused: Bool
  private var orders: [CloudOrder] {
    CloudCatalogPresentation.orders(for: model.models, ranked: model.ranked)
  }
  private var effectiveOrder: CloudOrder { orders.contains(order) ? order : .name }

  var body: some View {
    let visible = CloudCatalogPresentation.entries(
      catalogAvailable ? model.models : [], query: query, filters: filters, order: effectiveOrder)
    let active = visible.first { $0.id == selection } ?? visible.first
    let enabled = Set(account?.models.map(\.pickKey) ?? [])
    CatalogColumns {
      VStack(alignment: .leading, spacing: 0) {
        listHeader
        controls.disabled(!catalogAvailable)
        WorkspaceRule()
        HStack {
          Text("\(visible.count) of \(catalogAvailable ? model.models.count : 0) models")
          Spacer()
          if catalogAvailable && model.loading { ProgressView().controlSize(.mini) }
        }.font(.system(size: 10)).foregroundStyle(.secondary).padding(.horizontal, 16).padding(
          .vertical, 12)
        if catalogAvailable, let error = model.error {
          Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
            .padding(.horizontal, 16).padding(.bottom, 8)
        }
        ScrollViewReader { proxy in
          PaddockScrollView {
            LazyVStack(spacing: 2) {
              ForEach(visible) { entry in
                row(entry, selected: active?.id == entry.id, added: enabled.contains(entry.id)).id(
                  entry.id)
              }
            }.padding(.horizontal, 8).padding(.bottom, 16)
          }.scrollBounceBehavior(.basedOnSize)
            .focusable().focusEffectDisabled().focused($listFocused)
            .onMoveCommand { direction in
              guard direction == .up || direction == .down, !visible.isEmpty else { return }
              let index = visible.firstIndex { $0.id == active?.id } ?? 0
              let next = max(0, min(visible.count - 1, index + (direction == .down ? 1 : -1)))
              selection = visible[next].id
              proxy.scrollTo(visible[next].id)
            }
            .onChange(of: visible.map(\.id)) { _, ids in
              if let first = ids.first { proxy.scrollTo(first, anchor: .top) }
            }
        }.frame(maxHeight: .infinity)
        listFooter
      }.background(PaddockStyle.canvas)
        .accessibilityIdentifier("cloud-catalog-section")
    } detail: {
      if !catalogAvailable {
        connectionPrompt
      } else if let active {
        CloudModelDetailView(
          entry: active, browser: model, service: service,
          enabled: enabled, canAdd: canAdd, onAdd: add
        )
        .id(active.id)
      } else {
        VStack(spacing: 12) {
          if model.loading {
            ProgressView("Loading \(service.rawValue) models…")
          } else {
            Text(model.error == nil ? "No matching models" : "Catalog unavailable")
              .font(.system(size: 18, weight: .medium))
            Button(model.error == nil ? "Clear filters" : "Try again") {
              if model.error == nil {
                query = ""
                filters = []
              } else {
                Task { await model.refresh() }
              }
            }.buttonStyle(FlatButtonStyle())
            if model.error == nil, let pick = CloudCatalogPresentation.manualPick(query) {
              Button("Add \(pick.id) as a model ID") { add(pick) }
                .font(.system(size: 12)).buttonStyle(FlatButtonStyle())
                .disabled(!canAdd || enabled.contains(pick.pickKey))
                .help("Adds this exact ID without claiming it is available to your account.")
                .accessibilityIdentifier("cloud-add-manual-id")
            }
          }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }.task(id: "\(ObjectIdentifier(model))/\(catalogAvailable)") {
      if catalogAvailable { await model.loadIfNeeded() }
    }
    .task(id: active?.id) {
      if catalogAvailable && service == .openrouter { await model.loadProviders(active?.id) }
    }
    .accessibilityIdentifier("openrouter-browser")
  }

  private var canAdd: Bool {
    connections?.loaded == true && connections?.saving == false && connections?.hasDraft == false
  }
  private func add(_ pick: CloudModelPick) {
    Task { await connections?.add(pick, to: account, service: service) }
  }

  private var controls: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 7) {
        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        TextField("Search cloud models", text: $query).textFieldStyle(.plain)
          .focused($searchFocused).accessibilityIdentifier("openrouter-search")
        if !query.isEmpty {
          Button("Clear search", systemImage: "xmark.circle.fill") { query = "" }
            .labelStyle(.iconOnly).buttonStyle(.plain).foregroundStyle(.secondary)
        }
        Button("⌘F") { searchFocused = true }.buttonStyle(.plain)
          .keyboardShortcut("f", modifiers: .command).foregroundStyle(.secondary)
          .accessibilityLabel("Focus cloud model search")
      }.font(.system(size: 12)).padding(.horizontal, 10).frame(height: 36)
        .background(
          PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
        )
        .overlay(
          RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
            .stroke(searchFocused ? PaddockStyle.accent : PaddockStyle.border, lineWidth: 1))
      Dropdown(title: "Order by", value: effectiveOrder.title, fillsWidth: true) {
        Picker("Order by", selection: Binding(get: { effectiveOrder }, set: { order = $0 })) {
          ForEach(orders) { option in
            Text(option.title).tag(option)
          }
        }.pickerStyle(.inline)
      }.accessibilityIdentifier("openrouter-sort-order")
        .help(
          "Search relevance first. Token price compares input plus output rates; unknown values sort last."
        )
      let available = CloudFeature.allCases.filter { feature in
        catalogAvailable
          && (model.models.contains { feature.matches($0) } || filters.contains(feature))
      }
      if !available.isEmpty {
        HStack(spacing: 5) {
          ForEach(available) { feature in
            Button {
              if !filters.insert(feature).inserted { filters.remove(feature) }
            } label: {
              Image(systemName: feature.symbol).font(.system(size: 12)).frame(width: 30, height: 26)
                .background(
                  filters.contains(feature) ? PaddockStyle.elevated : .clear,
                  in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
            }.buttonStyle(QuietButtonStyle()).accessibilityLabel("Filter: \(feature.rawValue)")
              .accessibilityAddTraits(filters.contains(feature) ? .isSelected : [])
              .accessibilityIdentifier("cloud-filter-\(feature.rawValue)")
              .help("\(feature.rawValue) models\(filters.contains(feature) ? " · selected" : "")")
          }
          Spacer(minLength: 0)
        }
      }
    }.padding(16)
  }

  private func row(_ entry: CloudModel, selected: Bool, added: Bool) -> some View {
    // Sibling buttons, never a button nested inside a selectable row button.
    // Fixed height keeps lazy scroll geometry stable across recycled rows.
    HStack(spacing: 4) {
      Button {
        selection = entry.id
        listFocused = true
      } label: {
        HStack(spacing: 10) {
          ModelAvatar(vendor: CloudCatalogPresentation.vendor(entry) ?? service.vendor, size: 34)
          VStack(alignment: .leading, spacing: 6) {
            Text(CloudCatalogPresentation.name(entry)).font(.system(size: 12, weight: .medium))
              .lineLimit(1)
            if service == .openrouter || entry.promptPrice != nil || entry.completionPrice != nil {
              Text(CloudCatalogPresentation.priceSummary(entry)).font(.system(size: 10))
                .monospacedDigit().lineLimit(1)
                .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
              ForEach(CloudFeature.allCases.filter { $0.matches(entry) }) { feature in
                Image(systemName: feature.symbol).help(feature.rawValue)
              }
              if let ctx = entry.ctx { Text("\(CloudCatalogPresentation.tokens(ctx)) ctx") }
            }.font(.system(size: 10)).foregroundStyle(.secondary).frame(height: 12)
          }
          Spacer(minLength: 0)
        }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading).contentShape(
          Rectangle())
      }.buttonStyle(.plain).help(entry.display ?? entry.id)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("cloud-model-\(entry.id)")
      CloudAddButton(added: added, enabled: canAdd, label: entry.id) {
        add(CloudModelPick(model: entry, provider: nil))
      }
    }.padding(.horizontal, 10).frame(height: 76)
      .background(
        selected ? PaddockStyle.elevated : .clear,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
  }
}

struct CloudAddButton: View {
  let added: Bool
  let enabled: Bool
  let label: String
  var action: () -> Void
  var body: some View {
    Button(action: action) {
      Group {
        if added { Image(systemName: "checkmark") } else { Text("Add") }
      }.font(.system(size: 11, weight: .medium)).frame(width: 40, height: 28).contentShape(
        Rectangle())
    }.buttonStyle(QuietButtonStyle()).disabled(added || !enabled)
      .accessibilityLabel("\(added ? "Added" : "Add") \(label)")
      .accessibilityIdentifier("cloud-add-\(label)").help(
        added ? "Already in your Studio pickers" : "Add to Studio")
  }
}
