import PaddockStudio
import SwiftUI

struct StudioImageControls: View {
  @Bindable var chat: StudioWorkspace
  var body: some View {
    StudioImageSettingsForm(
      params: chat.state?.settings["imageParams"]?.object ?? [:],
      caps: chat.state?.composer?.imageCaps ?? [:], error: chat.error,
      update: { key, value in
        Task { await chat.perform("settings", ["imageParams": .object([key: value])]) }
      },
      reset: { Task { await chat.perform("settings", ["imageParams": .null]) } },
      lastSeed: chat.state?.settings["lastImageSeed"]?.number)
  }

  static func snappedSide(_ value: String, grid: Int, maxSide: Int, fallback: Int) -> Int {
    let g = max(1, grid)
    let ceiling = max(g, maxSide / g * g)
    let n = min(ceiling, max(g, Int(value) ?? fallback))
    return min(ceiling, max(g, Int((Double(n) / Double(g)).rounded()) * g))
  }
}

/// Match web ImageOptions' field order and per-conversation behavior, with
/// visible labels and one aligned control column. Dropdown.title is only an
/// accessibility label; it must never stand in for the visible form label.
struct StudioImageSettingsForm: View {
  let params: [String: StudioValue]
  let caps: [String: StudioValue]
  var error: String? = nil
  let update: (String, StudioValue) -> Void
  let reset: () -> Void
  var lastSeed: Double? = nil
  @State private var steps = ""
  @State private var seed = ""
  @State private var width = "1024"
  @State private var height = "1024"
  @State private var custom = false
  private enum Field: Hashable { case width, height, steps, seed }
  @FocusState private var focused: Field?

  init(
    params: [String: StudioValue], caps: [String: StudioValue], error: String? = nil,
    update: @escaping (String, StudioValue) -> Void, reset: @escaping () -> Void,
    lastSeed: Double? = nil
  ) {
    self.params = params
    self.caps = caps
    self.error = error
    self.update = update
    self.reset = reset
    self.lastSeed = lastSeed
    // Seed the draft without onAppear/onChange writes. Opening the popover
    // must not save settings or briefly show blank values for a saved recipe.
    _steps = State(initialValue: params["steps"]?.number.map { String(Int($0)) } ?? "")
    _seed = State(initialValue: params["seed"]?.number.map { String(Int($0)) } ?? "")
    let size = params["size"]?.text ?? "auto"
    _custom = State(initialValue: !Self.sizeChoices.contains { $0.0 == size })
    let dimensions = (size == "auto" ? caps["default_size"]?.text ?? "1024x1024" : size).split(
      separator: "x")
    _width = State(initialValue: dimensions.first.map(String.init) ?? "1024")
    _height = State(initialValue: dimensions.last.map(String.init) ?? "1024")
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        StudioPopoverHeading(title: "Picture settings")
        Button {
          custom = false
          steps = ""
          seed = ""
          reset()
        } label: {
          Image(systemName: "arrow.counterclockwise")
            .frame(width: 26, height: 26).contentShape(Rectangle())
        }.buttonStyle(QuietButtonStyle()).disabled(!changed)
          .help("Reset picture settings").accessibilityLabel("Reset picture settings")
          .accessibilityIdentifier("image-settings-reset")
      }
      VStack(spacing: 8) {
        row("Size") {
          Dropdown(
            title: "Size",
            value: custom
              ? "Custom"
              : Self.sizeChoices.first { $0.0 == params["size"]?.text }?.1 ?? "Model default",
            fillsWidth: true
          ) {
            ForEach(Self.sizeChoices, id: \.0) { value, label in
              selectedButton(label, selected: !custom && (params["size"]?.text ?? "auto") == value)
              {
                custom = false
                update("size", .string(value))
              }
            }
            selectedButton("Custom", selected: custom) {
              custom = true
              let initial =
                params["size"]?.text == "auto"
                ? caps["default_size"]?.text ?? "1024x1024" : params["size"]?.text ?? "1024x1024"
              setDimensions(initial)
              update("size", .string(initial))
            }
          }
        }
        if custom {
          row("Pixels") {
            HStack(spacing: 6) {
              TextField("Width", text: $width).accessibilityLabel("Image width")
                .focused($focused, equals: .width).onSubmit(commitSize)
              Text("×").foregroundStyle(.secondary).accessibilityHidden(true)
              TextField("Height", text: $height).accessibilityLabel("Image height")
                .focused($focused, equals: .height).onSubmit(commitSize)
            }.textFieldStyle(StudioPopoverFieldStyle()).monospacedDigit()
          }
        }
        choice(
          "Quality", key: "quality",
          values: [
            ("auto", "Model default"), ("low", "Low"), ("medium", "Medium"), ("high", "High"),
          ])
        row("Steps") {
          TextField("Automatic", text: $steps).textFieldStyle(StudioPopoverFieldStyle())
            .monospacedDigit().accessibilityLabel("Image denoising steps")
            .focused($focused, equals: .steps).onSubmit(commitSteps)
            .onChange(of: steps) { _, value in
              if value.isEmpty {
                update("steps", .null)
              } else if let n = Int(value), (1...Int(caps["max_steps"]?.number ?? 100)).contains(n)
              {
                update("steps", .number(Double(n)))
              }
            }
        }
      }
      VStack(spacing: 8) {
        row("Seed") {
          Dropdown(
            title: "Seed",
            value: params["seed"]?.number != nil
              ? "Pinned"
              : params["seed"]?.text == "random" ? "New every picture" : "Automatic",
            fillsWidth: true
          ) {
            selectedButton(
              "Automatic",
              selected: (params["seed"]?.text ?? "thread") == "thread"
                && params["seed"]?.number == nil
            ) {
              update("seed", .string("thread"))
            }
            selectedButton("New every picture", selected: params["seed"]?.text == "random") {
              update("seed", .string("random"))
            }
            selectedButton("Pinned", selected: params["seed"]?.number != nil) {
              let n =
                params["seed"]?.number ?? lastSeed ?? Double(Int.random(in: 0...Int(Int32.max)))
              seed = String(Int(n))
              update("seed", .number(n))
            }
          }
        }
        if params["seed"]?.number != nil {
          row("Number") {
            TextField("Pinned seed", text: $seed).textFieldStyle(StudioPopoverFieldStyle())
              .monospacedDigit().accessibilityLabel("Pinned image seed")
              .focused($focused, equals: .seed).onSubmit(commitSeed)
              .onChange(of: seed) { _, value in
                if let n = Int(value), (0...Int(Int32.max)).contains(n) {
                  update("seed", .number(Double(n)))
                }
              }
          }
        }
      }
      VStack(spacing: 8) {
        numberChoice("Count", key: "n", range: 1...max(1, min(4, Int(caps["max_n"]?.number ?? 1))))
        {
          $0 == 1 ? "1 picture" : "\($0) pictures"
        }
        choice(
          "Format", key: "format", fallback: "png",
          values: (caps["output_formats"]?.array?.compactMap(\.text) ?? ["png"]).map {
            ($0, $0.uppercased())
          })
        choice(
          "Background", key: "background",
          values: [("auto", "Model default"), ("opaque", "Opaque"), ("transparent", "Transparent")])
        if caps["stream"]?.boolean == true, (caps["max_partial_images"]?.number ?? 0) > 0 {
          numberChoice(
            "Previews", key: "previews",
            range: 0...max(0, min(3, Int(caps["max_partial_images"]?.number ?? 0)))
          ) { $0 == 0 ? "Final picture only" : "\($0) while rendering" }
          .disabled((params["n"]?.number ?? 1) > 1)
          .help(
            (params["n"]?.number ?? 1) > 1
              ? "Previews are available for one picture at a time"
              : "Picture previews during rendering")
        }
      }
      if let error {
        Text(error).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
      }
    }.font(.system(size: 12)).padding(16).frame(width: 360)
      .defaultFocus($focused, nil)
      .onChange(of: width) { _, _ in if validSize && custom { commitSize() } }
      .onChange(of: height) { _, _ in if validSize && custom { commitSize() } }
      .onChange(of: focused) { previous, _ in
        switch previous {
        case .width, .height: if custom { commitSize() }
        case .steps: commitSteps()
        case .seed: commitSeed()
        case nil: break
        }
      }
  }
  private static var sizeChoices: [(String, String)] {
    [
      ("auto", "Model default"), ("1024x1024", "Square · 1024 × 1024"),
      ("1536x1024", "Landscape · 1536 × 1024"), ("1024x1536", "Portrait · 1024 × 1536"),
    ]
  }
  private func setDimensions(_ size: String) {
    let parts = size.split(separator: "x")
    if parts.count == 2 {
      width = String(parts[0])
      height = String(parts[1])
    }
  }
  private func commitSize() {
    let grid = max(1, Int(caps["size_multiple"]?.number ?? 32))
    let maxSide = max(grid, Int(caps["max_side"]?.number ?? 2048))
    let previous = (params["size"]?.text ?? "1024x1024").split(separator: "x")
    width = String(
      StudioImageControls.snappedSide(
        width, grid: grid, maxSide: maxSide,
        fallback: previous.first.flatMap { Int($0) } ?? 1024))
    height = String(
      StudioImageControls.snappedSide(
        height, grid: grid, maxSide: maxSide,
        fallback: previous.last.flatMap { Int($0) } ?? 1024))
    update("size", .string("\(width)x\(height)"))
  }
  private func commitSteps() {
    let n = Int(steps).flatMap {
      $0 > 0 ? min($0, max(1, Int(caps["max_steps"]?.number ?? 100))) : nil
    }
    steps = n.map(String.init) ?? ""
    update("steps", n.map { .number(Double($0)) } ?? .null)
  }
  private func commitSeed() {
    guard let current = params["seed"]?.number else { return }
    let n = min(Int(Int32.max), max(0, Int(seed) ?? Int(current)))
    seed = String(n)
    update("seed", .number(Double(n)))
  }
  private var validSize: Bool {
    guard let w = Int(width), let h = Int(height) else { return false }
    let grid = max(1, Int(caps["size_multiple"]?.number ?? 32))
    let maxSide = Int(caps["max_side"]?.number ?? 2048)
    return [w, h].allSatisfy { $0 >= grid && $0 <= maxSide && $0 % grid == 0 }
  }
  private var changed: Bool {
    (params["size"]?.text ?? "auto") != "auto"
      || (params["quality"]?.text ?? "auto") != "auto" || params["steps"]?.number != nil
      || params["seed"]?.number != nil || (params["seed"]?.text ?? "thread") != "thread"
      || (params["n"]?.number ?? 1) != 1 || (params["format"]?.text ?? "png") != "png"
      || (params["background"]?.text ?? "auto") != "auto" || (params["previews"]?.number ?? 2) != 2
  }
  private func row<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View
  {
    HStack(spacing: 12) {
      Text(title).foregroundStyle(.secondary).frame(width: 74, alignment: .leading)
      content().frame(maxWidth: .infinity, alignment: .leading)
    }.frame(minHeight: 30)
      .accessibilityElement(children: .contain)
      .accessibilityIdentifier("image-settings-\(title.lowercased())")
  }
  private func selectedButton(_ title: String, selected: Bool, action: @escaping () -> Void)
    -> some View
  {
    Button(action: action) {
      if selected { Label(title, systemImage: "checkmark") } else { Text(title) }
    }
  }
  private func choice(
    _ title: String, key: String, fallback: String = "auto", values: [(String, String)]
  ) -> some View {
    let current = params[key]?.text ?? fallback
    let label = values.first { $0.0 == current }?.1 ?? current
    return row(title) {
      if values.count == 1, values[0].0 == current {
        Text(label).padding(.horizontal, 10).accessibilityLabel("\(title): \(label)")
      } else {
        Dropdown(title: title, value: label, fillsWidth: true) {
          ForEach(values, id: \.0) { value, label in
            selectedButton(label, selected: value == current) { update(key, .string(value)) }
          }
        }
      }
    }
  }
  private func numberChoice(
    _ title: String, key: String, range: ClosedRange<Int>, label: @escaping (Int) -> String
  ) -> some View {
    let saved = Int(params[key]?.number ?? (key == "previews" ? 2 : 1))
    // The request clamps preview availability, but rejects an unsupported
    // picture count. Don't disguise a count saved for another endpoint.
    let current = key == "previews" ? min(range.upperBound, max(range.lowerBound, saved)) : saved
    return row(title) {
      if range.lowerBound == range.upperBound, range.contains(current) {
        Text(label(current)).padding(.horizontal, 10).accessibilityLabel(
          "\(title): \(label(current))")
      } else {
        Dropdown(title: title, value: label(current), fillsWidth: true) {
          ForEach(Array(range), id: \.self) { n in
            selectedButton(label(n), selected: n == current) { update(key, .number(Double(n))) }
          }
        }
      }
    }
  }
}
