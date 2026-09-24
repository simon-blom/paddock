import AppKit
import Observation
import PaddockClient
import SwiftUI

@MainActor @Observable
final class BenchmarksModel {
  var selected: String?
  var concurrency = 1
  var long = false
  private(set) var running = false
  private(set) var reports: [BenchmarkReport] = []
  private(set) var error: String?
  private(set) var cancelled = false
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored let client: any ManagerLoading
  init(client: any ManagerLoading) { self.client = client }
  func load() async {
    do {
      reports = try await client.inspect(.benchmarkHistory, as: BenchmarkHistory.self).reports
    } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
  }
  func start(_ runner: RunnerInfo) {
    guard !running else { return }
    running = true
    error = nil
    cancelled = false
    let concurrency = concurrency
    let long = long
    task = Task {
      defer {
        running = false
        task = nil
      }
      do {
        _ = try await client.inspect(
          .benchmark(port: runner.port, pid: runner.pid, concurrency: concurrency, long: long),
          as: BenchmarkReport.self, timeout: .seconds(900))
        await load()
      } catch {
        if Task.isCancelled { cancelled = true } else { self.error = error.localizedDescription }
      }
    }
  }
  func cancel() {
    let pending = task
    pending?.cancel()
    // Completion can race cancellation after Rust persisted a full report.
    // Reconcile history on a fresh task rather than claiming it was discarded.
    Task {
      await pending?.value
      await load()
    }
  }
  func export(_ report: BenchmarkReport) {
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "Paddock-benchmark-\(report.id).json"
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      Task {
        do {
          _ = try await self.client.inspect(
            .exportBenchmark(id: report.id, path: url.path), as: NativeExportReceipt.self)
          NSWorkspace.shared.activateFileViewerSelecting([url])
        } catch { self.error = error.localizedDescription }
      }
    }
  }
}

struct BenchmarksView: View {
  @Bindable var model: BenchmarksModel
  let runners: [RunnerInfo]
  @State private var confirm = false
  private var choices: [RunnerInfo] { runners.filter { $0.model != nil && $0.status == "ok" } }
  private var selected: RunnerInfo? { choices.first { $0.id == model.selected } ?? choices.first }
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 28) {
        PageHeading(title: "Benchmarks") { EmptyView() }
        SettingsGroup(title: "Text generation") {
          if let selected {
            SettingsRow(title: "Instance", compact: true) {
              Dropdown(title: "Instance", value: selected.title) {
                ForEach(choices) { choice in Button(choice.title) { model.selected = choice.id } }
              }.disabled(model.running)
            }
            SettingsRow(title: "Prompt", compact: true) {
              Picker("Prompt", selection: $model.long) {
                Text("Short").tag(false)
                Text("Long").tag(true)
              }.pickerStyle(.segmented).frame(width: 190).disabled(model.running)
            }
            SettingsRow(title: "Concurrency", compact: true) {
              Picker("Concurrency", selection: $model.concurrency) {
                Text("1 request").tag(1)
                Text("4 requests").tag(4)
              }.pickerStyle(.segmented).frame(width: 190).disabled(model.running)
            }
            HStack {
              if model.running {
                ProgressView().controlSize(.small)
                Text("Measuring…")
                Button("Cancel") { model.cancel() }
              } else {
                Button("Run benchmark", systemImage: "speedometer") { confirm = true }
                  .disabled(selected.inFlight != 0)
              }
            }
          } else {
            Text("Start a local chat model to run a benchmark.").foregroundStyle(.secondary)
          }
          if model.cancelled {
            Text("Benchmark cancelled. Only completed runs appear in history.").foregroundStyle(
              .secondary)
          }
          if let error = model.error {
            Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
          }
        }
        ForEach(model.reports) { report in
          SettingsGroup(title: report.model) {
            HStack {
              Text(
                "\(Date(timeIntervalSince1970: Double(report.createdAtMs) / 1000).formatted(date: .abbreviated, time: .shortened)) · c=\(report.concurrency) · \(report.promptWords) prompt words"
              ).foregroundStyle(.secondary)
              Spacer()
              Button("Export…", systemImage: "square.and.arrow.up") { model.export(report) }
            }
            measure("Aggregate output · end-to-end", report.aggregateOutputTokS, unit: "tok/s")
            measure("Median first token", report.ttftMedianMs.map { $0 / 1000 }, unit: "s")
            measure("Streaming-event gap · p99", report.streamEventGapP99Ms, unit: "ms")
            DisclosureGroup("Measurement details") {
              VStack(alignment: .leading, spacing: 10) {
                Text(
                  "\(report.trials) trials · \(report.warmups) excluded warmup · \(report.outputLimit)-token reply limit"
                )
                Text(report.cachePolicy)
                if let context = report.maxCtx {
                  Text("Context: \(context.formatted()) · Workload: \(report.maxBatch ?? 1)")
                }
                ForEach(Array(report.samples.enumerated()), id: \.offset) { index, sample in
                  Text(
                    "Request \(index + 1): \(sample.inputTokens) input · \(sample.outputTokens) output · \(sample.finishReason)"
                  )
                }
              }.font(.caption).foregroundStyle(.secondary).padding(.top, 10)
            }
          }
        }
      }.padding(32).frame(maxWidth: 900).frame(maxWidth: .infinity)
    }.font(.system(size: 13)).buttonStyle(FlatButtonStyle())
      .task { await model.load() }
      .confirmationDialog(
        "Run a local benchmark?", isPresented: $confirm, titleVisibility: .visible
      ) {
        Button("Run") { if let selected { model.start(selected) } }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text(
          "One warmup and three measured trials use synthetic text, with up to 128 output tokens per request. Other requests can affect results. Model settings and conversation caches remain unchanged."
        )
      }
  }
  private func measure(_ label: String, _ value: Double?, unit: String) -> some View {
    HStack {
      Text(label).foregroundStyle(.secondary)
      Spacer()
      Text(value.map { "\($0.formatted(.number.precision(.fractionLength(2)))) \(unit)" } ?? "—")
        .monospacedDigit()
    }
  }
}
