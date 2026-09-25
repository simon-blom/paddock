import PaddockClient
import SwiftUI

/// Reports retain the backend's measurement semantics. End-to-end aggregate
/// throughput is not decode speed; streaming-event gaps are not token latency.
struct BenchmarkResultCard: View {
  let report: BenchmarkReport
  var stacked = false
  var export: () -> Void
  @State private var expanded = false

  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      HStack(alignment: .top, spacing: 16) {
        VStack(alignment: .leading, spacing: 6) {
          Text(report.model).fontWeight(.semibold).lineLimit(2).help(report.model)
          Text(
            Date(timeIntervalSince1970: Double(report.createdAtMs) / 1000),
            format: .dateTime.day().month(.abbreviated).year().hour().minute()
          )
          .font(.system(size: 11)).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
        Button("Export…", systemImage: "square.and.arrow.up", action: export).fixedSize()
          .accessibilityLabel("Export benchmark for \(report.model)")
      }
      Text(
        "\(report.concurrency) concurrent \(report.concurrency == 1 ? "request" : "requests") · \(report.promptWords.formatted()) prompt words"
      )
      .font(.system(size: 12)).foregroundStyle(.secondary)
      .fixedSize(horizontal: false, vertical: true)
      let layout =
        stacked
        ? AnyLayout(VStackLayout(alignment: .leading, spacing: 16))
        : AnyLayout(HStackLayout(alignment: .top, spacing: 20))
      layout {
        metric(
          "Output throughput", report.aggregateOutputTokS, unit: "tok/s",
          context: "Aggregate · end-to-end")
        metric("First token", report.ttftMedianMs.map { $0 / 1000 }, unit: "s", context: "Median")
        metric("Streaming gap", report.streamEventGapP99Ms, unit: "ms", context: "Event gap · p99")
      }
      DisclosureGroup("Measurement details", isExpanded: $expanded) {
        BenchmarkMeasurementDetails(report: report).padding(.top, 12)
      }.accessibilityIdentifier("benchmark-details")
    }.padding(18).frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
  }

  private func metric(_ title: String, _ value: Double?, unit: String, context: String) -> some View
  {
    VStack(alignment: .leading, spacing: 6) {
      Text(title).font(.system(size: 12)).foregroundStyle(.secondary)
      HStack(alignment: .firstTextBaseline, spacing: 5) {
        Text(value.map { $0.formatted(.number.precision(.fractionLength(2))) } ?? "—")
          .font(.system(size: 22, weight: .semibold)).monospacedDigit()
        Text(unit).font(.system(size: 11)).foregroundStyle(.secondary)
      }
      Text(context).font(.system(size: 11)).foregroundStyle(.secondary)
    }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
  }
}

struct BenchmarkMeasurementDetails: View {
  let report: BenchmarkReport
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      row("Measured trials", report.trials.formatted())
      row("Excluded warmups", report.warmups.formatted())
      row("Reply limit", "\(report.outputLimit.formatted()) tokens")
      row("Total output", "\(report.outputTokens.formatted()) tokens")
      row(
        "Measured duration",
        "\(report.wallSeconds.formatted(.number.precision(.fractionLength(2)))) s")
      if let context = report.maxCtx { row("Context", "\(context.formatted()) tokens") }
      if let batch = report.maxBatch { row("Instance concurrency", batch.formatted()) }
      if let version = report.runnerVersion { row("Runner", version) }
      Text(report.cachePolicy).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
      ForEach(Array(report.samples.enumerated()), id: \.offset) { index, sample in
        VStack(alignment: .leading, spacing: 5) {
          Text("Request \(index + 1)").fontWeight(.medium)
          Text(
            "\(sample.inputTokens.formatted()) input · \(sample.outputTokens.formatted()) output · \(sample.finishReason)"
          )
          .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
      }
    }.font(.system(size: 12))
  }
  private func row(_ title: String, _ value: String) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 16) {
      Text(title).foregroundStyle(.secondary)
      Spacer(minLength: 0)
      Text(value).monospacedDigit().multilineTextAlignment(.trailing).textSelection(.enabled)
    }.fixedSize(horizontal: false, vertical: true)
  }
}
