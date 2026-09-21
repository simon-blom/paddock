import Charts
import PaddockClient
import SwiftUI

struct GPUHistoryView: View {
  let history: GPUHistory
  @State private var metric: GPUHistoryMetric = .allocations
  @State private var seconds = 300
  @State private var selectedDate: Date?

  private var metrics: [GPUHistoryMetric] {
    GPUHistoryMetric.allCases.filter { key in history.points.contains { $0.values[key] != nil } }
  }
  private var active: GPUHistoryMetric {
    metrics.contains(metric) ? metric : (metrics.first ?? .allocations)
  }
  private var points: [GPUHistoryPoint] {
    guard let end = history.points.last?.date else { return [] }
    return history.points.filter { $0.date >= end.addingTimeInterval(-Double(seconds)) }
  }
  // Swift Charts joins missing values unless each uninterrupted run has its own
  // series. Split on missing sensors too, not just process/sleep boundaries.
  private var plotted: [(point: GPUHistoryPoint, value: Double, run: Int)] {
    var run = 0
    var previous: UInt64?
    return points.compactMap { point in
      if point.segment != previous { run += 1 }
      previous = point.segment
      guard let value = point.values[active] else {
        run += 1
        return nil
      }
      return (point, value, run)
    }
  }
  private var selected: GPUHistoryPoint? {
    guard let selectedDate else { return nil }
    return points.min {
      abs($0.date.timeIntervalSince(selectedDate)) < abs($1.date.timeIntervalSince(selectedDate))
    }
  }

  var body: some View {
    if !metrics.isEmpty {
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Picker("Metric", selection: $metric) {
            ForEach(metrics) { Text($0.title).tag($0) }
          }.labelsHidden().fixedSize()
          Spacer()
          Picker("History window", selection: $seconds) {
            Text("1 min").tag(60)
            Text("5 min").tag(300)
            Text("15 min").tag(900)
          }.labelsHidden().fixedSize()
        }.controlSize(.small)
        Chart {
          ForEach(plotted, id: \.point.id) { item in
            LineMark(
              x: .value("Time", item.point.date), y: .value(active.unit, item.value),
              series: .value("Segment", item.run)
            )
            .foregroundStyle(Color.primary).lineStyle(StrokeStyle(lineWidth: 1.5))
            .interpolationMethod(.linear)
          }
          if let latest = plotted.last {
            PointMark(x: .value("Time", latest.point.date), y: .value(active.unit, latest.value))
              .foregroundStyle(Color.primary).symbolSize(12)
          }
          if let selected {
            RuleMark(x: .value("Time", selected.date)).foregroundStyle(.secondary)
          }
        }
        .chartXSelection(value: $selectedDate)
        .chartYScale(domain: 0...max(1, (plotted.map(\.value).max() ?? 0) * 1.1))
        .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
        .chartYAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
        .frame(height: 130)
        .accessibilityLabel("\(active.title) history in \(active.unit)")
        HStack {
          Text((selected ?? points.last)?.date ?? Date(), style: .time)
          Spacer()
          if let value = (selected ?? points.last)?.values[active] {
            Text("\(value.formatted(.number.precision(.fractionLength(1)))) \(active.unit)")
          } else {
            Text("No measurement")
          }
        }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
      }
    }
  }
}
