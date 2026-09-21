import PaddockClient
import SwiftUI

struct GPUMetricsView: View {
  let snapshot: GPUSnapshot?
  let runners: [RunnerInfo]
  var history = GPUHistory()
  var stale = false

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 16) {
        Text("GPU").font(.headline)
        if stale {
          Text("Measurements paused").font(.caption).foregroundStyle(.secondary)
        }
        if let snapshot, snapshot.available {
          ForEach(snapshot.gpus) { gpu in
            VStack(alignment: .leading, spacing: 12) {
              Text(gpu.name).font(.subheadline.weight(.semibold))
              if let metal = gpu.metal {
                row("Thermal pressure", metal.thermalPressure?.capitalized ?? "Unavailable")
                row("Memory pressure", metal.memoryPressure?.capitalized ?? "Not reported")
                  .help(
                    "macOS reports memory-pressure changes. No reading is assumed before the first OS event."
                  )
                row("Unified memory", bytes(metal.unifiedMemoryTotal))
                row("Metal working set", bytes(metal.recommendedWorkingSet))
                  .help(
                    "Apple’s recommended working-set size, shared with other applications. Not dedicated VRAM."
                  )
                if let allocated = snapshot.metalAllocatedBytes {
                  row("Paddock allocations", bytes(allocated))
                  ProgressView(
                    value: Double(allocated), total: Double(max(1, metal.recommendedWorkingSet))
                  )
                  .tint(.primary).accessibilityLabel(
                    "Paddock allocations relative to Metal working set")
                }
              } else {
                if let used = gpu.memUsed { row("GPU memory", bytes(used)) }
                if let util = gpu.utilGpu {
                  row("Utilization", "\(util.formatted(.number.precision(.fractionLength(0))))%")
                }
                if let power = gpu.powerW {
                  row("Power", "\(power.formatted(.number.precision(.fractionLength(1)))) W")
                }
                if let temp = gpu.tempC {
                  row("Temperature", "\(temp.formatted(.number.precision(.fractionLength(0))))°C")
                }
              }
            }.padding(12).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 10))
          }
          GPUHistoryView(history: history)
          ForEach(snapshot.reconciliation?.runners ?? []) { runner in
            VStack(alignment: .leading, spacing: 10) {
              Text(
                runners.first(where: { $0.pid == runner.pid && $0.port == runner.port })?.title
                  ?? "Runner · \(runner.port)"
              )
              .font(.subheadline.weight(.medium)).lineLimit(2)
              if let metal = runner.metal {
                row("Metal allocations", bytes(metal.allocatedBytes))
                row(
                  "GPU time · cumulative",
                  "\(metal.gpuSecondsTotal.formatted(.number.precision(.fractionLength(2)))) s"
                )
                .help(
                  "Sum of completed command-buffer GPU spans since runner start. Spans may overlap; this is not device utilization."
                )
                if let ms = metal.lastCommandMs {
                  row(
                    "Last GPU command", "\(ms.formatted(.number.precision(.fractionLength(2)))) ms")
                }
              } else if snapshot.gpus.contains(where: { $0.metal != nil }) {
                Text("Measurements unavailable · restart with the updated runner")
                  .font(.caption).foregroundStyle(.secondary)
              }
              if let engine = runner.engine {
                row(
                  engine.phase.capitalized,
                  "\(engine.tokS.formatted(.number.precision(.fractionLength(1)))) tok/s")
                if engine.kvTotal > 0 { row("KV blocks", "\(engine.kvUsed) / \(engine.kvTotal)") }
              }
            }.padding(12).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 10))
          }
        } else {
          Text(snapshot == nil ? "Loading measurements…" : "GPU measurements unavailable")
            .foregroundStyle(.secondary)
        }
      }.padding(16).background(PaddockScrollStyle())
    }.frame(width: 340, height: panelHeight)
      .studioPopoverSurface()
  }

  private var panelHeight: CGFloat {
    min(
      540,
      270 + (history.points.isEmpty ? 0 : 210) + CGFloat(
        snapshot?.reconciliation?.runners.count ?? 0) * 156)
  }

  private func row(_ label: String, _ value: String) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 12) {
      Text(label).foregroundStyle(.secondary)
      Spacer(minLength: 8)
      Text(value).monospacedDigit()
    }.font(.caption)
  }

  private func bytes(_ value: UInt64?) -> String {
    guard let value else { return "—" }
    return "\((Double(value) / 1_073_741_824).formatted(.number.precision(.fractionLength(2)))) GiB"
  }
}
