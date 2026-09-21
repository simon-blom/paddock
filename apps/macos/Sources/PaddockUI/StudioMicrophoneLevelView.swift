import PaddockStudio
import SwiftUI

/// Only these nine bars tick, not the composer, transcript or runtime snapshot.
/// Signal/envelope come from capture; there are no random bars or idle pulse.
struct StudioMicrophoneLevelView: View {
  let meter: StudioMicrophoneMeter
  var listening = true
  var body: some View {
    TimelineView(.animation(minimumInterval: 1.0 / 60, paused: !listening)) { _ in
      let levels = listening ? meter.levels() : []
      Canvas { context, _ in
        for i in 0..<StudioMicrophoneMeter.bandCount {
          let level = levels.indices.contains(i) ? levels[i] : 0
          let height = Self.height(level)
          let rect = CGRect(x: Double(i) * 5, y: (18 - height) / 2, width: 3, height: height)
          context.fill(
            Path(roundedRect: rect, cornerRadius: 1.5),
            with: .color(.primary.opacity(0.42 + level * 0.58)))
        }
      }
    }.frame(width: 43, height: 18).accessibilityHidden(true)
  }
  nonisolated static func height(_ level: Double) -> Double {
    18 * (0.16 + min(1, max(0, level.isFinite ? level : 0)) * 0.84)
  }
}
