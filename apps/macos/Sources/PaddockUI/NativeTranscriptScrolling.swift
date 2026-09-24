import SwiftUI

/// Scroll observation must not invalidate NativeStudioTranscript's body: that
/// would replace its AppKit hosting root and update every rich-text attachment
/// for each wheel tick. Only this modifier and the Latest control observe the
/// reader's intent, and only discrete transitions publish view changes.
struct NativeTranscriptScrolling: ViewModifier {
  let intent: TranscriptScrollIntent
  let proxy: ScrollViewProxy
  @State private var viewport = CGSize.zero

  func body(content: Content) -> some View {
    content
      .defaultScrollAnchor(.bottom, for: .initialOffset)
      .onScrollPhaseChange { old, phase, context in
        if phase == .tracking || phase == .interacting {
          intent.readingDisclosure = false
          intent.pinned = false
        } else if phase == .idle && old != .idle && !intent.readingDisclosure {
          intent.pinned =
            context.geometry.contentSize.height - context.geometry.visibleRect.maxY
            + context.geometry.contentInsets.bottom <= 24
        }
      }
      .onScrollGeometryChange(for: TranscriptScrollGeometry.self) {
        TranscriptScrollGeometry(
          height: $0.contentSize.height,
          viewport: $0.containerSize,
          offset: $0.contentOffset.y, bottom: $0.visibleRect.maxY - $0.contentInsets.bottom)
      } action: { old, new in
        if viewport != new.viewport { viewport = new.viewport }
        let away = new.height - new.bottom > 24
        if intent.awayFromBottom != away { intent.awayFromBottom = away }
        // Actual message measurements and viewport resizes drive follow, not
        // scroll geometry changes. A user scrolling must never trigger scrollTo.
        guard old.height == new.height, old.viewport == new.viewport else { return }
        if !intent.awayFromBottom && !intent.readingDisclosure {
          intent.pinned = true
        } else if new.offset < old.offset && old.bottom <= old.height + 1 {
          // Keyboard, accessibility and scrollbar jumps have no wheel phase.
          // Ignore initial out-of-range offsets being clamped during layout.
          intent.pinned = false
        }
      }
      .task(id: FollowLayout(viewport: viewport, revision: intent.layoutRevision)) {
        // AppKit resizes the clip and hosting views in separate passes. Resolve
        // the anchor after real row growth or a viewport resize, never for
        // scroll-offset changes.
        await Task.yield()
        if intent.pinned && !Task.isCancelled && viewport.height > 0 {
          proxy.scrollTo("native-bottom", anchor: .bottom)
        }
      }
  }

  private struct FollowLayout: Equatable {
    let viewport: CGSize
    let revision: Int
  }
}

struct NativeTranscriptLatestControl: View {
  let intent: TranscriptScrollIntent
  let columnWidth: CGFloat
  let composerHeight: CGFloat
  let proxy: ScrollViewProxy

  var body: some View {
    if !intent.pinned && intent.awayFromBottom {
      NativeTranscriptLatestButton(columnWidth: columnWidth, composerHeight: composerHeight) {
        intent.readingDisclosure = false
        intent.pinned = true
        proxy.scrollTo("native-bottom", anchor: .bottom)
      }
    }
  }
}

private struct TranscriptScrollGeometry: Equatable {
  let height: CGFloat
  let viewport: CGSize
  let offset: CGFloat
  let bottom: CGFloat
}
