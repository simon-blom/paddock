import PaddockStudio
import SwiftUI

enum NativeCompareLayout {
  static let gap: CGFloat = 12
  static func laneWidth(available: CGFloat, count: Int) -> CGFloat {
    max(240, (max(0, available) - gap * CGFloat(max(0, count - 1))) / CGFloat(max(1, count)))
  }
}

/// Like ChatThread.vue: parallel columns, 240-point readable floor, horizontal
/// overflow, no wrapping or per-lane vertical scrollers. One group branch switch.
struct NativeCompareBlock: View {
  let block: StudioState.NativeTranscript.Block
  let transcript: StudioState.NativeTranscript
  let width: CGFloat
  var workspace: StudioWorkspace?

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let first = block.messages.first, let branch = first.actions?.branch {
        HStack(spacing: 8) {
          branchButton(
            "Previous comparison", icon: "chevron.left", id: branch.previous, message: first)
          Text("\(branch.index) / \(branch.count)").monospacedDigit()
            .accessibilityLabel("Comparison \(branch.index) of \(branch.count)")
          branchButton("Next comparison", icon: "chevron.right", id: branch.next, message: first)
        }.font(.system(size: 11)).foregroundStyle(.secondary).frame(maxWidth: .infinity)
      }
      PaddockScrollView(.horizontal) {
        HStack(alignment: .top, spacing: NativeCompareLayout.gap) {
          ForEach(block.messages) { message in
            VStack(alignment: .leading, spacing: 8) {
              NativeCompareHeader(message: message)
              NativeStudioMessage(
                message: message, workspace: workspace,
                target: StudioMessageTarget(transcript: transcript, messageId: message.id),
                inLane: true
              ).equatable()
            }
            .padding(.horizontal, 14).padding(.top, 10).padding(.bottom, 4)
            .frame(
              width: NativeCompareLayout.laneWidth(available: width, count: block.messages.count),
              alignment: .topLeading
            )
            .background(
              PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
            )
            .overlay(
              RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
                .strokeBorder(PaddockStyle.border, lineWidth: 1).allowsHitTesting(false)
            )
            .accessibilityIdentifier("native-compare-lane-\(message.id)")
          }
        }.padding(.bottom, 4)
      }.fixedSize(horizontal: false, vertical: true)
        .scrollBounceBehavior(.basedOnSize, axes: .horizontal)
    }.accessibilityIdentifier("native-compare-group-\(block.id)")
  }

  private func branchButton(
    _ label: String, icon: String, id: String?, message: StudioState.NativeTranscript.Message
  ) -> some View {
    Button(label, systemImage: icon) {
      guard let workspace, let id,
        let target = StudioMessageTarget(transcript: transcript, messageId: message.id)
      else { return }
      Task { await workspace.messageAction("branch", target: target, branch: id) }
    }.labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
      .disabled(
        id == nil || workspace?.ready != true || workspace?.busy == true
          || workspace?.hasMessageEdit == true
      )
      .help(label)
  }
}

struct NativeCompareHeader: View {
  let message: StudioState.NativeTranscript.Message
  var body: some View {
    ViewThatFits(in: .horizontal) {
      HStack(spacing: 8) {
        NativeMessageHeader(message: message, inLane: true)
        badges.fixedSize()
      }
      // Very narrow lanes retain both maker/name and every badge. No overlap
      // or disappearing labels; the comparison columns themselves never wrap.
      VStack(alignment: .leading, spacing: 6) {
        NativeMessageHeader(message: message, inLane: true)
        badges
      }
    }.frame(maxWidth: .infinity, alignment: .leading)
      .padding(.bottom, 8)
      .overlay(alignment: .bottom) {
        Color(nsColor: PaddockStyle.nsColor("borderSubtle")).frame(height: 1).allowsHitTesting(
          false)
      }
      .accessibilityIdentifier("native-compare-header-\(message.id)")
  }
  private var badges: some View {
    HStack(spacing: 8) {
      if let tools = message.chrome?.tools, !tools.isEmpty {
        badge(
          "tools", icon: "powerplug", help: "Ran with MCP tools: \(tools.joined(separator: ", "))")
      }
      if message.chrome?.fastest == true {
        badge(
          "fastest",
          help:
            "Finished first on this clip - every model had the same audio to transcribe and nothing else was using the GPU"
        )
      }
      if message.contended == true {
        badge(
          "shared GPU",
          help:
            "These models ran at the same time on one GPU, so these speeds include contention, not standalone model performance"
        )
      }
    }
  }
  private func badge(_ text: String, icon: String? = nil, help: String) -> some View {
    HStack(spacing: 3) {
      if let icon { Image(systemName: icon).font(.system(size: 11)) }
      Text(text).font(.system(size: 10))
    }.foregroundStyle(.secondary).padding(.horizontal, 7).padding(.vertical, 2)
      .overlay(
        Capsule().strokeBorder(Color(nsColor: PaddockStyle.nsColor("borderSubtle")), lineWidth: 1)
      )
      .help(help).accessibilityIdentifier("native-compare-badge-\(text)-\(message.id)")
  }
}
