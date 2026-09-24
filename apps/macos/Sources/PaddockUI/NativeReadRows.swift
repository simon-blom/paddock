import PaddockConversationCore
import SwiftUI

struct NativeReadQuestionRow: View {
  @Binding var question: ReadQuestion
  let onDuplicate: () -> Void
  let onMove: (Int) -> Void
  let onRemove: () -> Void
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        TextField("Question ID", text: $question.questionID).textFieldStyle(
          StudioPopoverFieldStyle()
        )
        .accessibilityLabel("Question ID")
        Picker("Type", selection: $question.kind) {
          ForEach(ReadQuestion.Kind.allCases, id: \.self) { Text($0.title).tag($0) }
        }.labelsHidden().fixedSize()
        Menu {
          Button("Move up", systemImage: "arrow.up") { onMove(-1) }
          Button("Move down", systemImage: "arrow.down") { onMove(1) }
          Button("Duplicate", systemImage: "plus.square.on.square", action: onDuplicate)
          Button("Remove", systemImage: "trash", role: .destructive, action: onRemove)
        } label: {
          Image(systemName: "ellipsis").frame(width: 24, height: 28)
        }
        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().accessibilityLabel(
          "Question actions")
      }
      TextField("What should the model decide?", text: $question.instructions, axis: .vertical)
        .textFieldStyle(StudioPopoverFieldStyle()).lineLimit(2...5)
      switch question.kind {
      case .noul:
        TextField("Yes means (optional)", text: $question.yesMeans).textFieldStyle(
          StudioPopoverFieldStyle())
        TextField("No means (optional)", text: $question.noMeans).textFieldStyle(
          StudioPopoverFieldStyle())
      case .choice:
        ForEach($question.options) { $option in
          HStack {
            TextField("Option", text: $option.name).textFieldStyle(StudioPopoverFieldStyle()).frame(
              maxWidth: 140)
            TextField("Description", text: $option.description).textFieldStyle(
              StudioPopoverFieldStyle())
            Button {
              question.options.removeAll { $0.id == option.id }
            } label: {
              Image(systemName: "minus.circle")
            }
            .buttonStyle(QuietButtonStyle()).accessibilityLabel("Remove option")
          }
        }
        Button("Add option", systemImage: "plus") { question.options.append(.init()) }
          .buttonStyle(QuietButtonStyle()).disabled(question.options.count >= 26)
      case .score:
        ForEach($question.levels) { $level in
          HStack {
            TextField("Level", text: $level.name).textFieldStyle(StudioPopoverFieldStyle())
            Button {
              question.levels.removeAll { $0.id == level.id }
            } label: {
              Image(systemName: "minus.circle")
            }
            .buttonStyle(QuietButtonStyle()).accessibilityLabel("Remove level")
          }
        }
        Button("Add level", systemImage: "plus") { question.levels.append(.init()) }
          .buttonStyle(QuietButtonStyle()).disabled(question.levels.count >= 26)
      }
      if let validation = question.validation {
        Text(validation).font(.caption).foregroundStyle(PaddockStyle.caution)
      }
    }.padding(12).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
  }
}

struct NativeReadAnswerView: View {
  let question: ReadQuestion
  let answer: ReadResponse.Answer
  let diagnostic: ReadResponse.Diagnostics.Question?
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(question.questionID).font(.caption).foregroundStyle(.secondary)
      Text(question.instructions).textSelection(.enabled)
      HStack(alignment: .firstTextBaseline) {
        Text(answer.label).font(.title3.weight(.semibold))
        if let score = answer.score {
          Text(score, format: .number.precision(.fractionLength(2))).monospacedDigit()
        }
        Spacer()
        Text("Confidence \(answer.confidence, specifier: "%.2f")").foregroundStyle(.secondary)
      }
      ForEach(Array(answer.bars.enumerated()), id: \.offset) { _, bar in
        probability(bar.name, bar.probability)
      }
      // Conditional label probabilities and outside mass are different
      // quantities; never combine them into a normalized stacked chart.
      probability("Outside the options", answer.outside)
      if answer.outside > 0.5 {
        Text("Most probability is outside these options.").font(.caption).foregroundStyle(
          PaddockStyle.caution)
      }
      if let diagnostic {
        HStack {
          Text("Entropy \(diagnostic.entropy, specifier: "%.3f")")
          Spacer()
          Text("Agreement \(answer.agreement * 100, specifier: "%.0f")%")
        }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
        if let reads = diagnostic.reads, reads.count > 1 {
          DisclosureGroup("Individual reads") {
            ForEach(Array(reads.enumerated()), id: \.offset) { i, read in
              HStack {
                Text("\(i + 1)").frame(width: 20, alignment: .leading)
                Text(read.pick)
                Spacer()
                Text(
                  "\(read.confidence, specifier: "%.2f") confidence · \(read.entropy, specifier: "%.3f") entropy"
                )
              }.font(.caption).monospacedDigit()
            }
          }
        }
      }
    }.padding(14).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 8))
  }
  private func probability(_ name: String, _ value: Double) -> some View {
    HStack(spacing: 12) {
      Text(name).frame(width: 130, alignment: .leading).lineLimit(2)
      ProgressView(value: min(1, max(0, value))).tint(.primary).accessibilityLabel(name)
      Text(value, format: .percent.precision(.fractionLength(1))).monospacedDigit().frame(
        width: 58, alignment: .trailing)
    }.font(.caption)
  }
}
