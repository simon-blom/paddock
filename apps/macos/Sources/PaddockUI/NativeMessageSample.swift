import Foundation
import PaddockStudio

/// Used only by the explicitly labelled Samples panel and native UI tests.
/// These illustrative measurements never enter a chat, ledger or benchmark.
enum NativeMessageSample {
  static let text = """
    ## A native response

    The response has a **model identity**, selectable Markdown and a footer with its recorded statistics.

    - Select across these paragraphs and list rows.
    - Open run details beside the copy button below.

    ```swift
    let author = response.model
    let latency = response.usage.totalTime
    ```

    The footer uses total request time, while tokens per second describes the answer stream.
    """

  static func message(text: String = text, streaming: Bool = false) throws
    -> StudioState.NativeTranscript.Message
  {
    let fixture: [String: Any] = [
      "id": "sample", "role": "assistant", "text": text,
      "reasoning":
        "Check the run's recorded author and measured usage, not the current composer selection.",
      "model": "qwen3.8-27b-mlx", "streaming": streaming, "stopped": false, "error": "",
      "incomplete": false,
      "chrome": [
        "modelName": "Qwen 3.8 27B", "vendor": "Alibaba", "spec": "MTP",
        "footer": "120 tokens · 60 tok/s · 5.6s",
        "footerHint":
          "Illustrative statistics, not a benchmark. Total time from send to done: 5.6s; first token: 350ms.",
        "thinkingLabel": "Thought for 3.0s", "thinkingMeta": "200 tokens · 67 tok/s",
        "cutNote": "Reply hit the max-token limit.",
        "promptText": "Explain the native message layout clearly.",
        "sections": [
          [
            "id": "provenance", "title": "Provenance",
            "rows": [
              ["label": "Model", "value": "qwen3.8-27b-mlx"],
              ["label": "Speculation", "value": "MTP"],
              ["label": "System prompt", "value": "Custom"],
              ["label": "Sampling", "value": "temp 0.8 · top-p 0.9 · top-k off"],
            ],
          ],
          [
            "id": "metrics", "title": "Metrics (illustrative)",
            "rows": [
              ["label": "Tokens", "value": "80 in · 120 out · 200 reasoning"],
              ["label": "Speed", "value": "60 tok/s · TTFT 350ms · 5.6s total"],
            ],
          ],
          [
            "id": "gpu", "title": "GPU environment (illustrative)",
            "rows": [["label": "Device", "value": "Apple M5 Pro"]],
          ],
        ],
      ],
    ]
    return try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.self,
      from: JSONSerialization.data(withJSONObject: fixture))
  }
}
