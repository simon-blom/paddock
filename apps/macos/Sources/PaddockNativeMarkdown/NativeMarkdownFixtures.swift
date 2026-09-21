import Foundation

public enum NativeMarkdownFixtures {
  public static let showcase = #"""
    # A native reading experience

    This is **MarkdownView**, with *native text*, `inline code`, tables, mathematics and a Swift-rendered diagram.

    ## Code

    ```swift
    struct Message: Identifiable {
        let id: UUID
        let content: String
    }
    let reply = Message(id: UUID(), content: "Hello, Mac.")
    ```

    ## Mathematics

    Inline: $E = mc^2$. Display:

    $$
    \int_0^1 x^2\,dx = \frac{1}{3}
    $$

    ## Diagram

    ```mermaid
    flowchart LR
      A[Native composer] --> B[Rust core]
      B --> C[Model]
      C --> D[MarkdownView]
      D --> E[BeautifulMermaid]
    ```

    ## Table and lists

    | Component | Ownership | Rendering |
    | :--- | :--- | :--- |
    | Window and composer | Swift | Native |
    | Markdown and diagrams | Swift | Native trial |
    | Chat persistence | Rust | Shared SQLite |

    - **Selection:** try copying a paragraph and the code block.
    - **Streaming:** use Replay to watch unfinished syntax settle.
      1. Open the diagram's source.
      2. Switch the app's renderer to compare the same conversation.

    > The native trial preserves source when a diagram cannot render.

    [MarkdownView project](https://github.com/LiYanan2004/MarkdownView)
    """#
  public static let diagrams = #"""
    ## Sequence
    ```mermaid
    sequenceDiagram
      participant User
      participant App
      participant Model
      User->>App: Send message
      App->>Model: Start response
      Model-->>App: Stream tokens
      App-->>User: Render natively
    ```

    ## State
    ```mermaid
    stateDiagram-v2
      [*] --> Ready
      Ready --> Streaming: Send
      Streaming --> Ready: Complete
    ```

    ## Class
    ```mermaid
    classDiagram
      class Message {
        +String id
        +String text
      }
      class Conversation {
        +String title
      }
      Conversation --> Message
    ```

    ## Entity relationship
    ```mermaid
    erDiagram
      CONVERSATION ||--o{ MESSAGE : contains
      MESSAGE {
        string id
        string text
      }
    ```

    ## XY chart
    ```mermaid
    xychart-beta
      x-axis [A, B, C]
      y-axis "Example values" 0 --> 10
      bar [3, 7, 5]
    ```

    ## Unsupported syntax is visible, not discarded
    ```mermaid
    pie title Example
      "A" : 40
      "B" : 60
    ```
    """#
  public static let long = (1...70).map { index in
    "## Section \(index)\n\nA longer **native transcript** tests scrolling, selection and retained parser state. Resize the window while this fixture is streaming.\n\n- First item\n- Second item with `code`\n"
  }.joined(separator: "\n")
}
