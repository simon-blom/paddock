// swift-tools-version: 6.3
import PackageDescription

// The app embeds the shared Rust management core, never an inference framework.
// Contracts and pure UI tests can run without loading the bundled Rust library.
let package = Package(
  name: "PaddockMac",
  platforms: [.macOS(.v15)],
  products: [
    .executable(name: "PaddockMac", targets: ["PaddockMac"]),
    // Development-only diagnostic executable. Never linked into the product app.
    .executable(name: "PaddockRendererLab", targets: ["PaddockRendererLab"]),
    .executable(name: "PaddockTranscriptCheck", targets: ["PaddockTranscriptCheck"]),
    .executable(name: "PaddockWorkspaceCheck", targets: ["PaddockWorkspaceCheck"]),
    .executable(name: "PaddockNativeCoreCheck", targets: ["PaddockNativeCoreCheck"]),
    .library(name: "PaddockClient", targets: ["PaddockClient"]),
    .library(name: "PaddockConversationCore", targets: ["PaddockConversationCore"]),
    .library(name: "PaddockUI", targets: ["PaddockUI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/LiYanan2004/MarkdownView.git", from: "3.0.0"),
    .package(url: "https://github.com/lukilabs/beautiful-mermaid-swift.git", from: "1.0.4"),
    .package(url: "https://github.com/sparkle-project/Sparkle.git", exact: "2.10.0"),
  ],
  targets: [
    .target(name: "PaddockClient"),
    // Native state/transport only. No UI framework, JavaScript engine or viewer.
    .target(
      name: "PaddockConversationCore", dependencies: ["PaddockClient"],
      resources: [.process("Resources")]),
    .target(name: "PaddockDesign", resources: [.process("Resources")]),
    .target(name: "PaddockWebAssets"),
    .target(name: "PaddockRendererHost", dependencies: ["PaddockWebAssets"]),
    .target(name: "PaddockTranscript", dependencies: ["PaddockWebAssets", "PaddockClient"]),
    .target(
      name: "PaddockStudio", dependencies: ["PaddockClient", "PaddockConversationCore"],
      resources: [.process("Resources")]),
    .target(
      name: "PaddockNativeMarkdown",
      dependencies: [
        "PaddockDesign",
        .product(name: "MarkdownView", package: "MarkdownView"),
        .product(name: "BeautifulMermaid", package: "beautiful-mermaid-swift"),
      ]),
    .executableTarget(name: "PaddockRendererLab", dependencies: ["PaddockRendererHost"]),
    .executableTarget(
      name: "PaddockTranscriptCheck", dependencies: ["PaddockTranscript", "PaddockClient"]),
    .executableTarget(
      name: "PaddockWorkspaceCheck", dependencies: ["PaddockStudio", "PaddockClient", "PaddockUI"]),
    .executableTarget(
      name: "PaddockNativeCoreCheck", dependencies: ["PaddockConversationCore", "PaddockClient"]),
    .target(
      name: "PaddockUI",
      dependencies: [
        "PaddockClient", "PaddockConversationCore", "PaddockStudio", "PaddockNativeMarkdown",
        "PaddockDesign",
        .product(name: "Sparkle", package: "Sparkle"),
      ],
      resources: [.process("Resources")]),
    .executableTarget(name: "PaddockMac", dependencies: ["PaddockUI"]),
    .testTarget(name: "PaddockClientTests", dependencies: ["PaddockClient"]),
    .testTarget(
      name: "PaddockConversationCoreTests",
      dependencies: ["PaddockConversationCore", "PaddockClient"], resources: [.copy("Fixtures")]),
    .testTarget(
      name: "PaddockUITests", dependencies: ["PaddockUI", "PaddockTranscript"],
      resources: [.copy("Fixtures")]),
    .testTarget(name: "PaddockRendererTests", dependencies: ["PaddockRendererHost"]),
    .testTarget(
      name: "PaddockTranscriptTests", dependencies: ["PaddockTranscript", "PaddockWebAssets"]),
  ],
  swiftLanguageModes: [.v6]
)
