# Paddock for macOS

A native SwiftUI/AppKit app for Apple Silicon. It is the manager and the Studio
in one process: there is no separate manager to install, no management web
server and no access-key prompt. The app embeds the same Rust management core
the web Studio uses, and starts the Metal runner as a child process when you
start a model.

This is a development foundation, not a finished product. Downloads, update,
notarization and distribution are not done, and nothing here is a performance
claim.

## What is in the bundle

- `Paddock.app`, executable `Paddock`, bundle ID `io.truespar.paddock`.
- The Rust management library in `Contents/Frameworks`. It owns the registry,
  readiness checks, the SQLite store, credentials and runner supervision. Swift
  talks to it through a private, versioned C ABI
  (`crates/paddock-desktop/include/paddock_desktop.h`); there is no second
  database and no inference framework in Swift.
- An optimized Metal runner in `Contents/Helpers`. New endpoints bind
  `127.0.0.1`.
- The content viewers (Lector, Scriptor, Traverse) and isolated HTML/SVG
  previews run in WebKit. The transcript, composer, Markdown, code, math,
  Mermaid and audio are native.

Closing the window keeps the app in the menu bar. Quit closes the core; runner
endpoints that were started independently are left running.

## Where data lives

The library is `~/Library/Application Support/Paddock`. An existing `~/paddock`
library from the command-line tools is used instead when it is the only one
present. `PADDOCK_DATA` overrides both and exists for isolated checks, not for
normal launches. The build script never moves or deletes models or data, and an
app bundle never becomes a data directory.

## Hardware

The Metal backend admits every unified-memory Apple Silicon GPU, Apple7 (M1)
through Apple10 (M5). Support below M5 is experimental: what has been qualified
on M5 does not carry over, and coverage on M1, M3 and M4 is limited to the GPU
test suite and a short same-weights greedy comparison on one model. M2 sits
between two measured families and has not been run.

## Build

Xcode 26.6 / Apple Swift 6.3.3 or newer, Swift 6 language mode, macOS 15 or
newer. `apps/macos/scripts/check-toolchain.sh` verifies the toolchain. The Swift
package depends on MarkdownView and beautiful-mermaid-swift; everything else is
in this repository.

From `apps/macos`:

```sh
bash ./scripts/check.sh
bash ./scripts/build-app.sh
open .build/Paddock.app
```

`build-app.sh --install` installs to `/Applications/Paddock.app` instead. Quit
the app and its runners first; the script refuses to overwrite a running bundle.

`build-app.sh` builds the Rust library, the Metal runner and
`studio/native-workspace` (with the Studio's locked dependencies), then signs a
development bundle. It picks the single valid Apple Development identity, or
takes `PADDOCK_CODESIGN_IDENTITY` when there are several. Without an identity,
ad-hoc signing needs an explicit `PADDOCK_ALLOW_ADHOC=1`, and the Keychain will
then ask again after every rebuild.

`swift build` on its own builds the UI without the Rust library. Launching that
executable reports a packaging error; it does not fall back to a network
manager.

## Checks

| Command (from `apps/macos`) | What it covers |
| --- | --- |
| `bash ./scripts/check.sh` | Format, Swift tests, an optimized build with warnings as errors, and a real Swift-to-Rust open/read/close/reopen against a fresh temporary data directory. Never opens your library and never loads a model. |
| `swift test` | The pure Swift cases only; integration cases that need the Rust library skip themselves. |
| `bash ./scripts/check-workspace.sh` | The Studio workspace, offscreen, against an isolated store and synthetic streams. |
| `bash ./scripts/check-native-core.sh` | Conversation state and transport, without UI. |
| `bash ./scripts/check-studio.sh <model-dir> <port>` | Opt-in: a real local chat against installed weights. It references the weights without copying them and stops its own endpoint afterwards. A functional check, not a benchmark. |

The rich-renderer check needs a real AppKit event loop and an unlocked,
foreground session, because an occluded WebKit view suspends mounting:

```sh
swift run -c release PaddockTranscriptCheck .build/studio-renderer
```

## Rendering Lab

A separate development-only app that loads the Studio's real renderer
dependencies and vendored WASM in WKWebView. It opens no product database and
starts no inference.

```sh
bash ./scripts/build-renderer-lab.sh
open .build/PaddockRenderingLab.app
```

Run all, then look through the fixture tabs. For process-memory measurements
quit the lab and drive it from the repository root with
`node studio/renderer-lab/profile.mjs <suite>`; the runner samples physical
footprint, disables screenshots and stops only the lab it started. Diagnostic
flags (`--trace-memory`, `--heap-snapshot`, `--trace-layers`) are intrusive: runs
that use them are for attribution and are never a latency or memory bar.

## Layout

| Path | What it is |
| --- | --- |
| `Sources/PaddockMac` | The app executable. |
| `Sources/PaddockUI` | Windows, Settings, the model browser, endpoints, cloud providers. |
| `Sources/PaddockStudio`, `PaddockTranscript`, `PaddockNativeMarkdown` | The chat workspace and its native rendering. |
| `Sources/PaddockConversationCore` | Conversation state and Responses transport. No UI framework. |
| `Sources/PaddockClient` | The Swift side of the C ABI. |
| `Sources/PaddockDesign` | Palette, controls, shared resources. |
| `Sources/PaddockRendererHost`, `PaddockWebAssets` | The WebKit host for the content viewers. |
| `Sources/*Check`, `Sources/PaddockRendererLab` | Development-only executables. Never linked into the app. |

The build and check scripts are in `apps/macos/scripts`.

Provider and maker marks in `Sources/PaddockUI/Resources` ship with their
provenance, the Simple Icons licence and disclaimer, and per-icon source
metadata. Nothing is fetched at run time.
