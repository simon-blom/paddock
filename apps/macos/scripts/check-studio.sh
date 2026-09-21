#!/bin/bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$app_dir/../.." && pwd)"
bash "$app_dir/scripts/check-toolchain.sh"
if [[ $# != 2 || ! -f "$1/config.json" || ! "$2" =~ ^[0-9]+$ || "$2" -lt 1024 || "$2" -gt 65535 ]]; then
  printf 'Usage: check-studio.sh /absolute/Qwen3.8-27B-MLX-4bit unused-port\n' >&2; exit 1
fi
if lsof -nP -iTCP:"$2" -sTCP:LISTEN >/dev/null; then
  printf 'The requested test port is already in use.\n' >&2; exit 1
fi
model_dir="$(cd -- "$1" && pwd)"
cargo build --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop
cargo build --manifest-path "$repo_dir/Cargo.toml" --release -p paddock-runner --no-default-features --features metal
(cd "$repo_dir/studio" && node_modules/.bin/vue-tsc --noEmit -p native-renderer/tsconfig.json && node_modules/.bin/vite build --config native-renderer/vite.config.ts)
export PADDOCK_DATA="$(mktemp -d "${TMPDIR:-/tmp}/paddock-macos-studio.XXXXXX")"
mkdir -p "$PADDOCK_DATA/models" "$PADDOCK_DATA/runners/0.1.6"
ln -s "$model_dir" "$PADDOCK_DATA/models/Qwen3.8-27B-MLX-4bit"
ln -s "$repo_dir/target/release/paddock-runner" "$PADDOCK_DATA/runners/0.1.6/paddock-runner"
export PADDOCK_DESKTOP_TEST_LIBRARY="$repo_dir/target/debug/libpaddock_desktop.dylib"
export PADDOCK_TRANSCRIPT_TEST_ASSETS="$app_dir/.build/studio-renderer"
export PADDOCK_STUDIO_TEST_PORT="$2"
export PADDOCK_STUDIO_INTEGRATION=1
printf 'Synthetic conversation and logs: %s\n' "$PADDOCK_DATA"
# Separate from parallel native tests. The test opens a visible fixture window
# (occluded WebKit suspends frame work), and stops only its own runner PID.
xcrun swift test --package-path "$app_dir" --filter StudioIntegrationTests -Xswiftc -warnings-as-errors
