#!/bin/bash
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$app_dir/../.." && pwd)"
bash "$app_dir/scripts/check-toolchain.sh"
export PADDOCK_MACOS_CONTRACT_DIR="$app_dir/.build/manager-contract"
export PADDOCK_UI_SNAPSHOT_DIR="$app_dir/.build/ui-snapshots"
test_dir="$(mktemp -d /tmp/paddock-macos-test.XXXXXX)"
export PADDOCK_DATA="$test_dir"
export XDG_RUNTIME_DIR="$test_dir/runtime"
mkdir -p "$PADDOCK_DATA" "$XDG_RUNTIME_DIR"
printf 'Isolated test data and admin sockets (no user library): %s\n' "$test_dir"

# Real HTTP router responses, but no socket, runner, models or user DB.
cargo test --manifest-path "$repo_dir/Cargo.toml" -p paddock-manager --test macos_client_contract
cargo test --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop --lib
cargo build --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop
export PADDOCK_DESKTOP_TEST_LIBRARY="$repo_dir/target/debug/libpaddock_desktop.dylib"
xcrun swift format lint --strict --recursive "$app_dir/Sources" "$app_dir/Tests" "$app_dir/Package.swift"
# AppKit tests share NSApplication, windows, menus and the main run loop. Suite
# serialization alone does not prevent another suite from changing that state.
xcrun swift test --package-path "$app_dir" --no-parallel -Xswiftc -warnings-as-errors
# Native popup tracking runs a nested AppKit event loop; isolate it from the
# other rendering suites' hosting-window creation/teardown.
PADDOCK_NATIVE_MENU_TEST=1 xcrun swift test --package-path "$app_dir" \
  --no-parallel --filter DropdownTests -Xswiftc -warnings-as-errors
# Release-check the shipping app. Diagnostic executables were compiled by the
# debug test build above; their @testable imports do not belong in release.
xcrun swift build --package-path "$app_dir" --product PaddockMac \
  --configuration release -Xswiftc -warnings-as-errors
