#!/bin/bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$app_dir/../.." && pwd)"
cargo build --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop
xcrun swift build --package-path "$app_dir" --product PaddockNativeCoreCheck -Xswiftc -warnings-as-errors
bin_dir="$(xcrun swift build --package-path "$app_dir" --show-bin-path)"
bundle="$app_dir/.build/PaddockNativeCoreCheck.app"
if pgrep -f "$bundle/Contents/MacOS/PaddockNativeCoreCheck" >/dev/null; then
  printf 'A native core check is already running.\n' >&2; exit 1
fi
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Frameworks"
cp "$bin_dir/PaddockNativeCoreCheck" "$bundle/Contents/MacOS/"
cp "$repo_dir/target/debug/libpaddock_desktop.dylib" "$bundle/Contents/Frameworks/"
cp "$repo_dir/packaging/macos/Info.plist" "$bundle/Contents/Info.plist"
plutil -replace CFBundleExecutable -string PaddockNativeCoreCheck "$bundle/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string io.truespar.paddock.native-core-check "$bundle/Contents/Info.plist"
codesign --force --sign - "$bundle/Contents/Frameworks/libpaddock_desktop.dylib"
codesign --force --sign - "$bundle"
native_check_data="$(mktemp -d /tmp/paddock-native-core-check.XXXXXX)"
printf 'Synthetic check data: %s\n' "$native_check_data"
PADDOCK_DATA="$native_check_data" "$bundle/Contents/MacOS/PaddockNativeCoreCheck"
