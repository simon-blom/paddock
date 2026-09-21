#!/bin/bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$app_dir/../.." && pwd)"
bash "$app_dir/scripts/check-toolchain.sh"
cargo build --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop
(cd "$repo_dir/studio" && node_modules/.bin/vue-tsc --noEmit -p native-workspace/tsconfig.json && node_modules/.bin/vite build --config native-workspace/vite.config.ts --logLevel error)
xcrun swift build --package-path "$app_dir" --product PaddockWorkspaceCheck -Xswiftc -warnings-as-errors
node "$repo_dir/studio/native-workspace/fixtures.mjs" "$app_dir/.build/workspace-fixtures"
bin_dir="$(xcrun swift build --package-path "$app_dir" --show-bin-path)"
bundle="$app_dir/.build/PaddockWorkspaceCheck.app"
if pgrep -f "$bundle/Contents/MacOS/PaddockWorkspaceCheck" >/dev/null; then
  printf 'A workspace check is already running.\n' >&2; exit 1
fi
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Frameworks" "$bundle/Contents/Resources"
# Native transcript checks now render the real design/Markdown targets too.
# Package their SwiftPM assets exactly as the development app does.
for resource_bundle in "$bin_dir/"*.bundle; do
  [[ -d "$resource_bundle" ]] || continue
  cp -R "$resource_bundle" "$bundle/Contents/Resources/"
done
cp "$bin_dir/PaddockWorkspaceCheck" "$bundle/Contents/MacOS/"
cp "$repo_dir/target/debug/libpaddock_desktop.dylib" "$bundle/Contents/Frameworks/"
cp "$repo_dir/packaging/macos/Info.plist" "$bundle/Contents/Info.plist"
plutil -replace CFBundleExecutable -string PaddockWorkspaceCheck "$bundle/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string io.truespar.paddock.workspace-check "$bundle/Contents/Info.plist"
codesign --force --sign - "$bundle/Contents/Frameworks/libpaddock_desktop.dylib"
codesign --force --sign - "$bundle"
workspace_check_data="$(mktemp -d /tmp/paddock-workspace-check.XXXXXX)"
printf 'Synthetic check data: %s\n' "$workspace_check_data"
PADDOCK_DATA="$workspace_check_data" "$bundle/Contents/MacOS/PaddockWorkspaceCheck" "$app_dir/.build/studio-workspace" "$app_dir/.build/workspace-fixtures"
