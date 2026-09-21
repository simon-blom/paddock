#!/bin/bash
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$app_dir/../.." && pwd)"
bash "$app_dir/scripts/check-toolchain.sh"
# A stable identity keeps Keychain authorization valid across development
# rebuilds. Ad-hoc designated requirements are tied to each binary's hash.
signing_identity="${PADDOCK_CODESIGN_IDENTITY:-}"
if [[ -z "$signing_identity" ]]; then
  development_identities="$(security find-identity -v -p codesigning | awk '/"Apple Development:/ { print $2 }')"
  if [[ -n "$development_identities" && "$development_identities" != *$'\n'* ]]; then
    signing_identity="$development_identities"
  elif [[ "${PADDOCK_ALLOW_ADHOC:-0}" == 1 ]]; then
    signing_identity=-
    printf 'Ad-hoc signing explicitly selected: Keychain may require authorization after every rebuild.\n' >&2
  else
    printf 'Choose a stable signing identity with PADDOCK_CODESIGN_IDENTITY (or explicitly set PADDOCK_ALLOW_ADHOC=1 for isolated tests).\n' >&2
    exit 1
  fi
fi
bundle="$app_dir/.build/Paddock.app"
case "${1:-}" in
  '') ;;
  --install) bundle="/Applications/Paddock.app" ;;
  *) printf 'Usage: build-app.sh [--install]\n' >&2; exit 1 ;;
esac
if [[ $# -gt 1 ]]; then
  printf 'Usage: build-app.sh [--install]\n' >&2; exit 1
fi
# Never replace executable pages under an app the user is still running.
if pgrep -f '/Paddock[.]app/Contents/(MacOS/Paddock|Helpers/paddock-runner)' >/dev/null; then
  printf 'Quit Paddock and stop its bundled runners before rebuilding the app.\n' >&2
  exit 1
fi
if [[ ! -x "$repo_dir/studio/node_modules/.bin/vite" ]]; then
  printf 'Install the locked Studio dependencies with npm ci in studio/ first.\n' >&2
  exit 1
fi
(cd "$repo_dir/studio" && node_modules/.bin/vue-tsc --noEmit -p native-workspace/tsconfig.json && node_modules/.bin/vite build --config native-workspace/vite.config.ts --logLevel error)
cargo build --manifest-path "$repo_dir/Cargo.toml" -p paddock-desktop
# Optimized native Metal inference and in-process CoreGraphics PDF rendering.
# No CUDA/PDFium archive or external runner install is needed on macOS.
cargo build --manifest-path "$repo_dir/Cargo.toml" --release -p paddock-runner --no-default-features --features metal
xcrun swift build --package-path "$app_dir"
bin_dir="$(xcrun swift build --package-path "$app_dir" --show-bin-path)"
mkdir -p "$bundle/Contents/MacOS"
mkdir -p "$bundle/Contents/Frameworks"
mkdir -p "$bundle/Contents/Helpers"
mkdir -p "$bundle/Contents/Resources"
# SwiftPM dependencies contain syntax-highlighting assets and math fonts.
# Preserve their separate bundle names so Bundle.module works after packaging.
for resource_bundle in "$bin_dir/"*.bundle; do
  [[ -d "$resource_bundle" ]] || continue
  cp -R "$resource_bundle" "$bundle/Contents/Resources/"
done
notices="$bundle/Contents/Resources/NativeMarkdownNotices"
mkdir -p "$notices"
for dependency in MarkdownView beautiful-mermaid-swift elk-swift RichText Highlightr SwiftMath; do
  install -m 644 "$app_dir/.build/checkouts/$dependency/LICENSE" "$notices/$dependency.txt"
done
install -m 644 "$app_dir/.build/checkouts/swift-markdown/LICENSE.txt" "$notices/swift-markdown.txt"
install -m 644 "$app_dir/.build/checkouts/swift-cmark/COPYING" "$notices/swift-cmark.txt"
install -m 644 "$app_dir/.build/checkouts/Highlightr/src/assets/highlighter/LICENSE" "$notices/highlight-js.txt"
install -m 644 "$app_dir/.build/checkouts/SwiftMath/Sources/SwiftMath/mathFonts.bundle/LICENSE" "$notices/math-fonts.txt"
if [[ -L "$bundle/Contents/Resources/StudioRenderer" ]]; then
  printf 'Refusing to replace a symlinked renderer resource directory.\n' >&2; exit 1
fi
# The retired web transcript is diagnostic-only, never packaged in Paddock.
# Preserve old generated resources outside the app for recovery/debugging.
if [[ -d "$bundle/Contents/Resources/StudioRenderer" ]]; then
  retired_renderer_dir="$(mktemp -d /tmp/paddock-retired-renderer.XXXXXX)"
  mv "$bundle/Contents/Resources/StudioRenderer" "$retired_renderer_dir/StudioRenderer"
fi
if [[ -L "$bundle/Contents/Resources/StudioWorkspace" ]]; then
  printf 'Refusing a symlinked content resource directory.\n' >&2; exit 1
fi
mkdir -p "$bundle/Contents/Resources/StudioWorkspace"
rsync -a --delete -- "$app_dir/.build/studio-workspace/" "$bundle/Contents/Resources/StudioWorkspace/"
cp "$repo_dir/target/release/paddock-runner" "$bundle/Contents/Helpers/paddock-runner"
cp "$repo_dir/target/debug/libpaddock_desktop.dylib" "$bundle/Contents/Frameworks/"
install_name_tool -id '@rpath/libpaddock_desktop.dylib' "$bundle/Contents/Frameworks/libpaddock_desktop.dylib"
# SwiftPM's internal target is PaddockMac; the installed process is Paddock.
cp "$bin_dir/PaddockMac" "$bundle/Contents/MacOS/Paddock"
rm -f -- "$bundle/Contents/MacOS/PaddockMac"
cp "$repo_dir/packaging/macos/Info.plist" "$bundle/Contents/Info.plist"
plutil -lint "$bundle/Contents/Info.plist"
# Local development signature only. No notarization,
# updater, installer, standalone manager or model data is bundled by this script.
codesign --force --sign "$signing_identity" "$bundle/Contents/Frameworks/libpaddock_desktop.dylib"
codesign --force --sign "$signing_identity" "$bundle/Contents/Helpers/paddock-runner"
codesign --force --sign "$signing_identity" "$bundle"
codesign --verify --strict "$bundle"
printf 'Paddock: %s\n' "$bundle"
