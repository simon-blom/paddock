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
# Info.plist is a packaging template, not the product version. Stamp local
# builds too, so rebuilding the current app never makes About say 0.1.0.
runner_package_id="$(cargo pkgid --manifest-path "$repo_dir/Cargo.toml" -p paddock-runner)"
app_version="${runner_package_id##*#}"
app_version="${app_version##*@}"
app_version="${app_version%%-*}"
if [[ ! "$app_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'Cannot derive app version from runner package identity.\n' >&2; exit 1
fi
app_build_number="${PADDOCK_MACOS_BUILD_NUMBER:-}"
if [[ -z "$app_build_number" ]]; then
  previous_version="$(plutil -extract CFBundleShortVersionString raw "$bundle/Contents/Info.plist" 2>/dev/null || true)"
  previous_build="$(plutil -extract CFBundleVersion raw "$bundle/Contents/Info.plist" 2>/dev/null || true)"
  app_build_number=1
  if [[ "$previous_version" == "$app_version" && "$previous_build" =~ ^[1-9][0-9]{0,3}$ ]]; then
    app_build_number=$((previous_build + 1))
  fi
fi
if [[ ! "$app_build_number" =~ ^[1-9][0-9]{0,3}(\.(0|[1-9][0-9]?)){0,2}$ ]]; then
  printf 'Set PADDOCK_MACOS_BUILD_NUMBER to N[.N[.N]] (at most 4/2/2 digits).\n' >&2; exit 1
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
# Swift Build can stamp the deployment floor (15.0) as the SDK version. That
# silently disables modern AppKit/SwiftUI behavior in the app, while XCTest's
# newer host still enables it. Link with the actual selected SDK and validate
# the executable before copying anything into the installed bundle.
swift_sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
swift_sdk_version="$(xcrun --sdk macosx --show-sdk-version)"
swift_args=(--package-path "$app_dir" --product PaddockMac --sdk "$swift_sdk_path"
  -Xlinker -platform_version -Xlinker macos -Xlinker 15.0 -Xlinker "$swift_sdk_version"
  -Xlinker -rpath -Xlinker '@executable_path/../Frameworks')
xcrun swift build "${swift_args[@]}"
bin_dir="$(xcrun swift build "${swift_args[@]}" --show-bin-path)"
bash "$app_dir/scripts/check-linked-sdk.sh" "$bin_dir/PaddockMac" "$swift_sdk_version"
mkdir -p "$bundle/Contents/MacOS"
mkdir -p "$bundle/Contents/Frameworks"
mkdir -p "$bundle/Contents/Helpers"
mkdir -p "$bundle/Contents/Resources"
sparkle_source="$app_dir/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
[[ -d "$sparkle_source" && ! -L "$bundle/Contents/Frameworks/Sparkle.framework" ]]
ditto "$sparkle_source" "$bundle/Contents/Frameworks/Sparkle.framework"
# SwiftPM dependencies contain syntax-highlighting assets and math fonts.
# Preserve their separate bundle names so Bundle.module works after packaging.
for resource_bundle in "$bin_dir/"*.bundle; do
  [[ -d "$resource_bundle" ]] || continue
  cp -R "$resource_bundle" "$bundle/Contents/Resources/"
done
notices="$bundle/Contents/Resources/NativeMarkdownNotices"
mkdir -p "$notices"
install -m 644 "$app_dir/.build/checkouts/Sparkle/LICENSE" "$notices/Sparkle.txt"
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
install -m 644 "$repo_dir/packaging/macos/Paddock.icns" "$bundle/Contents/Resources/Paddock.icns"
cp "$repo_dir/packaging/macos/Info.plist" "$bundle/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string "$app_version" "$bundle/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$app_build_number" "$bundle/Contents/Info.plist"
if [[ -n "${PADDOCK_UPDATE_PUBLIC_KEY:-}" ]]; then
  plutil -replace SUPublicEDKey -string "$PADDOCK_UPDATE_PUBLIC_KEY" "$bundle/Contents/Info.plist"
fi
plutil -lint "$bundle/Contents/Info.plist"
# Local development signature only, without notarization or model data.
# Sparkle's nested installer components must be signed before the framework.
codesign --force --sign "$signing_identity" "$bundle/Contents/Frameworks/libpaddock_desktop.dylib"
codesign --force --sign "$signing_identity" "$bundle/Contents/Helpers/paddock-runner"
sparkle_bundle="$bundle/Contents/Frameworks/Sparkle.framework"
for nested_code in Autoupdate XPCServices/Downloader.xpc XPCServices/Installer.xpc Updater.app; do
  codesign --force --sign "$signing_identity" --preserve-metadata=entitlements "$sparkle_bundle/Versions/B/$nested_code"
done
codesign --force --sign "$signing_identity" "$sparkle_bundle"
codesign --force --sign "$signing_identity" "$bundle"
codesign --verify --strict "$bundle"
printf 'Paddock: %s\n' "$bundle"
