#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/../.." && pwd)"
LAB_BUNDLE="$APP_ROOT/.build/PaddockRenderingLab.app"

# Distinct executable, resources and identity. Never stop/rebuild Paddock.app here.
if pgrep -f "$LAB_BUNDLE/Contents/MacOS/PaddockRendererLab" >/dev/null; then
  echo 'Quit Rendering Lab before rebuilding its bundle.' >&2
  exit 1
fi
if [[ ! -x "$REPO_ROOT/studio/node_modules/.bin/vite" ]]; then
  echo 'Install the locked Studio dependencies with npm ci in studio/ first.' >&2
  exit 1
fi
cd "$REPO_ROOT/studio"
node_modules/.bin/vite build --config renderer-lab/vite.config.ts
cd "$APP_ROOT"
xcrun swift build -c release --product PaddockRendererLab
SWIFT_BIN_DIR="$(xcrun swift build -c release --show-bin-path)"
mkdir -p "$LAB_BUNDLE/Contents/MacOS" "$LAB_BUNDLE/Contents/Resources"
cp "$SWIFT_BIN_DIR/PaddockRendererLab" "$LAB_BUNDLE/Contents/MacOS/"
# Synchronize only this generated resource directory, removing obsolete hashed
# chunks without accumulating copies of the WASM on every development build.
if [[ -L "$LAB_BUNDLE/Contents/Resources/RendererLab" ]]; then
  echo 'Refusing to replace a symlinked renderer resource directory.' >&2
  exit 1
fi
mkdir -p "$LAB_BUNDLE/Contents/Resources/RendererLab"
rsync -a --delete -- "$APP_ROOT/.build/renderer-web/" "$LAB_BUNDLE/Contents/Resources/RendererLab/"
cp "$APP_ROOT/RendererLab-Info.plist" "$LAB_BUNDLE/Contents/Info.plist"
codesign --force --sign - "$LAB_BUNDLE"
echo "Built development-only lab: $LAB_BUNDLE"
echo "Run: open \"$LAB_BUNDLE\" --args --run-all --output /tmp/paddock-renderer-report"
