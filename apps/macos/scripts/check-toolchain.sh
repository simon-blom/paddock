#!/bin/bash
set -euo pipefail

# Minimum verified production toolchain, not an auto-installer. Recheck Apple's
# stable release matrix before every milestone; do not select a beta globally.
compiler_version="$(xcrun swift --version | sed -nE 's/.*Apple Swift version ([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/p')"
if [[ -z "$compiler_version" ]]; then
    echo "Select a production Xcode toolchain with Apple Swift 6.3.3 or newer." >&2
    exit 1
fi
if ! awk -v version="$compiler_version" 'BEGIN { split(version, v, "."); exit !(v[1] > 6 || (v[1] == 6 && (v[2] > 3 || (v[2] == 3 && v[3] >= 3)))) }'; then
    echo "Apple Swift $compiler_version is older than the verified 6.3.3 baseline." >&2
    exit 1
fi
xcodebuild -version
xcrun swift --version
