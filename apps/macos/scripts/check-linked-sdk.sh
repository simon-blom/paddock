#!/bin/bash
set -euo pipefail

if [[ $# != 2 || ! -f "$1" || ! "$2" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
  printf 'Usage: check-linked-sdk.sh executable expected-macOS-SDK-version\n' >&2
  exit 1
fi
# Check every architecture, not the deployment target or Info.plist. AppKit
# chooses linked-on-or-after behavior using the main executable's SDK metadata.
xcrun vtool -show-build "$1" | awk -v expected="$2" '
  function version(value, parts) {
    split(value, parts, ".")
    return parts[1] * 10000 + parts[2] * 100 + parts[3]
  }
  $1 == "sdk" {
    count++
    if (version($2) != version(expected) || version($2) < version("26.0")) bad=1
  }
  END {
    if (!count || bad) {
      print "Incorrect linked SDK: native scroll-edge behavior would differ from tests. Rebuild with the selected macOS SDK (26+)." > "/dev/stderr"
      exit 1
    }
  }'
