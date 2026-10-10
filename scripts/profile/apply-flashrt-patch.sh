#!/usr/bin/env bash
# Keep RoboSys's decoder and tuning extensions alongside a fetchable upstream pin.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
checkout="${1:-$root/3rdparty/FlashRT}"
for name in flashrt-pi05-decoder-hook flashrt-stream-autotune; do
    patch="$root/patches/$name.patch"
    if git -C "$checkout" apply --check "$patch" 2>/dev/null; then
        git -C "$checkout" apply "$patch"
        echo "Applied $name."
    elif git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
        echo "$name is already applied."
    else
        echo "Cannot apply $name: checkout contains incompatible edits." >&2
        exit 1
    fi
done
