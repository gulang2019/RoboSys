#!/usr/bin/env bash
# Keep RoboSys's small decoder extension alongside a fetchable upstream pin.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
checkout="${1:-$root/3rdparty/FlashRT}"
patch="$root/patches/flashrt-pi05-decoder-hook.patch"
if git -C "$checkout" apply --check "$patch" 2>/dev/null; then
    git -C "$checkout" apply "$patch"
    echo 'Applied FlashRT decoder hook.'
elif git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
    echo 'FlashRT decoder hook is already applied.'
else
    echo 'Cannot apply FlashRT decoder hook: checkout contains incompatible edits.' >&2
    exit 1
fi
