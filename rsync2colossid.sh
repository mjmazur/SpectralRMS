#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_dir> <user@host> <remote_base_dir>"
    echo "Example:"
    echo "  $0 /data/videos spectral@colossid /srv/meteor/archive"
    exit 1
fi

SRC_DIR="${1%/}"
REMOTE_HOST="$2"
REMOTE_BASE="${3%/}"

find "$SRC_DIR" -maxdepth 1 -type f -name 'ev_*.mp4' -print0 |
while IFS= read -r -d '' file; do
    base=$(basename "$file")
    date=${base:3:8}

    rsync -avh --progress --ignore-existing \
        "$file" \
        "${REMOTE_HOST}:${REMOTE_BASE}/${date}/"
done