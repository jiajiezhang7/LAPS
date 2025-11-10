#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/johnny/action_ws"
VIDEOS_DIR="$ROOT/online_datasets/gtea/gtea/Videos"
OUT_DIR="$ROOT/online_datasets/gtea/gtea/Videos_test.split1"
SPLIT_FILE="$ROOT/online_datasets/gtea/gtea/splits/test.split1.bundle"
mkdir -p "$OUT_DIR"
while IFS= read -r f || [[ -n "$f" ]]; do
  f="${f%$'\r'}"
  [[ -z "$f" ]] && continue
  stem="${f%.txt}"
  src="$VIDEOS_DIR/${stem}.mp4"
  dst="$OUT_DIR/${stem}.mp4"
  if [[ -f "$src" ]]; then
    ln -sf "$src" "$dst"
    echo "[OK] $dst -> $src"
  else
    echo "[WARN] missing $src"
  fi
done < "$SPLIT_FILE"

