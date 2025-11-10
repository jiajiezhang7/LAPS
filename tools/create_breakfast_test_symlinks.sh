#!/usr/bin/env bash
set -euo pipefail
# Create a flat test video dir for Breakfast split1 by symlinking raw .avi files
# Mapping rules:
#   - Bundle items look like: P03_cam01_P03_cereals.txt
#   - Raw videos live under:  raw_videos/BreakfastII_15fps_qvga_sync/P03/{cam01,cam02,webcam01,webcam02}/P03_cereals.avi
#   - For stereo01, raw lives under: raw_videos/.../P03/stereo/P03_cereals_ch{0,1}.avi (prefer ch1, fallback ch0)

ROOT="/home/johnny/action_ws"
VIDEOS_ROOT="$ROOT/online_datasets/breakfast/breakfast/raw_videos/BreakfastII_15fps_qvga_sync"
OUT_DIR="$ROOT/online_datasets/breakfast/breakfast/Videos_test.split1"
SPLIT_FILE="$ROOT/online_datasets/breakfast/breakfast/splits/test.split1.bundle"

mkdir -p "$OUT_DIR"

while IFS= read -r f || [[ -n "${f:-}" ]]; do
  f="${f%$'\r'}"  # strip CR if any
  [[ -z "$f" ]] && continue
  stem="${f%.txt}"
  IFS='_' read -r P VIEW P2 ACT <<< "$stem"
  if [[ -z "${P:-}" || -z "${VIEW:-}" || -z "${ACT:-}" ]]; then
    echo "[WARN] Unrecognized line: $f" >&2
    continue
  fi
  src=""
  case "$VIEW" in
    cam01|cam02|webcam01|webcam02)
      src="$VIDEOS_ROOT/$P/$VIEW/${P}_${ACT}.avi"
      ;;
    stereo01)
      # Prefer ch1, fallback ch0
      if [[ -f "$VIDEOS_ROOT/$P/stereo/${P}_${ACT}_ch1.avi" ]]; then
        src="$VIDEOS_ROOT/$P/stereo/${P}_${ACT}_ch1.avi"
      elif [[ -f "$VIDEOS_ROOT/$P/stereo/${P}_${ACT}_ch0.avi" ]]; then
        src="$VIDEOS_ROOT/$P/stereo/${P}_${ACT}_ch0.avi"
      else
        src=""
      fi
      ;;
    *)
      echo "[WARN] Unknown VIEW '$VIEW' for $stem" >&2
      ;;
  esac
  dst="$OUT_DIR/${stem}.avi"
  if [[ -n "$src" && -f "$src" ]]; then
    ln -sfn "$src" "$dst"
    echo "[OK] $dst -> $src"
  else
    echo "[WARN] Missing raw video for $stem (expected at: $src)" >&2
  fi

done < "$SPLIT_FILE"

echo "Done. Symlinks in: $OUT_DIR"

