#!/usr/bin/env bash
set -euo pipefail
IN="/home/johnny/action_ws/online_datasets/breakfast/breakfast/Videos_train.split1_cam01"
OUT="/home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames_train"
LOG_DIR="/home/johnny/action_ws/output/otas_logs"
LOG="$LOG_DIR/step4_extract_train.log"
mkdir -p "$OUT" "$LOG_DIR"
# Do not truncate existing log; append so we can see historical progress
processed=0; skipped=0; failed=0
shopt -s nullglob
for v in "$IN"/*.avi; do
  bn=$(basename "$v" .avi)
  p=${bn%%_*}
  cam=${bn#*_}; cam=${cam%%_*}
  rest=${bn#*_*_}
  rest=${rest#${p}_}
  dest="$OUT/${p}_${cam}_${rest}"
  mkdir -p "$dest"
  if compgen -G "$dest/Frame_*.jpg" > /dev/null; then
    echo "[SKIP] $bn existing" >> "$LOG"; skipped=$((skipped+1)); continue
  fi
  if ffmpeg -hide_banner -loglevel error -y -i "$v" "$dest/Frame_%06d.jpg"; then
    echo "[OK] $bn" >> "$LOG"; processed=$((processed+1))
  else
    echo "[FAIL] $bn" >> "$LOG"; failed=$((failed+1))
  fi
done
echo "[DONE] processed=$processed skipped=$skipped failed=$failed" >> "$LOG"
echo "Wrote log to $LOG"

