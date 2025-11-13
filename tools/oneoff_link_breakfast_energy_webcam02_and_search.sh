#!/usr/bin/env bash
set -euo pipefail

# One-off helper: map existing Breakfast energy outputs to webcam02 stems
# and run threshold search once.
# It creates per-video dirs named like: P03_webcam02_P03_cereals
# and inside puts a symlink stream_energy_quantized_token_diff_l2_mean.jsonl
# pointing to the existing energy file (usually 'segments_split1_webcam02').

ROOT="/home/johnny/action_ws"
EN_ROOT="$ROOT/output/breakfast"
GT_DIR="$ROOT/online_datasets/breakfast/gt_segments_json/test.split1_webcam02"
SOURCE="quantized"
MODE="token_diff_l2_mean"

created=0
skipped=0

shopt -s nullglob
for src_dir in "$EN_ROOT"/*; do
  bn="$(basename "$src_dir")"
  [[ ! -d "$src_dir" ]] && continue
  [[ "$bn" == "ABD_split1" || "$bn" == "stats" || "$bn" == "thresholds" ]] && continue
  [[ "$bn" == *_webcam02_* ]] && continue

  # locate energy file within the source dir
  ef="$src_dir/segments_split1_webcam02"
  if [[ ! -f "$ef" ]]; then
    matches=("$src_dir"/stream_energy_*.jsonl)
    if [[ -f "${matches[0]:-}" ]]; then
      ef="${matches[0]}"
    else
      echo "[skip] no energy file in $src_dir" >&2
      ((skipped++)) || true
      continue
    fi
  fi

  id="${bn%%_*}"
  action="${bn#*_}"
  dst_stem="${id}_webcam02_${id}_${action}"
  dst_dir="$EN_ROOT/$dst_stem"

  if [[ ! -f "$GT_DIR/${dst_stem}_segments.json" ]]; then
    echo "[warn] no GT for $dst_stem; skip" >&2
    ((skipped++)) || true
    continue
  fi

  mkdir -p "$dst_dir"
  ln -sfn "$ef" "$dst_dir/stream_energy_${SOURCE}_${MODE}.jsonl"
  echo "[link] $dst_dir/stream_energy_${SOURCE}_${MODE}.jsonl -> $ef"
  ((created++)) || true

done

echo "[done] created=$created skipped=$skipped"

# Run threshold search (laps env)
mkdir -p "$EN_ROOT/thresholds/split1_webcam02"
conda run -n laps python "$ROOT/tools/threshold_search_with_gt.py" \
  --energy-root "$EN_ROOT" \
  --gt-dir "$GT_DIR" \
  --source "$SOURCE" \
  --mode "$MODE" \
  --target-fps 10 --stride 4 \
  --hysteresis-ratio 0.95 --up-count 2 --down-count 2 --cooldown-windows 1 \
  --max-duration-seconds 2.0 \
  --tolerance-sec 2.0 \
  --output "$EN_ROOT/thresholds/split1_webcam02/best_threshold_quantized_token_diff.json"

