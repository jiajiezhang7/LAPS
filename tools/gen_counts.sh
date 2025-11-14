#!/usr/bin/env bash
set -euo pipefail

BASE="/home/johnny/action_ws/datasets/output/figures"
CLUSTERED_BASE="/home/johnny/action_ws/datasets/output/clustered_results"

process_label() {
  local label="$1"
  local fig_dir="$BASE/$label"
  local stats_dir="$fig_dir/statistics"
  local jsonl="$fig_dir/cluster_assignments.jsonl"
  mkdir -p "$stats_dir"
  local out="$stats_dir/cluster_counts.txt"
  : > "$out"
  echo "label=$label" >> "$out"
  if [[ -f "$jsonl" ]]; then
    local total
    total=$(wc -l < "$jsonl" | tr -d ' ')
    echo "total_samples=$total" >> "$out"
    # parse best metrics from csv (max silhouette)
    local csv="$stats_dir/cluster_metrics_seq_model_cosine.csv"
    if [[ -f "$csv" ]]; then
      # skip header; find row with max silhouette
      local best_line
      best_line=$(awk -F, 'NR>1 {print $0}' "$csv" | awk -F, 'BEGIN{max=-1;line=""} {if ($5+0>max) {max=$5; line=$0}} END{print line}')
      if [[ -n "$best_line" ]]; then
        IFS=',' read -r method k n_samples dim silhouette db ch intra inter ratio <<< "$best_line"
        printf 'best_k=%s silhouette=%.4f DB=%.4f CH=%.2f intra/inter=%.4f\n' "$k" "$silhouette" "$db" "$ch" "$ratio" >> "$out"
      fi
    fi
    # detect unique clusters from jsonl
    for c in $(awk -F'"cluster"\s*:\s*' '{if(NR>0 && NF>1){split($2,a,/[,}]/); print a[1]}}' "$jsonl" | sort -n | uniq); do
      local cnt copy_cnt
      cnt=$(grep -Ec '"cluster"\s*:\s*'"$c" "$jsonl" | tr -d ' ')
      copy_cnt=$(awk -v k="$c" -F'"cluster"\s*:\s*' '{ if (NF>1){ split($2,a,/[,}]/); if (a[1]==k){ if ($0 ~ /"copied_to_disk"\s*:\s*true/) print 1 } } }' "$jsonl" | wc -l | tr -d ' ')
      printf 'cluster%s: 总数=%s, 已复制=%s\n' "$c" "$cnt" "$copy_cnt" >> "$out"
    done
  fi
}

process_label online_d01
process_label online_d02

echo "done"
