import json
import csv
from pathlib import Path
from collections import Counter

# Labels to process
LABELS = ["online_d01", "online_d02"]
BASE = Path("./datasets/output/figures")


def summarize_label(label: str) -> None:
    fig_dir = BASE / label
    stats_dir = fig_dir / "statistics"
    jsonl = fig_dir / "cluster_assignments.jsonl"
    out_txt = stats_dir / "cluster_counts.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    copied = Counter()

    if jsonl.exists():
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                c = int(o.get("cluster", -1))
                counts[c] += 1
                if o.get("copied_to_disk"):
                    copied[c] += 1

    # Try to read metrics and best k (by silhouette)
    metrics_csv = stats_dir / "cluster_metrics_seq_model_cosine.csv"
    best = None
    if metrics_csv.exists():
        with metrics_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    k = int(row["k"]) if row.get("k") else None
                    sil = float(row["silhouette"]) if row.get("silhouette") else None
                except Exception:
                    continue
                if k is None or sil is None:
                    continue
                if best is None or sil > best["silhouette"]:
                    # Keep the full row for reporting
                    row_copy = dict(row)
                    row_copy["k"] = k
                    row_copy["silhouette"] = sil
                    # cast numeric fields as needed
                    for fld in ("davies_bouldin", "calinski_harabasz", "intra_dist", "inter_centroid_dist", "intra_over_inter"):
                        if fld in row_copy and row_copy[fld] not in (None, ""):
                            try:
                                row_copy[fld] = float(row_copy[fld])
                            except Exception:
                                pass
                    best = row_copy

    with out_txt.open("w", encoding="utf-8") as w:
        total = sum(counts.values())
        w.write(f"label={label}\n")
        w.write(f"total_samples={total}\n")
        if best:
            w.write(
                "best_k={k} silhouette={silhouette:.4f} DB={davies_bouldin:.4f} CH={calinski_harabasz:.2f} intra/inter={intra_over_inter:.4f}\n".format(
                    **best
                )
            )
        for c in sorted(counts):
            w.write(f"cluster{c}: 总数={counts[c]}, 已复制={copied.get(c, 0)}\n")


if __name__ == "__main__":
    for L in LABELS:
        summarize_label(L)
    print("done")

