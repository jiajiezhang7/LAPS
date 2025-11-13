# 伪标签（y_pseudo）生成与阈值标定（Threshold Calibration）技术说明

本文档系统性阐述 Proxy Signal → y_pseudo 的无监督生成机制，以及利用 y_pseudo 对 E_action（quantized token diff）进行阈值标定（proxy-supervision）的具体实现，基于以下两份脚本的行为与源码：
- video_action_segmenter/compute_best_threshold.py
- video_action_segmenter/analyze_energy_jsonl.py

---

## 1. 术语与信号定义

- Proxy Signal（低级代理信号）：velocity token diff L2-mean。
  - 文件示例：`.../stream_energy_velocity_token_diff_l2_mean.jsonl`
  - 语义：度量相邻 token（未量化）在时间上的变化强度，反映“速度能量”。
- E_action（高级动作能量）：quantized token diff L2-mean。
  - 文件示例：`.../stream_energy_quantized_token_diff_l2_mean.jsonl`
  - 语义：在离散化（FSQ/VQ等）后的 token 序列上，度量相邻 token 差异，反映抽象动作变化强度。
- 窗口对齐（window alignment）：两条序列按 `window` 索引求交集并对齐（避免长度/起止不一致）。

---

## 2. 伪标签（y_pseudo）生成原理（完全无监督）

目标：仅基于 Proxy Signal 的分布，生成二值伪标签 y_pseudo，用于后续的阈值标定与评估，不使用任何 Ground Truth。

### 2.1 无监督阈值生成方法
对给定的 Proxy Signal 序列 v（按窗口对齐后的数值数组），支持三类“纯无监督”的阈值选择：

1) Otsu threshold（默认 `auto`）
- 通过最大化类间方差选取阈值。
- 源码参考：`compute_best_threshold.py: otsu_threshold()` 与 `analyze_energy_jsonl.py: otsu_threshold()`

2) 分位数（Quantile）
- 指定 `quantile:q`，例如 0.9/0.95，取 v 的分位点作为阈值。

3) 固定值（Value）
- 指定 `value:x`，直接使用给定数值。

伪代码（以 `thr_spec` 表示阈值策略）：
```python
# v: np.ndarray, proxy series (velocity energy)
if thr_spec == 'auto':
    thr = otsu_threshold(v)
elif thr_spec.startswith('quantile:'):
    thr = quantile(v, q)
elif thr_spec.startswith('value:'):
    thr = x
else:
    thr = otsu_threshold(v)
```

### 2.2 从 Proxy Signal 到二值伪标签 y_pseudo
给定阈值 thr，逐窗口生成：
```python
y_pseudo[t] = 1 if v[t] > thr else 0
```
- 该过程严格基于 v 的统计特性，与 GT 无关。
- 在 `compute_best_threshold.py` 中：`gen_label_threshold()` 产出 thr_v，然后 `y = (v > thr_v)`。
- 在 `analyze_energy_jsonl.py` 中：`make_velocity_labels()` 产出 y 与参考序列 ref=v。

结论：y_pseudo 的定义与生成是“无监督”的（unsupervised），即不依赖任何 GT 标注。

---

## 3. 使用 y_pseudo 进行 E_action 阈值标定（Best Threshold 机制）

目标：用 y_pseudo 作为“监督信号”来选择 E_action（x）上的最佳阈值，以更好地区分 y_pseudo=1 与 y_pseudo=0 的时段。该方法是“proxy-supervision”，不是“GT-supervision”。

### 3.1 数据对齐与输入
- 加载并对齐：`x`（E_action：quantized token diff）与 `v`（Proxy：velocity token diff），得到同一窗口集合下的 `x, v`。
- 以 `v` 的无监督阈值 thr_v 生成 `y_pseudo = (v > thr_v)`。

### 3.2 阈值搜索策略（search over candidates）
- 生成 E_action 的阈值候选集：`unique(x)` 与相邻点的中点（midpoints），增强鲁棒性。
- 对每个候选阈值 `τ`：
  - 计算预测 `ŷ = (x > τ)`，相对于 `y_pseudo` 统计指标：
    - F1 = 2·(Precision·Recall)/(Precision+Recall)
    - Youden's J = TPR − FPR
- 选取 `F1`/`J` 最优对应的 `τ` 作为推荐阈值。

伪代码（简化）：
```python
# x: E_action series, y: y_pseudo
cands = sorted_unique(x) ∪ midpoints(sorted_unique(x))
best_f1, best_j = (-inf, None), (-inf, None)
for tau in cands:
    yhat = (x > tau)
    f1, j = F1(y, yhat), J(y, yhat)
    update bests
return tau_f1_best, tau_j_best
```

源码参考：`compute_best_threshold.py: search_best_thresholds()` 与 `metrics_at_thr()`。

### 3.3 产出与解释
- 输出 JSON（示例字段）：
  - `label_threshold_on_velocity_token_diff`：用于生成 y_pseudo 的 thr_v
  - `quantized_token_diff_best.best_f1` / `best_j`：在 y_pseudo 监督下得到的 E_action 最佳阈值与指标
  - `quantized_stats`：E_action 的统计描述（min/max/mean/quantiles）
  - `smoothing`：可选的对 x（E_action）进行平滑的配置（不影响“无监督”本质）

---

## 4. 两个脚本的分工与协作

### 4.1 compute_best_threshold.py（单样本阈值标定）
- 输入：单段样本的 `stream_energy_quantized_token_diff_l2_mean.jsonl` 与 `stream_energy_velocity_token_diff_l2_mean.jsonl`。
- 步骤：
  1) 对齐窗口；
  2) 基于 velocity 产生 y_pseudo（Otsu/Quantile/Value）；
  3) 在 E_action 上搜索最佳阈值（F1/Youden's J）。
- 输出：包含 `best_f1`、`best_j`、`thr_v` 与统计信息的 JSON 报告。
- 可选：对 E_action 进行轻量平滑（EMA/MA），提升鲁棒性。

### 4.2 analyze_energy_jsonl.py（批量分析与可视化报告）
- 输入：目录下的多条 energy 系列（包括 velocity、quantized 等）。
- 步骤：
  1) 发现并加载所有 `stream_energy_*.jsonl`；
  2) 统一对齐窗口；
  3) 用 velocity 的 Otsu/Quantile/Value 得到 y_pseudo；
  4) 统计多种度量（AUC、Cohen's d、KS、分位差、相关性）；
  5) 生成图表（时序/直方/箱线/相关/ROC）与 HTML 报告。
- 作用：从“群体/批量”视角验证 E_action 与 Proxy 的区分度与相关性，辅助理解与选型。

---

## 5. 与论文补充材料 Fig S3 的对应关系

- Proxy = velocity token diff（无监督）
- 阈值 = Otsu / Quantile / 固定值（仅基于 velocity 的分布）
- y_pseudo = (velocity > thr)；E_action = quantized token diff
- 可视化：在同一时序图上展示 Proxy、E_action、无监督阈值线与 y_pseudo（阴影/阶梯线），准确体现“proxy-supervision”的校准思想。
- 相关脚本已修正：`scripts/supplement/plot_proxy_vs_e_action.py`
  - 示例命令：
    ```bash
    python scripts/supplement/plot_proxy_vs_e_action.py \
      --sample_dir datasets/output/energy_sweep_out/D01/D01_sample_1_seg001 \
      --outpdf supplement_output/segmentor/fig_S3_proxy_vs_e_action_D01_sample_1_seg001.pdf \
      --pseudo_thr auto
    ```

---

## 6. 数据流与模块协作（Mermaid）

```mermaid
flowchart LR
  A[Velocity token diff (v)] --> B[Unsupervised thresholding\n(Otsu/Quantile/Value)]
  B --> C[y_pseudo = (v > thr_v)]
  D[Quantized token diff (x)] --> E[Window alignment]
  A --> E
  E --> F[Threshold search on x\nvs y_pseudo]
  C --> F
  F --> G[Best thresholds for E_action\n(F1 / Youden's J)]
  C --> H[Visualization: y_pseudo shading/step]
  A --> H
  D --> H
```

---

## 7. 关键公式与度量

- Precision = TP / (TP + FP)
- Recall (TPR) = TP / (TP + FN)
- F1 = 2 · Precision · Recall / (Precision + Recall)
- Youden's J = TPR − FPR
- 在本流程中，TP/FP/FN/… 皆基于 `y_pseudo`（proxy-supervision），而非 GT。

---

## 8. 实践建议

- 首选 Otsu（auto）作为 y_pseudo 的无监督阈值；在分布偏斜时可尝试 `quantile:0.9/0.95`。
- 对 E_action 进行轻量平滑有时能提升阈值鲁棒性（见 `--smooth` 选项）。
- 批量评估请使用 `analyze_energy_jsonl.py`，单样本标定请使用 `compute_best_threshold.py`；图形展示配合 `scripts/supplement/plot_proxy_vs_e_action.py` 的修正版本。

