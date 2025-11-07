# 论文补充材料实验 ToDoList（可执行版）

本清单将补充材料中的每个小节拆解为可执行任务，逐项给出：脚本、数据、环境、命令、预期输出、验证方式、依赖顺序与注意事项。默认在 `laps` conda 环境运行（遵循全局规则）。

- 通用约定
  - 统一使用：`conda run -n laps python ...`
  - 变量占位：`{VIEW}`∈{D01,D02}；`{RAW}`=`/home/johnny/action_ws/datasets/gt_raw_videos/{VIEW}`；`{OUT}`=`/home/johnny/action_ws/datasets/output`
  - 现有能量文件（可直接复用）：
    - Optical Flow energy JSONL（TV-L1）：`{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_optical_flow_mag_mean.jsonl`
    - Action Energy JSONL（Quantized/Velocity）：若无，可用在线脚本生成；示例仓库已有 D02 样例：`/home/johnny/action_ws/video_action_segmenter/energy_sweep_out/D02_20250811064933/`
  - Ground Truth 目录：`/home/johnny/action_ws/datasets/gt_segments/{VIEW}`（包含 `{stem}_segments.json`）。若路径不同，请在命令中替换。
  - 仅当特别说明时才需要其他环境（如 `abd_env`/`otas`）。本补充实验全部可在 `laps` 完成。

---

## 1. 核心方法论详情（Methodology Details）

### 1.1 实验 1：Motion Tokenizer 架构与训练（Figure S1/S2, Table S1）

- 任务 1.1.1（Figure S1 架构图）
  - **脚本/资源**：`amplify/assets/architecture.png`
  - **数据**：无
  - **环境**：`laps`
  - **操作**：确认图片是否与当前模型配置一致（`amplify/cfg/train_motion_tokenizer.yaml`）。若不一致，更新标注（编码器/解码器/FSQ 流程与张量形状）。
  - **预期输出**：`docs/figures/Figure_S1_architecture.png`
  - **验证**：与配置文件字段（`num_layers/num_heads/hidden_dim/codebook_size`）一致。

- 任务 1.1.2（S2 训练损失曲线）
  - **脚本**：`amplify/train_motion_tokenizer.py`
  - **数据**：HDF5 预处理数据根目录（见配置 `root_dir`）
  - **环境**：`laps`
  - **命令（示例，快速出曲线）**：
    ```bash
    conda run -n laps python -m amplify.train_motion_tokenizer \
      root_dir=/home/johnny/action_ws/data/preprocessed_data_d01 \
      num_epochs=2 use_wandb=true run_name=supp_mat_s2 \
      save_interval=1
    ```
  - **预期输出**：
    - 控制台打印最终配置与 `Checkpoint dir: ...` 路径
    - wandb 曲线（`train_loss/val_loss`）或本地日志中的损失随 epoch 下降趋势截图
  - **验证**：损失曲线单调下降或稳定收敛；代码本身会打印 `Epoch ... Train Loss ... Val Loss ...`。

- 任务 1.1.3（S1/S2 对应的 Table S1 超参数）
  - **脚本/资源**：`amplify/cfg/train_motion_tokenizer.yaml`
  - **环境**：`laps`
  - **操作**：从 YAML 提取关键字段（Tracker、`num_layers/num_heads/hidden_dim/codebook_size/true_horizon/num_tracks/loss.*` 等），整理为 Markdown 表格。
  - **预期输出**：`docs/tables/Table_S1_hparams.md`
  - **验证**：与 YAML 完全对应；变更需注明 commit 哈希与日期。

- 注意事项
  - 若 `opencv`/`umap-learn` 缺失，不影响本节；仅训练依赖 PyTorch 与数据。

---

### 1.2 实验 2：Action Segmentor 阈值无监督优化（Figure S3/S4/S5）

- 任务 1.2.1（准备能量信号与 GT）
  - **脚本**：
    - 光流能量：`video_action_segmenter/scripts/compute_optical_flow_energy.py`
    - 行为能量（Quantized/Velocity）：可复用已有 JSONL，或用在线流式脚本生成：`video_action_segmenter/stream_inference.py`
  - **数据**：`{RAW}` 下原始 mp4；`{OUT}` 输出根目录；GT：`/home/johnny/action_ws/datasets/gt_segments/{VIEW}`
  - **环境**：`laps`
  - **命令（光流）**：
    ```bash
    conda run -n laps python -m video_action_segmenter.scripts.compute_optical_flow_energy \
      --view {VIEW} \
      --input-dir {RAW} \
      --output-root {OUT}/energy_sweep_out/{VIEW} \
      --target-fps 10 --ema-alpha 0.7 --resize-shorter 480
    ```
  - **命令（可选：在线生成 quantized/velocity 能量 JSONL）**：通过 `--params` 指定量化/速度模式的配置（仓库已提供 `params_d0{1,2}_quant.yaml` 与 `params_d0{1,2}_vel.yaml`）。
    ```bash
    # 量化差分能量（E_action）
    conda run -n laps python -m video_action_segmenter.stream_inference \
      --params video_action_segmenter/params_d02_quant.yaml

    # 速度差分能量（Proxy，用于伪标签/对齐）
    conda run -n laps python -m video_action_segmenter.stream_inference \
      --params video_action_segmenter/params_d02_vel.yaml
    ```
  - **预期输出**：每视频子目录含 `stream_energy_*.jsonl`
  - **验证**：JSONL 行为 `{window:int, energy:float, source, mode}`，窗口连续。

- 任务 1.2.2（Figure S3：代理信号 vs 我们的高级信号 + GT）
  - **脚本**：`video_action_segmenter/scripts/plot_energy_comparison.py`
  - **命令**：
    ```bash
    conda run -n laps python -m video_action_segmenter.scripts.plot_energy_comparison \
      --optical-flow-jsonl {OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_optical_flow_mag_mean.jsonl \
      --action-energy-jsonl {OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl \
      --gt-json /home/johnny/action_ws/datasets/gt_segments/{VIEW}/{video}_segments.json \
      --output-dir /home/johnny/action_ws/video_action_segmenter/figures/{VIEW} \
      --start-sec 0 --duration-sec 60 --dpi 300
    ```
  - **预期输出**：对比图（红=光流，蓝=E_action，绿虚线=GT 边界），PNG
  - **验证**：时间轴单位为秒；两条曲线归一化到 [0,1]；GT 边界与片段区间高亮可见。

  - **补充命令（S3 三轨：Velocity、y_pseudo(Otsu) 与 E_action + GT，一键导图）**：无需新增脚本，直接用内嵌 Python 生成论文风格图。
    ```bash
    conda run -n laps python - <<'PY'
    from pathlib import Path
    import json, numpy as np
    import matplotlib.pyplot as plt

    # 配置占位（请按需替换 VIEW/video/OUT）
    VIEW = '{VIEW}'
    OUT = '{OUT}'
    video = '{video}'
    target_fps, stride = 10.0, 4
    start_sec, duration_sec, dpi = 0.0, 60.0, 300

    def load_jsonl(p: Path):
        m = {}
        if not p.exists():
            return m
        for ln in p.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                o = json.loads(ln)
                m[int(o['window'])] = float(o['energy'])
        return m

    def w2s(w):
        return float(w) * stride / target_fps

    # 路径：Velocity/Quantized/GT
    v_path = Path(f"{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_velocity_token_diff_l2_mean.jsonl")
    q_path = Path(f"{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl")
    gt_path = Path(f"/home/johnny/action_ws/datasets/gt_segments/{VIEW}/{video}_segments.json")

    vel = load_jsonl(v_path)
    act = load_jsonl(q_path)
    gt = json.loads(gt_path.read_text(encoding='utf-8')) if gt_path.exists() else {'segments': []}

    # 裁剪到指定时间窗
    s0, e0 = start_sec, start_sec + duration_sec
    def slice_series(m):
        xs, ys = [], []
        for w, e in sorted(m.items()):
            t = w2s(w)
            if s0 <= t <= e0:
                xs.append(t); ys.append(e)
        return np.array(xs), np.array(ys)
    tx_v, vy = slice_series(vel)
    tx_q, qy = slice_series(act)
    if tx_v.size == 0 or tx_q.size == 0:
        raise SystemExit('No data in selected range')

    # Otsu 阈值生成 y_pseudo（Velocity -> 0/1）
    def otsu_thr(a, bins=128):
        a = a.astype(np.float64)
        hist, bin_edges = np.histogram(a, bins=bins)
        hist = hist.astype(np.float64)
        p = hist / np.maximum(hist.sum(), 1e-12)
        omega = np.cumsum(p)
        mu = np.cumsum(p * (bin_edges[:-1]))
        mu_t = mu[-1]
        sigma_b = (mu_t * omega - mu)**2 / np.maximum(omega * (1 - omega), 1e-12)
        k = np.nanargmax(sigma_b)
        return float(bin_edges[k])
    thr = otsu_thr(vy)
    y_pseudo = (vy > thr).astype(float)

    # 归一化能量到 [0,1]
    def norm(a):
        a = a.astype(np.float64)
        return (a - a.min()) / (a.ptp() + 1e-8)
    vy_n = norm(vy)
    qy_n = norm(qy)

    # 作图
    import os
    out_dir = Path(f"/home/johnny/action_ws/video_action_segmenter/figures/{VIEW}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14,6), dpi=dpi)
    ax.plot(tx_v, vy_n, color='#E74C3C', lw=2.5, label='Velocity Energy (Proxy)')
    ax.plot(tx_q, qy_n, color='#3498DB', lw=2.5, label=r"$E_{action}$ (Quantized)")
    # y_pseudo 以台阶线展示（放大到 0~1 区间）
    ax.step(tx_v, 0.9*y_pseudo, where='post', color='black', lw=2.0, label='Pseudo Labels (Otsu)')

    # 叠加 GT 边界
    s, e = s0, e0
    for seg in gt.get('segments', []):
        a, b = float(seg['start_sec']), float(seg['end_sec'])
        if not (b < s or a > e):
            ax.axvline(max(a,s), color='#2ECC71', ls='--', lw=1.5, alpha=0.6)
            ax.axvline(min(b,e), color='#2ECC71', ls='--', lw=1.5, alpha=0.6)
            ax.axvspan(max(a,s), min(b,e), color='#2ECC71', alpha=0.08)

    ax.set_xlim(s0, e0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95)
    plt.tight_layout()
    out_png = out_dir / f's3_proxy_pseudo_vs_action_{video}_{int(s0)}s_{int(duration_sec)}s.png'
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    print('Saved', out_png)
    PY
    ```

- 任务 1.2.3（Figure S4：阈值扫描曲线 + 推荐阈值）
  - **脚本（无监督阈值搜索）**：`video_action_segmenter/compute_best_threshold.py`
  - **命令（输出最佳 F1/Youden）**：
    ```bash
    conda run -n laps python -m video_action_segmenter.compute_best_threshold \
      --quantized-jsonl {OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl \
      --velocity-jsonl  {OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_velocity_token_diff_l2_mean.jsonl \
      --label-threshold auto \
      --output-json {OUT}/energy_sweep_report/{VIEW}/best_threshold_quantized_token_diff.json
    ```
  - **可选（跨视频稳健聚合阈值）**：`video_action_segmenter/aggregate_thresholds.py`
    ```bash
    conda run -n laps python -m video_action_segmenter.aggregate_thresholds \
      --view {VIEW} \
      --quantized-root {OUT}/energy_sweep_out/{VIEW} \
      --velocity-root  {OUT}/energy_sweep_out/{VIEW} \
      --label-threshold auto \
      --output-json {OUT}/energy_sweep_report/{VIEW}/best_threshold_quantized_token_diff_agg.json
    ```
  - **命令（导出完整阈值-指标曲线为 CSV，用于绘制 Figure S4；基于现有函数的 one-liner）**：
    ```bash
    conda run -n laps python - <<'PY'
    from pathlib import Path
    import json, csv
    import numpy as np
    from video_action_segmenter.compute_best_threshold import load_energy_jsonl, align_by_windows, gen_label_threshold, metrics_at_thr
    q = Path('{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl')
    v = Path('{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_velocity_token_diff_l2_mean.jsonl')
    wq,x = load_energy_jsonl(q); wv,y0 = load_energy_jsonl(v)
    x,y0 = align_by_windows(wq,x,wv,y0)
    thr_v = gen_label_threshold(y0, 'auto'); y = (y0>thr_v).astype(int)
    xs = np.unique(x); mids = (xs[:-1]+xs[1:])/2.0 if xs.size>1 else xs
    cands = np.unique(np.concatenate([xs,mids]))
    out = []
    for t in cands:
        m = metrics_at_thr(x,y,float(t))
        out.append({'thr':m['thr'],'F1':m['f1'],'TPR':m['tpr'],'FPR':m['fpr'],'Precision':m['precision'],'Recall':m['recall']})
    out_dir = Path('{OUT}/energy_sweep_report/{VIEW}'); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'threshold_curve.csv','w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print('Wrote', out_dir/'threshold_curve.csv')
    PY
    ```
  - **预期输出**：`best_threshold_quantized_token_diff.json` 与 `threshold_curve.csv`
  - **验证**：CSV 中 F1 峰值处的阈值应接近 JSON 中的 `best_f1.thr`。

  - **命令（从 CSV 绘制 Figure S4 PNG）**：
    ```bash
    conda run -n laps python - <<'PY'
    from pathlib import Path
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    VIEW = '{VIEW}'; OUT = '{OUT}'
    csv_path = Path(f"{OUT}/energy_sweep_report/{VIEW}/threshold_curve.csv")
    xs, f1 = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row['thr'])); f1.append(float(row['F1']))
    xs = np.array(xs); f1 = np.array(f1)
    i = int(np.nanargmax(f1))
    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    ax.plot(xs, f1, color='#2C3E50', lw=2.5)
    ax.scatter([xs[i]],[f1[i]], color='#E74C3C', s=60, zorder=5, label=f'Max F1 @ thr={xs[i]:.4f}')
    ax.set_xlabel(r'$\\theta_{on}$', fontsize=16, fontweight='bold')
    ax.set_ylabel('F1-score', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    out_png = csv_path.parent / 's4_threshold_curve.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved', out_png)
    PY
    ```

- 任务 1.2.4（Figure S5：Hysteresis & Debounce 敏感性）
  - **脚本**：
    - 离线分割：`tools/segment_from_energy.py`
    - 评估：`tools/eval_segmentation.py`
  - **命令（示例：固定 `thr_on`，扫描 `hysteresis_ratio` 与 `up/down_count`）**：
    ```bash
    # 1) 用 GT 搜索得到的阈值做离线分割（逐视频写 {stem}_segments.json）
    conda run -n laps python tools/segment_from_energy.py \
      --view {VIEW} \
      --energy-root {OUT}/energy_sweep_out/{VIEW} \
      --threshold-json {OUT}/energy_sweep_report/{VIEW}/best_threshold_quantized_token_diff.json \
      --output-root {OUT}/segmentation_outputs/{VIEW} \
      --source quantized --mode token_diff_l2_mean \
      --target-fps 10 --stride 4 \
      --hysteresis-ratio 0.90 --up-count 2 --down-count 2 --cooldown-windows 1 --max-duration-seconds 2.0

    # 2) 评估（F1@tol 与 mAP@IoU）
    conda run -n laps python tools/eval_segmentation.py \
      --pred-root {OUT}/segmentation_outputs/{VIEW} \
      --gt-dir /home/johnny/action_ws/datasets/gt_segments/{VIEW} \
      --tolerance-sec 2.0 --tolerance-secs 1.0 2.0 3.0 \
      --iou-thrs 0.5 0.75 \
      --output {OUT}/segmentation_outputs/{VIEW}/eval_summary.json
    ```
    - 重复上述分割与评估，改变：`--hysteresis-ratio ∈ {0.9,0.95,0.98}`；`--up-count/--down-count ∈ {1,2,3}`；可固定 `cooldown_windows=1`，或对 `max-duration-seconds` 做 1.5/2.0/3.0 的敏感性。
  - **预期输出**：每组参数的 `eval_summary.json`，可汇总为曲线（F1 vs ratio / F1 vs up/down）。
  - **验证**：F1 曲线在最优点附近平缓，证明鲁棒性。

  - **批量网格扫描（建议使用独立输出目录避免覆盖）**：
    ```bash
    # 推荐网格：r in {0.90,0.95,0.98}; up/down in {1,2,3}
    VIEW={VIEW}; OUT={OUT}
    for r in 0.90 0.95 0.98; do
      for u in 1 2 3; do
        for d in 1 2 3; do
          conda run -n laps python tools/segment_from_energy.py \
            --view ${VIEW} \
            --energy-root ${OUT}/energy_sweep_out/${VIEW} \
            --threshold-json ${OUT}/energy_sweep_report/${VIEW}/best_threshold_quantized_token_diff.json \
            --output-root ${OUT}/segmentation_outputs/${VIEW}/r${r}/u${u}_d${d} \
            --source quantized --mode token_diff_l2_mean \
            --target-fps 10 --stride 4 \
            --hysteresis-ratio ${r} --up-count ${u} --down-count ${d} --cooldown-windows 1 --max-duration-seconds 2.0
          conda run -n laps python tools/eval_segmentation.py \
            --pred-root ${OUT}/segmentation_outputs/${VIEW}/r${r}/u${u}_d${d} \
            --gt-dir /home/johnny/action_ws/datasets/gt_segments/${VIEW} \
            --tolerance-sec 2.0 --tolerance-secs 1.0 2.0 3.0 \
            --iou-thrs 0.5 0.75 \
            --output ${OUT}/segmentation_outputs/${VIEW}/r${r}/u${u}_d${d}/eval_summary.json
        done
      done
    done
    ```

  - **命令（聚合敏感性并出图：F1 vs r；F1 vs up/down 热力/曲线）**：
    ```bash
    conda run -n laps python - <<'PY'
    from pathlib import Path
    import json, re
    import numpy as np
    import matplotlib.pyplot as plt

    VIEW = '{VIEW}'; OUT = '{OUT}'
    base = Path(f"{OUT}/segmentation_outputs/{VIEW}")
    tol_key = 'F1@2.0s_mean'  # 主报告口径

    # 读取 (r, u, d) -> F1@2.0s_mean
    data = {}
    for r_dir in base.glob('r*/'):
        mr = re.match(r'r([0-9.]+)', r_dir.name)
        if not mr: continue
        r = float(mr.group(1))
        for ud in r_dir.glob('u*_d*/'):
            mu = re.match(r'u(\d+)_d(\d+)', ud.name)
            if not mu: continue
            u, d = int(mu.group(1)), int(mu.group(2))
            eval_path = ud / 'eval_summary.json'
            if not eval_path.exists():
                continue
            try:
                summ = json.loads(eval_path.read_text(encoding='utf-8'))['summary']
                f1 = float(summ.get(tol_key, np.nan))
            except Exception:
                f1 = np.nan
            data.setdefault(r, {}).setdefault(u, {})[d] = f1

    # 图1：F1 vs r（固定 u=2,d=2，如存在）
    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=300)
    rs, vals = [], []
    for r in sorted(data.keys()):
        f = data.get(r, {}).get(2, {}).get(2, np.nan)
        rs.append(r); vals.append(f)
    ax1.plot(rs, vals, marker='o', lw=2.0, color='#34495E')
    ax1.set_xlabel(r'$r=\\theta_{off}/\\theta_{on}$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1@2.0s (mean)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, ls=':')
    (base / 's5_f1_vs_ratio.png').write_bytes(b'')  # touch
    plt.tight_layout(); plt.savefig(base / 's5_f1_vs_ratio.png', dpi=300, bbox_inches='tight'); plt.close()

    # 图2：F1 vs (u,d)（固定 r=0.95，如存在）
    import itertools
    r_fix = 0.95
    grid_u = [1,2,3]; grid_d = [1,2,3]
    M = np.full((len(grid_u), len(grid_d)), np.nan, float)
    for i,u in enumerate(grid_u):
        for j,d in enumerate(grid_d):
            M[i,j] = data.get(r_fix, {}).get(u, {}).get(d, np.nan)
    fig2, ax2 = plt.subplots(figsize=(6,5), dpi=300)
    im = ax2.imshow(M, cmap='viridis', vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax2.set_xticks(range(len(grid_d))); ax2.set_xticklabels([str(x) for x in grid_d])
    ax2.set_yticks(range(len(grid_u))); ax2.set_yticklabels([str(x) for x in grid_u])
    ax2.set_xlabel('down_count d', fontsize=12); ax2.set_ylabel('up_count u', fontsize=12)
    ax2.set_title('F1@2.0s (mean) @ r=0.95')
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(base / 's5_f1_vs_updown_heatmap.png', dpi=300, bbox_inches='tight'); plt.close()
    print('Saved', base / 's5_f1_vs_ratio.png', 'and', base / 's5_f1_vs_updown_heatmap.png')
    PY
    ```

- 注意事项
  - TV-L1 依赖 `opencv-contrib-python`。若缺失，先在 `laps` 安装：`pip install opencv-contrib-python`。

---

### 1.3 实验 3：Frozen Transformer 嵌入模型超参搜索（Table S2/S3）

- 任务 1.3.1（S2：Pooling 策略对聚类质量）
  - **脚本**：`umap_vis/scripts/sequence_model_embedding.py`
  - **数据**：包含 `**/code_indices/*.codes.json` 的输出根目录（每段含 `quantized_windows`），例如：`/home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW}`
  - **环境**：`laps`
  - **依赖**：`umap-learn`、`plotly`（若缺失：`pip install umap-learn plotly`）
  - **命令（网格搜索，一次性得到不同 pooling 的指标）**：
    ```bash
    conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
      --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW} \
      --fig-dir /home/johnny/action_ws/umap_vis/figure/{VIEW} \
      --stats-dir /home/johnny/action_ws/umap_vis/statistics/{VIEW} \
      --metric cosine --neighbors 15 --min-dist 0.1 \
      --grid-search
    ```
  - **预期输出**：
    - 网格总表：`sequence_model_grid_search.csv`
    - 2D/3D UMAP：`umap_2d_seq_model_grid_best.png`、`umap_3d_seq_model_grid_best.html`
  - **验证**：CSV 中 `pooling=mean` 条目与正文推荐一致，Silhouette 最优或接近最优。

- 任务 1.3.2（S3：不同架构 L/H/d 对聚类质量）
  - **脚本**：同上
  - **命令**：复用网格 CSV，或显式跑一组“最佳配置+k 扩展”的流程：
    ```bash
    conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
      --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW} \
      --fig-dir /home/johnny/action_ws/umap_vis/figure/{VIEW} \
      --stats-dir /home/johnny/action_ws/umap_vis/statistics/{VIEW} \
      --metric cosine --neighbors 15 --min-dist 0.1 \
      --use-best-grid-config --d-model 256 --n-layers 4 --n-heads 4 --pooling mean \
      --k-min 2 --k-max 10 --k-analysis-max 15
    ```
  - **预期输出**：`cluster_metrics_seq_model_cosine.csv` 与 `best_config_metrics_vs_k.png`；据此整理 Table S3。
  - **验证**：`(L=4,H=4,d=256)` 的分数与论文正文一致或更优。

---

## 2. 实验结果的充分补充（Extended Results）

### 2.1 实验 4：更多分割对比（Figure S6–S10）

- 任务 2.1.1（生成多段曲线并挑选）
  - **脚本**：
    - 批量采样：`video_action_segmenter/scripts/batch_plot_energy_segments.py`
    - 单段采样（论文风格）：`video_action_segmenter/scripts/plot_energy_segment_from_jsonl_for_paperteaser.py`
  - **数据**：任一 `{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl`
  - **命令（批量生成 20 张，从中挑选 5 张用于 S6–S10）**：
    ```bash
    conda run -n laps python -m video_action_segmenter.scripts.batch_plot_energy_segments \
      --jsonl {OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl \
      --params ./video_action_segmenter/params_{VIEW}_quant.yaml \
      --segment-length 150 --num-plots 20 \
      --output-dir /home/johnny/action_ws/video_action_segmenter/figures/{VIEW}/paper_segments
    ```
  - **提示**：片段时长（秒）≈ `segment_length * stride / target_fps`。默认 `stride=4, target_fps=10` 时，`segment-length=150` ≈ 60 秒；可调至 225 ≈ 90 秒。
  - **预期输出**：高质量能量曲线 PNG；从中挑选 5 张（60–90s 范围，可用 `plot_energy_comparison.py` 指定 `--duration-sec` 控制窗口）。
  - **验证**：明确包含光流基线 vs 我们的信号（必要时用 `plot_energy_comparison.py` 重绘并叠加 GT）。

-- 任务 2.1.2（Segmentor ON/OFF 叠加，四轨图 必选）
  - **命令（四轨叠加：OF, E_action, GT, Segmentor）**：无需改脚本，内嵌 Python 一键生成。
    ```bash
    conda run -n laps python - <<'PY'
    from pathlib import Path
    import json, numpy as np
    import matplotlib.pyplot as plt

    VIEW='{VIEW}'; OUT='{OUT}'; video='{video}'
    target_fps, stride = 10.0, 4
    start_sec, duration_sec, dpi = 0.0, 90.0, 300

    def load_jsonl(p: Path):
        m = {}
        if not p.exists(): return m
        for ln in p.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                o = json.loads(ln); m[int(o['window'])] = float(o['energy'])
        return m
    def w2s(w): return float(w) * stride / target_fps

    of_path = Path(f"{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_optical_flow_mag_mean.jsonl")
    ae_path = Path(f"{OUT}/energy_sweep_out/{VIEW}/{video}/stream_energy_quantized_token_diff_l2_mean.jsonl")
    gt_path = Path(f"/home/johnny/action_ws/datasets/gt_segments/{VIEW}/{video}_segments.json")
    pred_path = Path(f"{OUT}/segmentation_outputs/{VIEW}/{video}/segmented_videos/{video}_segments.json")

    OF = load_jsonl(of_path); AE = load_jsonl(ae_path)
    GT = json.loads(gt_path.read_text(encoding='utf-8')) if gt_path.exists() else {'segments': []}
    PRED = json.loads(pred_path.read_text(encoding='utf-8')) if pred_path.exists() else {'segments': []}

    s0, e0 = start_sec, start_sec + duration_sec
    def slice_series(m):
        xs, ys = [], []
        for w, e in sorted(m.items()):
            t = w2s(w)
            if s0 <= t <= e0:
                xs.append(t); ys.append(e)
        return np.array(xs), np.array(ys)
    tx_of, ofy = slice_series(OF); tx_ae, aey = slice_series(AE)
    if tx_of.size == 0 or tx_ae.size == 0:
        raise SystemExit('No data in selected range')
    # 归一化
    def norm(a): a=a.astype(np.float64); return (a-a.min())/(a.ptp()+1e-8)
    ofy_n, aey_n = norm(ofy), norm(aey)

    fig, ax = plt.subplots(figsize=(14,6), dpi=dpi)
    ax.plot(tx_of, ofy_n, color='#E74C3C', lw=2.5, label='Optical Flow Magnitude')
    ax.plot(tx_ae, aey_n, color='#3498DB', lw=2.5, label=r"$E_{action}$ (Quantized)")

    # GT 边界
    for seg in GT.get('segments', []):
        a, b = float(seg['start_sec']), float(seg['end_sec'])
        if not (b < s0 or a > e0):
            ax.axvline(max(a,s0), color='#2ECC71', ls='--', lw=1.5, alpha=0.6)
            ax.axvline(min(b,e0), color='#2ECC71', ls='--', lw=1.5, alpha=0.6)
            ax.axvspan(max(a,s0), min(b,e0), color='#2ECC71', alpha=0.08)

    # Segmentor ON/OFF（黑色台阶线）
    segs = [(float(s['start_sec']), float(s['end_sec'])) for s in PRED.get('segments', [])]
    segs = [(max(s0,a), min(e0,b)) for (a,b) in segs if not (b < s0 or a > e0)]
    if segs:
        xs = [s0]; ys = [0.0]
        for a,b in sorted(segs):
            xs += [a,a,b,b]; ys += [0.0,1.0,1.0,0.0]
        xs.append(e0); ys.append(0.0)
        ax.step(xs, ys, where='post', color='black', lw=2.0, alpha=0.9, label='Segmentor (ON=1)')

    ax.set_xlim(s0, e0); ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':'); ax.legend(loc='upper right', fontsize=13)
    import os
    out_dir = Path(f"/home/johnny/action_ws/video_action_segmenter/figures/{VIEW}/four_tracks"); out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f's6to10_four_tracks_{video}_{int(s0)}s_{int(duration_sec)}s.png'
    plt.tight_layout(); plt.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    print('Saved', out_png)
    PY
    ```
  - **验证**：四轨均出现且时间轴一致；Segmentor ON/OFF 与能量峰谷与 GT 边界对应关系清晰。

- 任务 2.1.3（Figure S11：失败案例）
  - **做法**：根据 `eval_summary.json` 中的低分视频挑选 1–2 个，用 `plot_energy_comparison.py` 可视化并在 caption 中说明失败原因（微小动作触发不足/背景扰动误触发）。

### 2.2 实验 5：聚类结果定性（Figure S12+）

- 任务 2.2.1（按簇导出视频样本）
  - **脚本**：`umap_vis/scripts/sequence_model_embedding.py`（`--export-video-samples`）
  - **命令**：
    ```bash
    conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
      --data-dir /home/johnny/action_ws/datasets/output/segmentation_outputs/{VIEW} \
      --fig-dir /home/johnny/action_ws/umap_vis/figure/{VIEW} \
      --stats-dir /home/johnny/action_ws/umap_vis/statistics/{VIEW} \
      --metric cosine --neighbors 15 --min-dist 0.1 \
      --use-best-grid-config --d-model 256 --n-layers 4 --n-heads 4 --pooling mean \
      --k-min 3 --k-max 3 --export-video-samples
    ```
  - **预期输出**：将每簇最多 100 个视频复制到 `/home/johnny/action_ws/classify_res/cluster_{k}/`
  - **验证**：随机抽样视频预览，簇内语义一致性高。

- 任务 2.2.2（生成缩略图网格）
  - **做法**：对每个导出视频，用 `ffmpeg` 抽取起始/中间/结束 3 帧，拼成 3×3/4×4 网格（可用 ImageMagick/自写脚本）。
    ```bash
    # 示例：抽 3 帧
    ffmpeg -y -i input.mp4 -vf "select='eq(n,0)+eq(n,round(n_frames/2))+eq(n,n_frames-1)',scale=320:-1,tile=1x3" -frames:v 1 grid.png
    ```
  - **预期输出**：每簇一张图（S12 起），图注写明语义。

---

## 3. 数据集与讨论（Dataset & Discussion）

### 3.1 实验 6：数据集详情与标注规范（Figure S15, Table S4）

- 任务 3.1.1（S15：工位设置示例帧）
  - **脚本**：`tools/extract_frames_for_view.py`
  - **命令**：
    ```bash
    conda run -n laps python tools/extract_frames_for_view.py \
      --raw-dir /home/johnny/action_ws/datasets/gt_raw_videos/{VIEW} \
      --view {VIEW} \
      --out-dir /home/johnny/action_ws/comapred_algorithm/OTAS/data/breakfast/frames
    ```
  - **预期输出**：`frames/{VIEW}_{VIEW}_{stem}/Frame_*.jpg`；从中挑选代表性 Top-down / Exocentric 帧入图。
  - **补充**：收集/整理一张现场工作照到 `docs/figures/workstation_photo.jpg`，用于 S15(a)。如需马赛克或授权，请在导出前处理。

- 任务 3.1.2（S4：数据统计表）
  - **脚本**：`tools/analyze_hdf5_dataset.py`
  - **命令**：
    ```bash
    conda run -n laps python tools/analyze_hdf5_dataset.py \
      --root-dir /home/johnny/action_ws/data/preprocessed_data_d01 \
      --cfg amplify/cfg/train_motion_tokenizer.yaml \
      --out /home/johnny/action_ws/datasets/output/analysis_report_d01.json --verbose
    ```
  - **预期输出**：统计&一致性检查 JSON（时长、分辨率、轨迹数分布、运动统计、推荐项等）。据此整理 Table S4。
  - **验证**：`recommendations` 列表为空或仅含轻微建议（如尺寸不一致）。

- 任务 3.1.3（标注一致性）
  - **做法**：若多标注者版本可用，计算 F1@tol / mAP@IoU 的互评分；若暂无，文字描述标注规范与质控流程。

---

## 实验顺序与依赖关系

- **优先级 1（可并行）**
  - 光流/行为能量 JSONL 生成（1.2.1）
  - 数据统计（3.1.2）
- **优先级 2（基于能量/GT）**
  - S3/S4 可视化与阈值（1.2.2/1.2.3）
  - 离线分割与评估（1.2.4）
- **优先级 3（嵌入与聚类）**
  - 序列模型网格与最佳配置（1.3.1/1.3.2）
  - 簇样本导出与可视化（2.2）
- **独立**
  - 架构图/训练曲线/超参数表（1.1.x）
  - 工位示例帧与表格（3.1.1/3.1.2）

---

## 注意事项与常见问题（Troubleshooting）

- **环境与依赖**
  - 始终使用 `laps` 环境运行命令。
  - TV-L1 需要 `opencv-contrib-python`；UMAP/Plotly：`pip install umap-learn plotly`。
- **路径与数据**
  - `gt_dir` 需包含 `{stem}_segments.json`。若文件名不一致，可用 `tools/eval_segmentation.py` 内的通配逻辑或重命名。
  - 若已有仓库示例能量（`video_action_segmenter/energy_sweep_out/D02_...`），可直接跳到 1.2.2 之后的步骤。
- **性能与稳定性**
  - 在线推理可调低 `resize_shorter/target_fps`；启用 `amp=true` 减少显存。
  - 分割敏感性建议小网格（3×3×1）即可体现鲁棒性。
- **验证口径**
  - F1 边界容差：报告 `1.0/2.0/3.0s`，IoU：`0.5/0.75`。
  - 阈值曲线峰值与推荐阈值应一致/接近；S6–S10 的对比图应覆盖长序列片段。

---

## 产物清单（预期）

- 图表
  - Figure S1（架构）、S2（训练曲线）、S3（代理 vs 我们 + GT）、S4（阈值曲线）、S5（敏感性）、S6–S10（多段对比）、S11（失败案例）、S12+（簇示例）、S15（工位帧）
- 表格
  - Table S1（超参数）、S2（Pooling 对比）、S3（架构对比）、S4（数据集统计）
- 中间件与报告
  - `best_threshold_quantized_token_diff.json`、`threshold_curve.csv`、`eval_summary.json`、`sequence_model_grid_search.csv` 等

---

## 说明
- 本 ToDoList 充分基于当前代码库的现有脚本与输出格式，所有步骤均可在 `laps` 环境运行完成；不需要 `abd_env` / `otas` 环境。
- 若需扩展：可在不改动核心逻辑的前提下增加小工具（例如将预测分段叠加到对比图的绘图脚本），遵循“最小侵入原则”。
