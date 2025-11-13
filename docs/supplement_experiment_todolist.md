# Supplement Experiments To-Do and Progress

更新时间: 2025-11-12

注意：所有补充材料产物统一输出至 supplement_output/，并记录生成命令与参数。环境：laps/abd_env/otas 按规则切换。

- Fig S2 — Motion Tokenizer 训练损失曲线
  - 状态: 已完成
  - 输入: supplement_output/tokenizer/loss_curves.csv
  - 脚本: scripts/supplement/plot_loss_from_csv.py
  - 环境: base
  - 输出: supplement_output/tokenizer/fig_S2_loss.pdf, loss_curves.csv
  - 备注: 最初在 laps 环境绘图失败（numpy/matplotlib 二进制不兼容），已改为 base 环境从 CSV 渲染


- Fig S3 — 代理信号 vs E_action 可视化
  - 状态: 已完成
  - 输入: energy_sweep_out 与 best_threshold 报告
  - 脚本: scripts/supplement/plot_proxy_vs_e_action.py
  - 环境: base
  - 输出: supplement_output/segmentor/fig_S3_proxy_vs_e_action_D01_sample_2_seg001.pdf, supplement_output/segmentor/fig_S3_proxy_vs_e_action_D02_sample_1_seg001.pdf

- Fig S4 — θ_on 参数扫描
  - 状态: 已完成
  - 输入: energy_sweep_out（optical_flow mag_mean）与 GT 标注
  - 脚本: scripts/supplement/sweep_theta_on.py
  - 环境: base
  - 输出: supplement_output/segmentor/fig_S4_theta_sweep.pdf, supplement_output/segmentor/fig_S4_theta_sweep.csv

- Fig S5 — 滞回/去抖动参数敏感性
  - 状态: 已完成
  - 输入: energy_sweep_out（optical_flow mag_mean）与 GT 标注；θ_on 来自 fig_S4_theta_sweep.csv
  - 脚本: scripts/supplement/sweep_sensitivity.py
  - 关键发现: 最优参数（F1）= hr 0.65, up 1, down 5；F1=0.1996，J=0.1110（详见日志）

  - 环境: base
  - 输出: supplement_output/segmentor/fig_S5a_sensitivity_f1.pdf, supplement_output/segmentor/fig_S5b_sensitivity_jindex.pdf, supplement_output/segmentor/fig_S5_sensitivity.csv



- Table S1 — Motion Tokenizer 训练超参数
  - 状态: 已完成
  - 输入: wandb config.yaml (D01, D02)
  - 脚本: scripts/supplement/extract_hparams_to_csv.py
  - 输出: supplement_output/tokenizer/table_S1_hparams.csv


- Table S2/S3 — Frozen Transformer 池化策略与 K 值搜索
  - 状态: 已完成
  - 输入: datasets/output/segmentation_outputs/D01_LAPS 与 D02_LAPS 下的 code_indices/*.codes.json（quantized_windows）
  - 脚本: scripts/supplement/k_search.py
  - 环境: 计算=laps；绘图=base
  - 输出: supplement_output/clusters/table_S2_k_search_metrics.csv, supplement_output/clusters/fig_k_search.pdf

- Table S2/S3 — Cluster 扩展指标（新数据源 Online Inference）
  - 状态: 已完成
  - 输入: /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector 下的 code_indices/*.codes.json（quantized_windows 或 quantized_vectors）
  - 脚本: scripts/supplement/k_search.py
  - 环境: 计算=laps；绘图=base
  - 输出: supplement_output/clusters/table_S2_k_search_extended_metrics.csv, supplement_output/clusters/fig_k_search_extended.pdf
  - 指标: silhouette, davies_bouldin, calinski_harabasz, intra_dist, inter_centroid_dist, intra_over_inter

- Table S2 — Pooling & Metric 对比（4×2 组合）
  - 状态: ✅ 已完成
  - 代码更新：
    - umap_vis/scripts/sequence_model_embedding.py：TinyTransformer.encode_one() 新增 max pooling 分支（element-wise max）
    - scripts/supplement/pooling_metric_compare.py：--poolings choices/default 扩展为 ["mean", "cls", "attn", "max"]
  - 输入: 同上新数据源（online_inference_output，D02 共 6444 段）
  - 脚本: scripts/supplement/pooling_metric_compare.py
  - 环境: 计算=laps；绘图=base
  - 输出: supplement_output/clusters/table_S2_pooling_metric_comparison.csv, supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf
  - 组合: pooling ∈ {mean, cls, attn, max} × metric ∈ {euclidean, cosine}
  - 代表性指标（按每种 pooling 的 Silhouette 最高行）：
    - Mean Pooling (Ours): Silhouette=0.547, CH=3327.9（cosine, K=3）
    - CLS Token Pooling: Silhouette=0.135, CH=515.6（cosine, K=2）
    - Max Pooling: Silhouette=0.314, CH=1253.0（cosine, K=3）
    - Attention Pooling: Silhouette=0.362, CH=1838.7（cosine, K=2）
  - LaTeX 表：tab:pooling_strategy 数值已填充




- Table S2 — Frozen Transformer 架构影响（L/H/d 搜索 + K 搜索，pooling=mean）
  - 状态: 已完成
  - 输入: datasets/output/segmentation_outputs/D02_LAPS 下的 code_indices/*.codes.json（quantized_windows）
  - 脚本: umap_vis/scripts/sequence_model_embedding.py（--use-best-grid-config + 显式 --d-model/--n-layers/--n-heads 覆盖）
  - 环境: 计算=laps（绘图在 laps 因 numpy/matplotlib 二进制不兼容报错；不影响 CSV；如需图请在 base 渲染）
  - 输出: supplement_output/transformer_search/transformer_architecture_comparison.csv；以及每配置子目录下 stats/cluster_metrics_seq_model_cosine.csv 与 stats/best_config_k_analysis.csv
  - 备注: 高亮配置 (L=4,H=4,d=256) 数值来自 umap_vis/statistics/d02/sequence_model_grid_search.csv（best_k=3, silhouette≈0.5998, CH≈4015.65），与既有结果一致

- 后续（占位）
  - Table S2/S3（Frozen Transformer 搜索+K搜索）
  - Fig S6–S11（定性对比/失败案例）
  - Fig S12–S14（聚类定性）
  - Fig S1、Fig S15、Table S4（人工整理/收集）

记录规范：
- 每次生成后，追加“产物路径、生成命令、时间戳、环境、输入摘要”。



进度日志：
- 2025-11-12 生成：
  - 创建 supplement_output/ 目录结构与 README；新增脚本 scripts/supplement/plot_loss_curve.py、extract_hparams_to_csv.py
  - 执行命令（env=laps）：
    - conda run -n laps python scripts/supplement/extract_hparams_to_csv.py --runs wandb/run-20251026_153549-ywfngyyh wandb/run-20251026_153814-3ydzlt14 --out supplement_output/tokenizer/table_S1_hparams.csv
      - 产物：supplement_output/tokenizer/table_S1_hparams.csv（完成）
- 2025-11-12 渲染 Fig S2（env=base）：
  - 命令：conda run -n base python scripts/supplement/plot_loss_from_csv.py --csv supplement_output/tokenizer/loss_curves.csv --outpdf supplement_output/tokenizer/fig_S2_loss.pdf
  - 产物：supplement_output/tokenizer/fig_S2_loss.pdf（完成）

- 2025-11-12 生成 Fig S3（env=base）：
  - 命令：
    - conda run -n base python scripts/supplement/plot_proxy_vs_e_action.py --sample_dir datasets/output/energy_sweep_out/D01/D01_sample_2_seg001 --report_dir datasets/output/energy_sweep_report/D01 --outpdf supplement_output/segmentor/fig_S3_proxy_vs_e_action_D01_sample_2_seg001.pdf
    - conda run -n base python scripts/supplement/plot_proxy_vs_e_action.py --sample_dir datasets/output/energy_sweep_out/D02/D02_sample_1_seg001 --report_dir datasets/output/energy_sweep_report/D02 --outpdf supplement_output/segmentor/fig_S3_proxy_vs_e_action_D02_sample_1_seg001.pdf
  - 产物：
    - supplement_output/segmentor/fig_S3_proxy_vs_e_action_D01_sample_2_seg001.pdf（完成）
    - supplement_output/segmentor/fig_S3_proxy_vs_e_action_D02_sample_1_seg001.pdf（完成）


- 2025-11-12 生成 Fig S4（env=base）：
  - 命令：conda run -n base python scripts/supplement/sweep_theta_on.py --sample_roots datasets/output/energy_sweep_out/D01 datasets/output/energy_sweep_out/D02 --gt_dir datasets/gt_annotations --energy_file stream_energy_optical_flow_mag_mean.jsonl --theta_min 0.5 --theta_max 3.0 --theta_step 0.1 --outpdf supplement_output/segmentor/fig_S4_theta_sweep.pdf --outcsv supplement_output/segmentor/fig_S4_theta_sweep.csv


  - 产物：supplement_output/segmentor/fig_S4_theta_sweep.pdf（完成）；supplement_output/segmentor/fig_S4_theta_sweep.csv（完成）


- 2025-11-12 生成 Fig S5（env=base）：
  - 命令：conda run -n base python scripts/supplement/sweep_sensitivity.py --sample_roots datasets/output/energy_sweep_out/D01 datasets/output/energy_sweep_out/D02 --gt_dir datasets/gt_annotations --energy_file stream_energy_optical_flow_mag_mean.jsonl --theta_csv supplement_output/segmentor/fig_S4_theta_sweep.csv --outpdf_f1 supplement_output/segmentor/fig_S5a_sensitivity_f1.pdf --outpdf_j supplement_output/segmentor/fig_S5b_sensitivity_jindex.pdf --outcsv supplement_output/segmentor/fig_S5_sensitivity.csv
  - 产物：supplement_output/segmentor/fig_S5a_sensitivity_f1.pdf（完成）；supplement_output/segmentor/fig_S5b_sensitivity_jindex.pdf（完成）；supplement_output/segmentor/fig_S5_sensitivity.csv（完成）

  - 关键发现：最优参数（F1）= hr 0.65, up 1, down 5；F1=0.1996，J=0.1110

- 2025-11-12 生成 Table S2/S3（K=2..10，Frozen Transformer mean pooling；env=laps 计算 + base 绘图）：
  - 命令：
    - conda run -n laps python scripts/supplement/k_search.py --data_roots datasets/output/segmentation_outputs/D01_LAPS datasets/output/segmentation_outputs/D02_LAPS --mode seq_model --pooling mean --d_model 256 --n_layers 4 --n_heads 4 --k_min 2 --k_max 10 --metric cosine --stage compute --out_csv supplement_output/clusters/table_S2_k_search_metrics.csv --out_pdf supplement_output/clusters/fig_k_search.pdf
    - conda run -n base python scripts/supplement/k_search.py --stage plot --out_csv supplement_output/clusters/table_S2_k_search_metrics.csv --out_pdf supplement_output/clusters/fig_k_search.pdf
  - 产物：supplement_output/clusters/table_S2_k_search_metrics.csv（完成）；supplement_output/clusters/fig_k_search.pdf（完成）
  - 备注：共加载 786 段，向量维=768；Silhouette 与 CH 指标随 K 的曲线已绘制

  - 说明：D01/D02 分别计算后取宏平均；子图布局动态调整避免空白子图


- 2025-11-12 重新生成 Fig S5（env=base，凑满子图=8 个 hr）：
  - 命令：conda run -n base python scripts/supplement/sweep_sensitivity.py --sample_roots datasets/output/energy_sweep_out/D01 datasets/output/energy_sweep_out/D02 --gt_dir datasets/gt_annotations --energy_file stream_energy_optical_flow_mag_mean.jsonl --theta_csv supplement_output/segmentor/fig_S4_theta_sweep.csv --hysteresis_ratio 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --outpdf_f1 supplement_output/segmentor/fig_S5a_sensitivity_f1.pdf --outpdf_j supplement_output/segmentor/fig_S5b_sensitivity_jindex.pdf --outcsv supplement_output/segmentor/fig_S5_sensitivity.csv
  - 产物：S5a/S5b 单页 PDF 与 CSV 已覆盖更新

- 2025-11-12 生成 Table S2/S3 扩展（K=2..10；env=laps 计算 + base 绘图；数据源=online_inference_output）：
  - 命令：
    - conda run -n laps python scripts/supplement/k_search.py --data_roots /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector --mode seq_model --pooling mean --d_model 256 --n_layers 4 --n_heads 4 --k_min 2 --k_max 10 --metric cosine --stage compute --out_csv supplement_output/clusters/table_S2_k_search_extended_metrics.csv --out_pdf supplement_output/clusters/fig_k_search_extended.pdf
    - conda run -n base python scripts/supplement/k_search.py --stage plot --out_csv supplement_output/clusters/table_S2_k_search_extended_metrics.csv --out_pdf supplement_output/clusters/fig_k_search_extended.pdf
  - 产物：supplement_output/clusters/table_S2_k_search_extended_metrics.csv（完成）；supplement_output/clusters/fig_k_search_extended.pdf（完成）
  - 备注：共加载 6444 段，向量维=768；已计算 6 指标并绘制 2×3 子图



- 2025-11-12 更新 Table S2 — Pooling & Metric 对比（4×2 组合；K=2..10；env=laps 计算 + base 绘图；数据源=online_inference_output）：
  - 命令：
    - conda run -n laps python scripts/supplement/pooling_metric_compare.py --data_roots /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector --poolings mean cls attn max --distance_metrics euclidean cosine --k_min 2 --k_max 10 --stage compute --out_csv supplement_output/clusters/table_S2_pooling_metric_comparison.csv --out_pdf supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf
    - conda run -n base python scripts/supplement/pooling_metric_compare.py --stage plot --out_csv supplement_output/clusters/table_S2_pooling_metric_comparison.csv --out_pdf supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf
  - 产物：supplement_output/clusters/table_S2_pooling_metric_comparison.csv（完成；数据行=72）；supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf（完成）
  - 组合：pooling ∈ {mean, cls, attn, max} × metric ∈ {euclidean, cosine}
  - 代表性指标：Mean 0.547 / CH 3327.9（cosine, K=3）；CLS 0.135 / CH 515.6（cosine, K=2）；Max 0.314 / CH 1253.0（cosine, K=3）；Attn 0.362 / CH 1838.7（cosine, K=2）
  - LaTeX：tab:pooling_strategy 已填充值


- 2025-11-12 生成 Table S2 — Frozen Transformer 架构影响（env=laps；数据源=D02_LAPS；pooling=mean；K=2..10 + k-analysis-max=15）：
  - 命令：
    - conda run -n laps python umap_vis/scripts/sequence_model_embedding.py --data-dir datasets/output/segmentation_outputs/D02_LAPS --fig-dir supplement_output/transformer_search/L2_H2_d128/figure --stats-dir supplement_output/transformer_search/L2_H2_d128/stats --metric cosine --neighbors 15 --min-dist 0.1 --use-best-grid-config --d-model 128 --n-layers 2 --n-heads 2 --pooling mean --device cpu --k-min 2 --k-max 10 --k-analysis-max 15
    - conda run -n laps python umap_vis/scripts/sequence_model_embedding.py --data-dir datasets/output/segmentation_outputs/D02_LAPS --fig-dir supplement_output/transformer_search/L4_H4_d128/figure --stats-dir supplement_output/transformer_search/L4_H4_d128/stats --metric cosine --neighbors 15 --min-dist 0.1 --use-best-grid-config --d-model 128 --n-layers 4 --n-heads 4 --pooling mean --device cpu --k-min 2 --k-max 10 --k-analysis-max 15
    - conda run -n laps python umap_vis/scripts/sequence_model_embedding.py --data-dir datasets/output/segmentation_outputs/D02_LAPS --fig-dir supplement_output/transformer_search/L6_H8_d256/figure --stats-dir supplement_output/transformer_search/L6_H8_d256/stats --metric cosine --neighbors 15 --min-dist 0.1 --use-best-grid-config --d-model 256 --n-layers 6 --n-heads 8 --pooling mean --device cpu --k-min 2 --k-max 10 --k-analysis-max 15
    - conda run -n laps python umap_vis/scripts/sequence_model_embedding.py --data-dir datasets/output/segmentation_outputs/D02_LAPS --fig-dir supplement_output/transformer_search/L6_H8_d512/figure --stats-dir supplement_output/transformer_search/L6_H8_d512/stats --metric cosine --neighbors 15 --min-dist 0.1 --use-best-grid-config --d-model 512 --n-layers 6 --n-heads 8 --pooling mean --device cpu --k-min 2 --k-max 10 --k-analysis-max 15
  - 产物：
    - supplement_output/transformer_search/transformer_architecture_comparison.csv（汇总 5 行）
    - 每配置子目录 stats/cluster_metrics_seq_model_cosine.csv 与 stats/best_config_k_analysis.csv
  - 备注：绘图保存在 laps 环境 tight_layout() 处因 numpy/matplotlib 二进制不兼容报错终止，但在保存 CSV 之后发生；不影响数据完整性。若需图可用 conda run -n base 复现绘图阶段。
