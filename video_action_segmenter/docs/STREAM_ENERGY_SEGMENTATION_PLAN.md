# 流式动作分段：基于潜在能量曲线的实施方案（逐步推进）

本方案围绕 Motion Tokenizer 的窗口级输出（T=16，stride=4）构建“帧级/准帧级”的能量曲线，用于在长视频/实时流中自动定位动作开始与结束点。文档涵盖从最小可行实现（MVP）到逐步增强的多信号融合与阈值策略，并记录关键注意点与输出规范。

相关代码位置：
- 推理短片脚本：`amplify_motion_tokenizer/inference_short_clip.py`
- 流式脚本（本方案改造点）：`video_action_segmenter/stream_inference.py`
- 参数样例：`video_action_segmenter/params_window_track.yaml`（仅作参考，流式脚本使用独立 `params.yaml`）

---

## 0. 背景与术语

- 输入窗口：`T=16` 帧，速度张量形状 `(T-1, N, 2)`，其中 `N=grid_size^2=400`。
- 编码器序列长度：`d = encoder_sequence_len = 16`，隐藏维度 `D = hidden_dim = 768`（默认）。
- 量化前连续表征：`to_quantize`，形状 `(d, D)`，代表对整窗的可学习摘要（非逐帧对齐）。
- 量化后 latent（用于解码 memory）：`quantized`，形状 `(d, D)`。
- 离散码序列（Latent Action Sequence）：`code_sequences`，形状 `(d,)`，若 FSQ 返回 digits 则经 `_fsq_digits_to_ids()` 合并。

说明：`d` 不等于 `T-1`，`d` 是模型配置的序列长度，`T-1` 是输入速度的时间步数。

---

## 1. 阶段一（MVP）：基于未量化 latent 的窗口能量 E_t

- 能量信号（推荐首选）：对 `to_quantize (d,D)` 计算 token 的 L2 范数并取均值。
  - 记 `Z ∈ R^{d×D}`，`E = mean_i ||Z[i]||_2`。
  - 直觉：运动强/复杂时，潜在幅值整体更大。
- 时间轴：以 stride=4 生成序列 `[E_0, E_4, E_8, ...]`；时间戳对齐到“窗口中心”（t+7 或 t+7.5 帧）。
- 输出：
  - 控制台打印每窗 `E_t`。
  - 可选写入 JSONL：每行 `{win_idx, energy, source='prequant', mode='l2_mean', ts}`。
  - 可选导出 `to_quantize` 到 `.npy`，命名 `prequant_win_{idx:06d}.npy`（形状 `(d,D)`，float32）。
- 阈值分割（最简）：固定阈值对 `E_t` 上/下穿点作为动作起止；离线先进行经验标定。

---

## 2. 阶段二：替代/补充能量信号与融合

- 量化后 latent 能量：`quantized` 上同样计算 `mean_i ||Q[i]||_2`。更稳健但可能被量化压缩。
- 码变化率：相邻窗口 `code_sequences` 的汉明距离/变更比例（反映语义状态变化）。
- 速度能量（对照）：对输入 `vel_norm` 计算 `mean ||v||_2`（窗口内均值），直观反映物理位移。
- 融合策略：`E_fused = α·E_prequant + β·E_code_rate + γ·E_vel`，系数可网格搜索或少量标注调参。

---

## 3. 阶段三：平滑、阈值与段落生成

- 平滑：短窗均值/EMA/Savitzky–Golay，削弱毛刺，增强可分割性。
- 阈值：
  - 固定阈值（需视频间归一化，如 z-score 或分位归一化）。
  - 自适应阈值（Otsu/分位/高低双阈值滞回）。
- 结构约束：
  - 最短段时长、最小间隔、空隙填充与形态学闭运算，抑制碎片化。
  - 起止点细化：在阈上区间边缘附近取局部极小/最大值。

---

## 4. 阶段四：流式系统集成与性能

- 线程模型：主线程采样与显示；后台线程完成“跟踪 + 前向 + 能量 +（可选）写盘”。
- 预算：实时预算 `stride / target_fps`（秒）；打印 `track/fwd/total` 与预算 `budget`，监控是否超时。
- I/O 策略：
  - 能量 JSONL 轻量、顺序追加写即可。
  - `to_quantize` 若启用导出，默认逐窗一文件；可考虑按段或定期 Rolling（后续阶段再做）。

---

## 5. 输出规范与读取

- 能量 JSONL（示例）：
  ```json
  {"win_idx": 42, "energy": 0.3781, "source": "prequant", "mode": "l2_mean", "ts": 1695000000.123}
  ```
- 预量化 latent `.npy`：形状 `(d,D)`，float32；可直接用 `np.load(path)` 读取。
- 控制台日志中显示：`latent(d,D)=(16,768) | energy=... | budget=...`。

---

## 6. 关键注意点

- `d != T-1`：`d=encoder_sequence_len`，当前配置 `d=16`，`T-1=15` 仅是输入速度的时间步数。
- 时间对齐：`E_t` 代表 `[t, t+15]`，分析/可视化时将时间戳放在“窗口中心”。
- 量化差异：不同 FSQ 版本可能返回单一索引或 digits；合并逻辑使用 `_fsq_digits_to_ids()`。
- 归一化一致性：速度归一化按 `decoder_window_size` 半径（见 `inference_short_clip.py::_normalize_velocities`），确保与训练一致。
- 实时稳定性：建议优先使用 `prequant` 能量；如需更鲁棒可与 `code_rate/velocity` 融合并加滞回阈值。
- 存储开销：`to_quantize` 导出可控，默认关闭；按需开启并设置输出目录。

---

## 7. 实施清单（与代码映射）

- [x] 在 `stream_inference.py` 中新增前向路径，输出 `to_quantize (d,D)` 与 `quantized (d,D)`、可选 `codes (d,)`。
- [x] 计算每窗能量 `E_t`（默认 `prequant + l2_mean`）。
- [x] 控制台打印能量；可选写入 JSONL（`stream_energy.jsonl`）。
- [x] 可选导出 `to_quantize` 为 `.npy`（`stream_prequant/`）。
- [ ] 后续：平滑、阈值、滞回与段落生成的在线实现（另行迭代）。

---

如需将阈值/滞回/平滑参数纳入配置文件（`params.yaml`），建议新增：
- `energy.source`: `prequant|quantized|velocity`
- `energy.mode`: `l2_mean|token_diff_l2_mean`
- `energy.jsonl_output`: `true|false`
- `energy.jsonl_path`: 路径
- `export.prequant`: `true|false`
- `export.prequant_dir`: 路径
