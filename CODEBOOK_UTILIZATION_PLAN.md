# 码本塌陷诊断与改进方案（分阶段）

本文档用于记录当前训练中“码本塌陷”的现象、诊断依据、以及分阶段的代码改进方案与监控指标。所有改动均基于现有代码结构：
- 模型入口：`amplify_motion_tokenizer/models/motion_tokenizer.py`
- 训练脚本：`amplify_motion_tokenizer/train.py`
- 数据集与掩码：`amplify_motion_tokenizer/dataset/velocity_dataset.py`

---

## 一、现状与诊断

- 训练期间的指标（TensorBoard 截图）：
  - `train/unique_codes_step` 长期约为 1；`train/unique_codes_epoch_local` 快速衰减到 ~1；`train/entropy_step`、`train/entropy_norm_step` 接近 0。
  - `train/effective_codebook_size_seen` ≈ 2048（由 `fsq_levels=[8,8,8,4]` 所致），仅表明“最大 id 出现过”，并不代表整体分布均匀。
  - `train/acc_epoch` ≈ 0.60，与数据分析中 `center_prop ≈ 0.6035` 高度吻合，提示模型倾向输出中心类。
- 数据侧（启用静态点过滤后）：
  - `dynamic_point_ratio≈0.47`，`perplexity_ratio≈0.052`，`center_prop≈0.6035`。较未过滤有明显改善，但中心类仍占多数。

结论：量化后的 code 分布极端单峰，码本严重塌陷；模型对中心类倾向性强，导致 token 与解码端均朝“中心”聚集。

---

## 二、根因分析（可能）

- **[标签不均衡]** 中心类仍然占比较高（~0.60），对模型构成强诱导。
- **[缺乏正则]** 训练目标中未实际加入“鼓励多码使用”的正则项（目前仅记录熵，不参与反传）。
- **[量化输入分布]** 送入 FSQ 的特征幅度/分布可能过于集中（接近 0 或方差过小），导致落入极少数 code。
- **[优化动态]** 学习率、调度与噪声水平可能不足以探索更多 code。

---

## 三、改进路线（分阶段实施）

> 原则：从低风险、易回滚的“损失层面微调”开始，逐步推进至模型结构与数据层优化；每一阶段都有明确的验收指标与回滚策略。

### 阶段 0｜可观测性完善（无需改变训练目标）

- **[增加训练期监控]**
  - 在 `train.py` 记录：
    - `train/dynamic_point_ratio_step`（本步掩码为 True 的比例）。
    - `val/dynamic_point_ratio_epoch`（验证集按 epoch 聚合的动态点比例）。
  - 在 `train.py` 增加“跨进程聚合”的码本统计（可选）：
    - 使用 `accelerate.gather()` 聚合全进程 `code_ids` 后再计算 `entropy/perplexity/unique_codes`。
  - 目的：确认塌陷是否在所有进程一致，以及掩码在训练期的实际占比波动。

- **验收标准**：日志能稳定产出；无需改变损失。

---

### 阶段 1｜损失层面微调（低风险，建议先做）

- **[熵正则纳入反传]**
  - 在 `train.py` 的每步中，使用现有的 `code_ids` 分布 `p` 计算归一化熵 `H_norm = H / log(K)`。
  - 将总损失改为：`loss_total = CE - λ · H_norm`。
  - 新增配置：
    - `training.code_entropy_weight`（浮点，默认 0.0）。
    - `training.code_entropy_warmup_epochs`（整型，默认 0；线性 warmup λ 从 0 逐步升至设定值）。
  - 建议起始值：λ=0.02，warmup=10 个 epoch；若仍塌缩，逐步加到 0.05~0.1。

- **[替代/补充：对齐均匀分布的 KL]**
  - 方案 1 的替代或并行项：`loss_total = CE + λ · KL(p || U)`（U 为均匀分布）。
  - 二者效果相近，实际取其一即可；建议先用熵正则，简单直观。

- **[日志与排查]**
  - 新增 `train/loss_total_step`（用于对比 CE 与加正则的影响）。
  - 观测 `train/entropy_norm_step` 是否上升、`train/unique_codes_step` 是否>1 并逐步提升。

- **验收标准**：
  - 10 个 epoch 内，`train/unique_codes_step` 明显升高，`entropy_norm_step` 上升至 ≥0.2；`acc_epoch` 不显著恶化（降幅 < 0.02）。
  - 若训练不稳或指标恶化，则逐步下调 λ 或延长 warmup，必要时回滚。

---

### 阶段 2｜标签不均衡对策（中风险，按需开启）

- **[类别重加权]**
  - 在 `train.py` 提供损失函数切换：
    - `training.class_weighting.method: none|balanced|ring_decay`。
    - `balanced`：根据离线统计（`analysis/analyze_label_distribution.py` 的 `before.class_counts`）计算类别权重 `w_c ∝ 1 / freq_c`。
    - `ring_decay`：中心类与近邻环级别降权（根据 `velocity_to_labels` 的几何映射确定 ring，再按 ring index 分配权重）。
  - 可选参数：`training.class_weighting.center_gamma`（中心权重缩放），`training.class_weighting.min_weight`（防极端）。

- **[Focal Loss]**
  - 配置：`training.focal.enabled: true|false`，`training.focal.gamma`（如 1.5~2.0），`training.focal.alpha`（对少数类加权）。
  - 替换或叠加至 CE（注意与上面的类别重加权交互）。

- **验收标准**：
  - `perplexity_ratio`、`entropy_norm_step` 持续提升；`center_prop` 降至 ≤0.55；整体 `acc_epoch` 不显著下降。

---

### 阶段 3｜量化前特征分布整形（中风险，效果通常明显）

- **[输入归一化/拉伸]**（`models/motion_tokenizer.py` 在 FSQ 前）
  - 新增可选模块 `QuantInputNorm`：
    - `none | layernorm | rmsnorm`（对 `(B,d,D)` 的最后一维归一化）。
  - 新增可选“有界拉伸”门：`quant_input_tanh_scale`
    - `y = tanh(s · x)`，`s` 可学习，初值 1.0，`s` 限制在 `[s_min, s_max]`（如 `[0.5, 3.0]`）。
  - 可选噪声：`quant_input_noise_std`（训练期在 FSQ 前加入小高斯噪声，如 `std=0.01`）。

- **配置建议**：
  - `model.quant_input_norm: "layernorm"`；
  - `model.quant_input_tanh_scale: { enabled: true, init: 1.0, s_min: 0.5, s_max: 3.0, trainable: true }`；
  - `model.quant_input_noise_std: 0.01`（可从 0 开始，必要时再加）。

- **验收标准**：
  - `unique_codes_step`、`entropy_norm_step` 明显回升；`code_usage_hist` 呈多峰；训练稳定性无明显降低。

---

### 阶段 4｜数据侧增强（低~中风险）

- **[静态点过滤的精细化]**
  - 当前（`p95@0.04, min_keep=16`）已将 `center_prop` 压至 ~0.60。若阶段 1~3 仍不足，可：
    - 继续小步提高阈值（`p95@0.05`）或试 `metric=mean@0.03`（强调持续运动）。
    - 目标参考：`dynamic_point_ratio ∈ [0.45, 0.60]`、`center_prop ≤ 0.55`、`perplexity_ratio ≥ 0.10`。

- **[采样与增强]**（可选）
  - Clip 级采样时对“动态占比高”的样本给予更高采样权重（需在自定义采样器中实现）。
  - 速度域轻微抖动/缩放（需确保与标签映射一致，不引入偏移）。

---

### 阶段 5｜训练策略（低风险）

- **[调度与稳定性]**
  - 梯度裁剪（如 `clip_grad_norm=1.0`）。
  - 学习率 warmup + cos/step 衰减（若尚未使用）。
  - 观察 `bf16/fp16` 的数值稳定性，如需可暂退 `mixed_precision=no` 做对照。

---

## 四、监控面板（建议）

- 码本利用：
  - `train/unique_codes_step`、`train/unique_codes_raw_step`、`train/entropy_step`、`train/entropy_norm_step`、`train/perplexity_step`、`train/code_usage_hist`、`train/effective_codebook_size_seen`。
- 训练效果：
  - `train/acc_step`、`train/acc_epoch`、`train/loss_step`、`train/loss_epoch`、`val/*`。
- 掩码占比：
  - `train/dynamic_point_ratio_step`、`val/dynamic_point_ratio_epoch`（阶段 0 新增）。

---

## 五、实施顺序与回滚策略

1. **阶段 0**：仅加日志（可立即合并）。
2. **阶段 1**：熵正则加入反传（默认 λ=0，代码合入后由 YAML 开启）。观察 5~10 epoch：
   - 若指标改善：保留；
   - 若不稳：减小 λ 或延长 warmup；若仍不稳，回滚至 λ=0。
3. **阶段 2**：类别重加权/焦点损失（与阶段 1 不冲突，分支试验）。
4. **阶段 3**：量化前整形（逐项开关，单独 ablation）。
5. **阶段 4~5**：数据与训练策略微调。

---

## 六、配置项汇总（新增/修改）

- 训练：
  - `training.code_log_interval: int`（默认等于 `training.log_interval`）
  - `training.code_entropy_weight: float`（默认 0.0）
  - `training.code_entropy_warmup_epochs: int`（默认 0）
  - `training.class_weighting.method: str`（`none|balanced|ring_decay`）
  - `training.class_weighting.center_gamma: float`（可选）
  - `training.class_weighting.min_weight: float`（可选）
  - `training.focal.enabled: bool`（默认 false）
  - `training.focal.gamma: float`、`training.focal.alpha: float|None`
- 模型：
  - `model.quant_input_norm: str`（`none|layernorm|rmsnorm`）
  - `model.quant_input_tanh_scale: { enabled, init, s_min, s_max, trainable }`
  - `model.quant_input_noise_std: float`（默认 0.0）

---

## 七、验收指标（阶段性目标）

- 10 个 epoch 内：
  - `train/unique_codes_step ≥ 8` 且逐步上升；
  - `train/entropy_norm_step ≥ 0.2`；
  - `train/perplexity_step` 明显上升；
  - `train/acc_epoch` 变化在可接受范围（降幅 < 0.02）。
- 30 个 epoch 内：
  - `unique_codes_step ≥ 32`；
  - `entropy_norm_step ≥ 0.35`；
  - `code_usage_hist` 呈多峰，不再单峰贴边；
  - 验证集指标无明显退化。

---

## 八、开发与测试清单（Checklists）

- **阶段 0（日志）**
  - [ ] `train.py` 记录 `train/dynamic_point_ratio_step`、`val/dynamic_point_ratio_epoch`。
  - [ ] （可选）跨进程聚合 code 分布并记录 `*_global` 指标。

- **阶段 1（熵正则）**
  - [ ] `train.py` 计算 `H_norm` 并纳入损失；实现 warmup 调度；新增 YAML 项。
  - [ ] 保留 `train/total_loss_step` 日志并新增 `train/loss_total_step`（实际反传的总损失）。
  - [ ] 用 5~10 epoch 做冒烟，检查稳定性。

- **阶段 2（不均衡）**
  - [ ] 支持 `balanced` 类权重（从分析报告或运行期累计频率导出）。
  - [ ] 支持 `focal loss` 与参数配置。

- **阶段 3（量化前整形）**
  - [ ] 在 `motion_tokenizer.py` 中插入可选的 `QuantInputNorm`、tanh 拉伸、噪声。
  - [ ] Ablation：独立开关对比。

- **阶段 4（数据）**
  - [ ] `static_filter` 小步调整（p95@0.05 或 mean@0.03），观察 `center_prop/perplexity_ratio`。
  - [ ] （可选）采样器按 `dynamic_point_ratio` 加权。

- **阶段 5（策略）**
  - [ ] 学习率调度、梯度裁剪对照实验。

---

## 九、参考与注意事项

- FSQ 索引获取已在 `MotionTokenizer.forward()` 中做了版本兼容（支持/不支持 `return_indices=True`）。
- 所有新增项应默认“关闭”（或 λ=0），以便回滚；通过 YAML 精确开启。
- 任何引入随机性的模块（如噪声）请仅在训练模式启用（`self.training`）。

---

如需，我可以按上述顺序逐步实现并提交，每一步都会在 TensorBoard 上加入必要的指标，便于你评估与决策下一步。
