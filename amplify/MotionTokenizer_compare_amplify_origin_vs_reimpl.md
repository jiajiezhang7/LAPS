# Motion Tokenizer 对比报告

对比对象：
- 源实现（官方开源）：`amplify_origin/`
  - 主要文件：
    - `amplify/models/motion_tokenizer.py`
    - `train_motion_tokenizer.py`
    - `cfg/train_motion_tokenizer.yaml`
    - `amplify/utils/data_utils.py`
    - `amplify/models/losses.py`
    - `amplify/models/transformer.py`
    - `amplify/utils/model/attn_masks.py`
- 复现实现（你的实现）：`amplify_motion_tokenizer/`
  - 主要文件：
    - `models/motion_tokenizer.py`
    - `models/components.py`
    - `train.py`
    - `dataset/velocity_dataset.py`
    - `utils/helpers.py`
    - `configs/tokenizer_config*.yaml`

---

## 结论速览

- 相同点（核心思想一致）：
  - **VAE 思路**：编码器 → FSQ 离散化 → 解码器，均以**局部窗口分类**（W×W）学习相对位移（速度）。
  - **Transformer 骨干**：均使用 Transformer Encoder/Decoder 结构与自注意/交叉注意。
  - **FSQ 量化**：均采用 `vector_quantize_pytorch.FSQ`，隐空间维度均为 `hidden_dim`，离散序列长度为 `d`。
  - **基础输入**：都使用归一化速度（相对位移）作为输入训练信号（非直接位置）。

- 关键差异（会影响行为/指标/兼容性）：
  - **时序与点的建模粒度**：
    - 源实现将“时间步（×视角）”作为 token（`VAEEncoder` 中把 `num_tracks*point_dim` 拼到特征维；token 数≈`T-1` 或 `views*(T-1)`）。
    - 复现实现将“关键点”作为解码查询，并将时间维在编码后**线性投影压缩**到长度 `d`（`encoder_output_projection: S=(T-1)*N → d`）。解码时对每个点输出一组 logits，并在时间维上**复制**到 `T-1`（时序上同窗内各步一致）。
  - **是否支持多视角/图像条件**：
    - 源实现支持 `per_view`、`cond_on_img`，可用 `VisionEncoder` 提供图像 token 并在编码器内做跨注意。
    - 复现实现目前不包含图像条件与多视角逻辑（单视角、仅速度）。
  - **损失函数**：
    - 源实现：`relative_ce`（交叉熵），按视角权重聚合；目标来自像素空间的相对位置差（`get_autoregressive_indices_efficient`）。
    - 复现实现：**Focal Loss**（γ=2，支持对中心类别加权），并引入**静态点掩码**与可选**码本熵正则**（启发式、非可导）。
  - **标签/归一化细节**：
    - 源实现：先将轨迹从归一化坐标还原到像素，再计算相对位移并映射到 `rel_cls_img_size` 网格索引。
    - 复现实现：直接从**已归一化到 [-1,1] 的速度**经 `decoder_window_size (W)` 映射为标签（`velocity_to_labels`）。
  - **解码器查询与时间维表达**：
    - 源实现：以**全零序列**作为查询，长度=`views*(T-1)`，经自注意与跨注意（到 codes）后叠加点位与视角嵌入，再 MLP 输出**逐时刻** logits。
    - 复现实现：以**N 个可学习点查询**作为目标序列，跨注意到 codes，输出**每点** logits，并复制到 `T-1`。

---

## 模型架构逐项对比

### 输入表示与 Token 设计
- 源实现（`amplify/models/motion_tokenizer.py`）
  - 输入：`x` 为速度 `(b, v, t, n, d)`，其中 `t = track_pred_horizon - 1`。
  - `VAEEncoder` 中：
    - `per_view=true` → `x` 重排为 `b (v t) (n d)`；`per_view=false` → `b t (v n d)`。
    - 即：**时间步（×视角）是 token 维**，每个 token 的特征维包含 `num_tracks*point_dim`（以及视角拼接）。
  - `num_timesteps = cfg.track_pred_horizon - 1`；解码器 `q_seq_len = kv_seq_len = views*(T-1)`。

- 复现实现（`amplify_motion_tokenizer/models/motion_tokenizer.py`）
  - 输入：速度 `(B, T-1, N, 2)`。
  - 展平为序列 `S = (T-1)*N`，线性投影到 `hidden_dim` 并加学习位置编码 `pos_embed: (1, S, D)`。
  - 经过 `KeypointEncoder`（内置 `nn.TransformerEncoder`），随后通过 `encoder_output_projection: (B, D, S) → (B, D, d)` 把序列长度压缩到 `d = encoder_sequence_len`。
  - 量化后的 memory 长度为 `d`；解码器以 **N 个可学习查询**作为目标序列，并跨注意 memory 得到每点表示。

【影响与评注】
- 源实现对**时间维**的表示更直接（每步一个 token），能生成**逐时刻不同**的位移分类。
- 复现实现将**时间维压缩**后再解码到“点”，随后再**在时间上复制**，因此**同窗内各步的标签一致**，这与论文/源码“逐步相对位移分类”的设定有一定偏差。

### 编码器（Encoder）
- 源实现：
  - `TransformerEncoder` 或带条件的 `TransformerDecoder`（当 `cond_on_img=true`），掩码可选 `causal_mask` / `diag_cond_mask` / `None`。
  - 支持图像条件：`VisionEncoder` 提供 cond tokens，编码器以 Q（轨迹）对 KV（图像）跨注意。

- 复现实现：
  - 使用内置 `nn.TransformerEncoder`（`KeypointEncoder`），因 token 序列是 `S=(T-1)*N`，使用 `generate_square_subsequent_mask(S)` 实现因果掩码。
  - 不包含图像条件分支。

### 量化（FSQ）
- 源实现：
  - `FSQ(dim=hidden_dim, levels=get_fsq_level(codebook_size))`。
  - `get_fsq_level()` 仅支持特定 2 的幂尺寸（如 2048→`[8,8,6,5]`）。

- 复现实现：
  - `FSQ(levels=config['model']['fsq_levels'], dim=hidden_dim)`，`levels` 由配置直接给出（如 `[8,8,8,4]`→2048，或 `[8,8,4,4]`→1024）。
  - 为兼容不同 FSQ 版本，运行时探测 `return_indices` 形参；训练中缓存 `last_fsq_indices` 以便统计。

### 解码器（Decoder）
- 源实现：
  - `VAEDecoder`：
    - 以**全零**序列为查询，长度=`views*(T-1)`，跨注意到 codes。
    - 与 `track_pos_emb`（点位）与 `view_emb`（视角）拼接，经 `decoder_mlp` 后投影到 `out_dim=W×W`。
    - 输出 reshape 后为 `(b, v, t, n, W×W)`，可逐时刻得到相对位移分布。

- 复现实现：
  - `KeypointDecoder`（内置 `nn.TransformerDecoder`）以 **N 个可学习查询**解码，得到每个点的 logits `(B, N, W×W)`，随后复制到 `(B, T-1, N, W×W)`。
  - 查询不包含时间位置编码；时序差异主要已在编码阶段被压缩。

### 输出与恢复
- 源实现：
  - `decode()` 返回 `(x_recon, rel_logits)`，其中 `x_recon` 为**逐时刻**相对位移 `(b, v, t, n, 2)`（由 `rel_cls_logits_to_diffs` 从 logits 网格反推）。
  - 可用 `velocities_to_points()` 与初始点积分恢复轨迹。

- 复现实现：
  - `forward()` 返回 `logits: (B, T-1, N, W×W)`；不直接返回速度/点，需要外部按 `velocity_to_labels()` 的逆映射自行处理（目前主要用于分类监督，而非重建）。

---

## 训练与损失对比

### 任务定义与标签生成
- 源实现：
  - 目标来自相邻时刻**像素坐标**差，映射到 `rel_cls_img_size`（默认 `[15,15]`）网格索引。
  - 使用 `compute_relative_classification_loss()` 计算 CE（形状展平为 `(b, v*t*n)`）。

- 复现实现：
  - 直接将**归一化速度**基于 `decoder_window_size (W)` 量化到 `W×W` 类别（`utils/helpers.py::velocity_to_labels`）。
  - 更侧重“局部窗口内的量化”，避免与全局图像尺寸耦合（已在预处理阶段做按 W 的归一化，参考 `inference_short_clip.py::_normalize_velocities`）。

### 损失与正则
- 源实现：
  - 仅 `relative_ce`（交叉熵），按视角权重求平均（`cfg.loss.loss_weights`）。
  - 日志记录**码本困惑度**（不参与 loss）。

- 复现实现：
  - **Focal Loss**：处理类别不均衡，支持对“中心类（近零位移）”降低权重（`alpha_center`），缓解“静止偏置”。
  - **静态点掩码**：按速度模长筛除低动态点，对 CE 加权平均，仅在掩码为 True 位置统计。
  - **码本熵正则（可选）**：基于 FSQ 索引直方图的归一化熵，作为启发式正则加入总损失（对索引不可导）。

### 优化与调度
- 源实现：
  - `AdamW`，可选 `cosine` 调度（含 warmup），AMP=True，支持 `torch.compile`，Hydra/W&B 训练基建完善。

- 复现实现：
  - `AdamW`，**Accelerate** 驱动多卡/混精（默认 bf16），TensorBoard 日志。
  - 训练目录结构包含 `best.pth`/`last.pth`/分 epoch checkpoint，以及（可选）优化器状态独立保存，支持 resume。

---

## 数据与预处理对比

- 源实现：
  - 基于 `LiberoDataset` 读取 LIBERO 数据，处理图像、多视角、插值（`linear|spline`）、轨迹坐标 `(c,r)→(r,c)`、归一化到 `[-1,1]`，并按 `img_shape` 与 `rel_cls_img_size` 建立标签。

- 复现实现：
  - 提供视频级离线推理与窗口化抽帧（`inference_short_clip.py`），调用 CoTracker 离线模型逐窗跟踪，导出**速度** `(T-1,N,2)`。
  - 速度按 `W` 半径归一化到 `[-1,1]`，与 `velocity_to_labels()` 一致；训练集直接读取预处理得到的 `.pt` 速度张量（`dataset/velocity_dataset.py`）。
  - 支持**静态点过滤**掩码（`static_filter`），避免静态点主导训练信号。

---

## 指标与可视化

- 源实现：
  - 轨迹重建可视化（GT vs 预测），`get_traj_metrics()`（MSE/L1/跨轨距离/像素容忍度准确率/ΔAUC），码本困惑度。

- 复现实现：
  - 训练/验证 CE、Acc，掩码占比，码本唯一码计数/熵/（归一化）困惑度统计（以 `last_fsq_indices` 为基础）。
  - 默认不做轨迹重建可视化（可在推理脚本对 `latent_tokens/labels` 另行分析）。

---

## 兼容性与迁移建议

- 若要让复现实现更接近源实现：
  1. **时间维度建模**：
     - 将 token 设计从“点×时间”改为“时间（×视角）”粒度：编码器输入改为 `b t (v n d)` 或 `b (v t) (n d)`，解码器查询长度为 `views*(T-1)`；去掉“时间复制”的做法，使输出 logits **随时间步变化**。
  2. **点/视角嵌入**：
     - 在解码后加入 `track_pos_emb` 与 `view_emb`，再经 MLP 融合，如 `VAEDecoder`。
  3. **标签映射一致性**：
     - 将标签生成从“W 半径归一化速度”切换为“像素空间相对位移 + `rel_cls_img_size` 网格”，使损失与源码完全一致。
  4. **多模态条件（可选）**：
     - 引入 `cond_on_img` 与 `VisionEncoder`，在编码器内做跨注意，支持图像条件。

- 若保持复现实现当前风格但提升表现/可解释性：
  1. **时间敏感解码**：
     - 保留 `d` 作为 memory 长度，但将解码器目标从 N 个查询扩展为 `T-1`×N 或 `T-1`，并加入时间位置编码，避免时间复制导致的“同窗等价”。
  2. **可导的码本使用正则**：
     - 将离散熵启发式替换为可导近似（如基于软 one-hot 的 KL 或温度化的 Gumbel 近似）。
  3. **窗口跨步一致性**：
     - 在推理/训练中增加跨窗一致性约束或数据增强（随机起点、随机 stride），减少“复制标签”带来的时序不稳定性。

---

## 配置差异与对应关系

- 源实现：
  - `cfg/train_motion_tokenizer.yaml`
    - `hidden_dim=768, num_heads=8, num_layers=2`
    - `codebook_size=2048`（FSQ levels 由 `get_fsq_level()` 自动推断）
    - `rel_cls_img_size=[15,15]`，`img_shape=[128,128]`
    - `cond_on_img=false, per_view=false`

- 复现实现：
  - `configs/tokenizer_config*.yaml`
    - `hidden_dim=768, num_heads=8, num_layers=2, decoder_window_size=15, num_classes=225`
    - `encoder_sequence_len=d`（通常与 T 接近，如 16）
    - `fsq_levels` 显式给出：如 `[8,8,8,4]`→2048 或 `[8,8,4,4]`→1024
    - 数据侧包含 `static_filter` 配置，推理/预处理包含 `target_fps`、`resize_shorter`、`window_stride` 等视频参数

---

## 实践影响（风险与收益）

- **时序表达的差异**（源实现“逐时刻分布” vs 复现“同窗复制”）会影响：
  - 下游任务若依赖**逐步**的 token（如自回归 FD）则更契合源实现；复现实现更像“窗口级 code + 点级标签”的表征。
  - 复现的“时序压缩”能提升计算/存储效率，但可能损失时序细节表达能力。

- **损失与标签管线差异**：
  - FocalLoss + 静态点掩码对现实视频中“静止占比高”的数据更鲁棒，但与论文/源码的相对 CE 存在目标定义差异，影响“严格复现”。

- **FSQ 级别配置**：
  - 源实现要求 codebook 为 2 的幂且满足内置映射；复现实现灵活但需与 FSQ 版本的 `return_indices` 行为匹配，建议训练/推理统一环境。

---

## 参考代码位置（便于查阅）

- 源实现：
  - 模型：`amplify/models/motion_tokenizer.py`（`MotionTokenizer`, `VAEEncoder`, `VAEDecoder`）
  - 训练：`train_motion_tokenizer.py`
  - 配置：`cfg/train_motion_tokenizer.yaml`
  - 标签/转换：`amplify/utils/data_utils.py`（`rel_cls_logits_to_diffs`, `get_autoregressive_indices_efficient`）
  - 损失：`amplify/models/losses.py`（`compute_relative_classification_loss`）
  - 注意力掩码：`amplify/utils/model/attn_masks.py`

- 复现实现：
  - 模型：`amplify_motion_tokenizer/models/motion_tokenizer.py`（`MotionTokenizer`）
  - 组件：`amplify_motion_tokenizer/models/components.py`（`KeypointEncoder`, `KeypointDecoder`）
  - 训练：`amplify_motion_tokenizer/train.py`（`FocalLoss`、熵正则、Accelerate/TensorBoard）
  - 数据：`amplify_motion_tokenizer/dataset/velocity_dataset.py`（静态点掩码）
  - 工具/标签：`amplify_motion_tokenizer/utils/helpers.py`（`velocity_to_labels`）
  - 推理：`amplify_motion_tokenizer/inference_short_clip.py`（窗口跟踪、归一化、latent/codes 导出）

---

## 结语

两套实现都贯彻了“速度→离散 code”的核心思想。若目标是严格对齐官方实现用于与其下游模块联动（FD/ID/AMPLIFY），建议优先按“兼容性与迁移建议”逐步靠拢；若目标是**高效标注/索引**与**视频动作片段分析**（窗口级 latent + 点级标签），当前复现实现具备工程效率与可解释性，且支持码本使用度监控与静态点抑制等实用增强。
