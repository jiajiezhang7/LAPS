# Motion Tokenizer 训练与推理代码总览

本文梳理了工作空间内“Motion Tokenizer”的全部训练与推理相关代码、调用关系与配置要点，便于你快速理解与定位。

---

## 代码快速索引（按职责分组）

- 模型实现
  - `amplify/models/motion_tokenizer.py`
    - 核心类与方法：`MotionTokenizer`, `VAEEncoder`, `VAEDecoder`, `MotionTokenizer.encode()`, `MotionTokenizer.decode()`, `MotionTokenizer.forward()`
    - 量化器：`vector_quantize_pytorch.FSQ`（`FSQ`）
    - 加载器：`load_motion_tokenizer()`
  - `amplify/models/transformer.py`
    - 编解码骨架：`TransformerEncoder`, `TransformerDecoder`，以及 `SelfAttention`, `CrossAttention` 组件
  - `amplify/utils/model/attn_masks.py`
    - 注意力掩码：`full_mask`, `causal_mask`, `causal_cond_mask`, `diag_cond_mask`, `block_mask` 等
  - （损失函数）`amplify/models/losses.py`
    - `compute_relative_classification_loss()`（相对分类交叉熵）

- 训练脚本
  - `train_motion_tokenizer.py`
    - 训练主脚本（Hydra 配置），`train_epoch()`, `val_epoch()`, `main()`
  - 配置文件：`cfg/train_motion_tokenizer.yaml`

- 数据与工具
  - `amplify/utils/train.py`
    - 数据集/加载器：`get_datasets()`, `get_dataloaders()`
    - 训练通用：`get_checkpoint_dir()`, `latest_checkpoint_from_dir()`, `save_checkpoint()`, `load_checkpoint()`
  - `amplify/utils/data_utils.py`
    - 轨迹/速度互转：`points_to_velocities()`, `velocities_to_points()`
    - 分类 logits → 相对位移：`rel_cls_logits_to_diffs()`, `rel_indices_to_diffs()`
    - 其他：`top_k_top_p_filtering()`，插值、归一化等
  - `amplify/utils/metrics.py`
    - 轨迹评估：`get_traj_metrics()`
    - 码本使用：`get_normalized_codebook_perplexity()`

- 与 Motion Tokenizer 集成的上下游
  - 前向动力学（预测代码索引）
    - `amplify/models/forward_dynamics.py`（类：`ForwardDynamics`，方法：`get_cond_tokens()`, `update()`, `predict()`）
    - `train_forward_dynamics.py`（训练/验证环节如何使用 Motion Tokenizer 生成 GT / 解码预测）
  - 逆向动力学（将 codes 映射到动作）
    - `amplify/models/inverse_dynamics.py`（类：`InverseDynamics`，支持 `gaussian`/`diffusion`/`flow` 动作头）
    - `train_inverse_dynamics.py`（如何从 GT 或 FD 预测的 codes 构造条件输入）
  - 统一策略封装与评估
    - `amplify/amplify.py`（类：`AMPLIFY`，方法：`act()`, `predict_codes()`, `predict_traj()`）
    - `eval_libero.py`（评估与 rollout，使用 `AMPLIFY`）

---

## 模型结构与核心 API

- `MotionTokenizer`（`amplify/models/motion_tokenizer.py`）
  - 编码器：`VAEEncoder`
    - 输入形状（速度）：`(b, v, t, n, d)`，其中 `v` 为视角数，`t = track_pred_horizon - 1`（速度序列），`n` 为 track 数，`d` 为点维度（默认 2）。
    - 可选图像条件：`cond_on_img=True` 时通过 `VisionEncoder` 生成序列条件，使用 `TransformerDecoder` 进行“条件编码”（Q 来自轨迹，KV 来自图像 token）；否则用 `TransformerEncoder`。
    - 注意力掩码：由 `cfg.causal_encoder` 控制，取值支持：
      - `True`：因果掩码 `causal_mask()`
      - `'diag'`：对角掩码 `diag_cond_mask()`（每步仅看自身时间步，cond token 全可见）
      - `False`：全可见
  - 量化器：`FSQ(dim=cfg.hidden_dim, levels=get_fsq_level(cfg.codebook_size))`，将连续隐向量 `z` 量化为离散代码（索引）
  - 解码器：`VAEDecoder`
    - 将 codes 解码为相对分类 logits，再通过 `rel_cls_logits_to_diffs()` 转为相对位移，最后可积累为点轨迹
    - 解码过程会拼接位置与视角嵌入，使用 `TransformerDecoder`
  - 关键方法
    - `encode(x, cond=None) -> z`
    - `decode(codes) -> (x_recon_velocities, rel_logits)`
      - 输出 `rel_logits` 维度逻辑：先 `(b, v*t*n, d)`，再 reshape 为 `(b, v, t, n, d)` 进行 softmax / argmax（见 `rel_cls_logits_to_diffs()`）
    - `forward(x, cond=None) -> (x_recon_velocities, codebook_indices, rel_logits)`
    - `get_loss(x_recon, rel_logits, gt_vel, gt_traj)`
      - 默认使用 `relative_ce`，内部调用 `compute_relative_classification_loss()`，对每视角加权求和
  - 配置关键项
    - 输入/输出维度推导：`get_vae_in_out_dim(cfg)`（是否 `per_view` 会影响输入维度）
    - `track_pred_horizon - 1` 用作速度序列长度
    - `loss.rel_cls_img_size` 定义相对分类网格（默认 `[15, 15]`）

- 注意
  - 文件内有 `load_vae_encoder()` / `load_vae_decoder()` 使用 `VAE(...)` 的残留代码段，但未定义 `VAE` 类；实际工程中请使用 `load_motion_tokenizer()` 加载完整模型（已在其它训练脚本中采用）
  - `MotionTokenizer.decode()` 返回的是速度，需要配合 `velocities_to_points()` 重建点轨迹

---

## 训练脚本：Motion Tokenizer（VAE）训练

- 入口：`train_motion_tokenizer.py`
  - 数据与加载器
    - `get_datasets()`/`get_dataloaders()`（`amplify/utils/train.py`）
    - 数据包含 `tracks`（轨迹点）、`images`（如需条件）等键，形状/时长受 `cfg` 控制
  - 训练步骤（`train_epoch()`）
    - 将点轨迹 `x` 转为速度 `x_gt = points_to_velocities(x, time_dim=2)`
    - 前向：
      - `cond_on_img=True`：`x_recon, codebook_indices, rel_logits = model(x_gt, img)`
      - 否则：`model(x_gt)`
    - 损失：`model.get_loss(x_recon, rel_logits, gt_vel=x_gt, gt_traj=x)`
      - 内部相对分类交叉熵（下节详述）
    - 日志：
      - 将 `rel_logits -> diffs -> points` 复原可视化，记录 `get_traj_metrics()` 与 `codebook_perplexity`
  - 验证步骤（`val_epoch()`）同理
  - 训练细节
    - AMP：`torch.autocast(...)` 与 `GradScaler`（可通过 `cfg.amp` 开关）
    - 学习率调度：`cosine` + 预热（可选）
    - 梯度累积：`cfg.batch_size / cfg.gpu_max_bs`
    - 模型编译：`torch.compile(model)`（可选）
    - 检查点：`save_checkpoint()`/`load_checkpoint()` 与断点续训

- 示例命令（来自 `README.md`）
  - `python train_motion_tokenizer.py run_name=<run name> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'`

---

## 损失、数据转换与指标

- 相对分类损失（`amplify/models/losses.py`）
  - `compute_relative_classification_loss(logits, targets, batch_traj, cfg)`
    - 先用 `get_autoregressive_indices_efficient()` 将 `batch_traj` 与 `targets`（t 与 t-1）映射到相对网格索引
    - 对 `logits.reshape(-1, d)` 与平坦化的索引做 `F.cross_entropy`（忽略 `-1`）
    - 返回 `(b, v, t, n)` 的未归约 loss，最终在 `MotionTokenizer.get_loss()` 中按视角加权聚合
- 速度/点互转（`amplify/utils/data_utils.py`）
  - `points_to_velocities(points, time_dim)`
  - `velocities_to_points(velocities, time_dim, init_points)`
- 分类 logits → 位移 → 轨迹（`amplify/utils/data_utils.py`）
  - `rel_cls_logits_to_diffs(logits, ...)`：
    - `softmax` 后 `argmax` 获得相对索引，转像素位移，再归一化到全局图像坐标系下的相对增量
- 指标（`amplify/utils/metrics.py`）
  - `get_traj_metrics()`：MSE/L1、跨轨距离、归一化像素容忍度准确率、ΔAUC 等
  - `get_normalized_codebook_perplexity()`：码本使用频率熵的归一化

---

## 推理与上下游集成

### Standalone（仅用 Motion Tokenizer）
- 编码：`MotionTokenizer.encode(x, cond_img)`
- 量化：`codes, indices = quantize(z)`（内部 FSQ）
- 解码：`vel, rel_logits = MotionTokenizer.decode(codes)`
- 轨迹点重建：`velocities_to_points(vel, time_dim=2, init_points=traj[:, :, [0]])`

### 前向动力学（预测代码索引）
- 类与训练：`amplify/models/forward_dynamics.py`, `train_forward_dynamics.py`
- 条件序列（图像+文本）构造：`ForwardDynamics.get_cond_tokens(obs, goal)`
  - `img_tokens -> img_proj`，`text_emb -> text_proj`，拼接为长度 `cond_seq_len` 的 token 序列
  - 注意力掩码：`causal_cond_mask(cond_seq_len + pred_seq_len, cond_seq_len)`，保证预测步骤仅看过去与条件
- 训练（teacher forcing，`update()`）
  - 用 GT 速度经过 `MotionTokenizer.encode()`+`quantize()` 得到 `gt_indices`
  - 以 `[cond, sos, gt_codes[:-1]]` 作为输入，输出 logits（`unembed`），计算 CE 与预测索引
- 自回归推理（`predict()`）
  - 以 `[cond, sos]` 开始，循环预测当前 index，转代码向量拼回序列
  - 采样策略：`argmax` 或 `topk`（见 `top_k_top_p_filtering()`）
- 与 Motion Tokenizer 解码联动
  - 训练/验证中常将预测索引映射为 `codes`，再 `MotionTokenizer.decode(codes)` 得到速度，最后还原点轨迹用于指标与可视化（见 `train_forward_dynamics.py`）

### 逆向动力学（将 codes → 动作）
- 类与训练：`amplify/models/inverse_dynamics.py`, `train_inverse_dynamics.py`
- 条件输入可选：
  - `img_tokens`（多视角拼接后线性投影）
  - `text_tokens`（原始 T5 或预处理嵌入）
  - `proprioception`（本体感觉）
  - `codes`（来自 Motion Tokenizer 的 GT 编码，或来自 ForwardDynamics 预测编码）
- 动作头（可插拔）：
  - `GaussianActionHead`（默认，支持 `std` 与可选 `action_squash`）
  - `DiffusionActionHead`（可选，需额外依赖）
  - `FlowActionHead`（可选，需额外依赖）
- 训练时若启用 `cond_on_tracks`：
  - 若提供 `forward_dynamics_checkpoint`，则通过 FD 预测 `indices -> codes`
  - 否则使用 GT 轨迹经 Motion Tokenizer `encode + quantize` 得到 `codes`
- 可视化：`vis_recon=True` 时，支持将 `codes` 经 `decode -> velocities_to_points()` 可视化轨迹

### 统一策略 `AMPLIFY` 与评估
- `amplify/amplify.py`（类：`AMPLIFY`）
  - `act(images, proprio, text/text_emb, ar_sampling)`：
    - 图像与文本编码成条件 token → 前向动力学预测 `indices` → 转 `codes` → 逆向动力学输出动作
  - `predict_codes(images, ...) -> (indices, codes)`
  - `predict_traj(images, init_queries, ...) -> pred_traj`
    - `codes -> motion_tokenizer.decode(codes) -> velocities_to_points()`
- `eval_libero.py`
  - 加载 `AMPLIFY` checkpoint
  - rollout 中调用 `policy.act(...)` 执行动作；可选用 `policy.predict_traj(...)` 可视化预测轨迹
  - 评估时常将 `policy.motion_tokenizer.encoder/decoder` 迁移到 CPU 以节省 GPU 显存（见 `eval_libero.py` 中的 `.to('cpu')`）

---

## 配置与超参数（`cfg/train_motion_tokenizer.yaml` 要点）

- 训练
  - `batch_size`, `gpu_max_bs`, `num_epochs`, `lr`, `lr_schedule`, `clip_grad`, `weight_decay`, `amp`, `compile`
  - `train_datasets`, `val_datasets`（形如 `'[libero_10:traj0.9]'`）
  - `cond_cameraviews`（默认 `agentview`, `eye_in_hand`）
  - `img_shape`, `true_horizon`, `track_pred_horizon`, `num_tracks`, `interp_method`
- 模型
  - `type: transformer`，`hidden_dim: 768`，`num_heads: 8`，`num_layers: 2`，`codebook_size: 2048`
  - `causal_encoder: true`（编码器注意力掩码）
  - `decoder_mlp_hidden_dim: 256`
  - `cond_on_img: false`（默认关闭图像条件）
  - `per_view: false`（默认跨视角拼接）
- 损失
  - `loss.loss_fn: relative_ce`
  - `loss.rel_cls_img_size: [15, 15]`
  - 视角权重：`loss.loss_weights.agentview: 0.9`, `eye_in_hand: 0.1`

---

## 端到端调用关系图

```mermaid
flowchart TD
  subgraph Data
    A[tracks (points)] -->|points_to_velocities| B[velocities]
    I[images] --> C[VisionEncoder (optional)]
  end

  subgraph MotionTokenizer
    B --> E[encode()]
    C -->|cond_on_img| E
    E --> F[FSQ quantize -> codes/indices]
    F --> G[decode(codes) -> rel_logits]
    G --> H[rel_cls_logits_to_diffs -> velocities]
  end

  subgraph ForwardDynamics
    I --> J[img_tokens]
    T[text/text_emb] --> K[text_tokens]
    J --> L[get_cond_tokens]
    K --> L
    L --> M[update() or predict()]
    M --> N[predicted indices -> codes]
  end

  subgraph InverseDynamics
    N --> O[cond_on_tracks]
    I --> O
    T --> O
    P[proprio] --> O
    O --> Q[action head -> actions]
  end

  H --> R[velocities_to_points -> traj (for vis/metrics)]
  N --> G
```

---

## 使用与命令参考

- 训练 Motion Tokenizer（VAE）
  - `python train_motion_tokenizer.py run_name=<run name> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'`
- 训练 Forward Dynamics（依赖已训练的 Motion Tokenizer）
  - `python train_forward_dynamics.py run_name=<run name> forward_dynamics.motion_tokenizer.checkpoint=<path/to/mt.pt> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'`
- 训练 Inverse Dynamics（可选依赖 FD 预测 codes）
  - `python train_inverse_dynamics.py run_name=<run name> motion_tokenizer_checkpoint=<path/to/mt.pt> forward_dynamics_checkpoint=<path/to/fd.pt> train_datasets='[libero_10:action0.9]' val_datasets='[libero_10:action-0.1]'`
- 打包统一策略（AMPLIFY）
  - `python -m amplify.bundle_amplify --mt_ckpt <mt.pt> --fd_ckpt <fd.pt> --id_ckpt <id.pt> --name my_amplify_checkpoint`
- 评估
  - `python eval_libero.py dataset='[libero_10]' run_name=<run name> amplify_checkpoint=<checkpoints/AMPLIFY/my_amplify_checkpoint.pt>`

---

## 实践注意事项与小结

- 注意时间维度：速度序列长度为 `track_pred_horizon - 1`；点轨迹重建需提供初始点 `init_points`。
- `cond_on_img` 打开后，编码器改用跨模态 `TransformerDecoder`；显存与计算开销将上升。
- `per_view=false` 时将多视角拼接成统一序列；`per_view=true` 则每视角独立 token（影响 `in_dim/out_dim` 与时序展开方式）。
- 前向动力学掩码 `causal_cond_mask` 确保预测步骤的自回归性与条件可见性。
- `motion_tokenizer.py` 内的 `load_vae_encoder/decoder()` 使用了未定义的 `VAE` 符号，推测为早期残留；实际工程加载请使用 `load_motion_tokenizer()`。
- 码本困惑度（perplexity）与轨迹指标可用于监控训练健康状况；需要时可在 `train_motion_tokenizer.py`/`train_forward_dynamics.py` 中调整日志频率与可视化开关。

如需我继续完善这份文档（例如加入更细粒度的模块交互图/类图，或补充运行示例与常见问题），请告诉我你的需求。

---

## 复现 Motion Tokenizer：端到端步骤

本节提供从环境、数据到训练、评估的完整可复现实操指引。确保在仓库根目录 `amplify_origin/` 下执行命令。

### 1) 环境与依赖
- 建议 Python 3.10（项目 `README.md` 使用 3.10）。
- 基础依赖安装：
  - 克隆 AMPLIFY 并可编辑安装：
    - `pip install -e .`
  - 第三方项目（参考 `README.md`）：
    - LIBERO（数据与评测基准）
      - 克隆并安装依赖，可执行 `python benchmark_scripts/download_libero_datasets.py` 下载数据
      - 将 `LIBERO/libero/datasets` 软链到 `preprocessed_data/libero_demos`
    - CoTracker（离线点跟踪器，若需自行预处理轨迹）
      - `git clone https://github.com/facebookresearch/co-tracker.git && pip install -e co-tracker`

提示：如遇 `egl_probe` 或 Transformers 版本冲突，参考主仓库 `README.md` 的 Notes 小节（含一键修复命令）。

### 2) 数据准备（两种路径）
- 直接下载预处理好的点轨迹（推荐快速上手）
  - 从主 `README.md` 提供的 Google Drive 链接下载对应 LIBERO 套件的轨迹包，放到 `preprocessed_data/<libero_10|libero_90|...>`
- 自行预处理
  - 运行 `python -m preprocessing.preprocess_libero mode=tracks suite='<libero_10|libero_90|libero_object|libero_spatial|libero_goal>'`
  - 可选：预处理文本嵌入（用于 Forward/Inverse Dynamics）
    - `python -m preprocessing.preprocess_libero mode=text suite='<...>'`
    - 产物位于 `preprocessed_data/<dataset>/text/` 下（每任务一个 `*.hdf5`）

数据加载器关键点（`amplify/loaders/libero_dataset.py`）：
- 将 CoTracker 的 `(col,row)` 顺序转为 `(row,col)`，并归一化到 `[-1,1]`。
- 如 `track_pred_horizon != true_horizon`，按 `interp_method`（`linear|spline`）插值到预测长度。
- 视图顺序由 `cfg.cond_cameraviews` 控制，默认 `['agentview','eye_in_hand']`。

### 3) 配置关键项（`cfg/train_motion_tokenizer.yaml`）
- 数据/时序
  - `img_shape: [128,128]`、`true_horizon: 16`、`track_pred_horizon: 16`（速度长度为 `-1`）。
  - `num_tracks: 400`（每帧跟踪点数量）。
  - `cond_cameraviews: ['agentview','eye_in_hand']`。
  - `interp_method: linear|spline` 控制预处理插值方式。
- 模型结构
  - `hidden_dim: 768`、`num_heads: 8`、`num_layers: 2`。
  - `type: transformer`、`attn_pdrop: 0.1`、`causal_encoder: true|false|'diag'`。
  - `per_view: false`（false=多视角在通道维拼接；true=每视角独立时间序列，序列长度×视角数，显存更大）。
  - `decoder_mlp_hidden_dim: 256`。
  - 解码器层数为编码器层数的一半：`decoder.num_layers = int(cfg.num_layers/2)`。
- 条件图像（可选）
  - `cond_on_img: false`（默认关闭）。开启后：
    - 编码器切换为 `TransformerDecoder`，Q 来自轨迹序列、KV 来自图像 token（由 `VisionEncoder` 提供）。
    - 视觉 token 序列长度 = `VisionEncoder.seq_len * num_views`，显存/计算显著上升。
- 损失与网格
  - `loss.loss_fn: relative_ce`、`loss.rel_cls_img_size: [15,15]`。
  - 分类网格 `[15,15]` 与 `img_shape` 共同决定从分类 index 到全局像素位移的缩放（见 `rel_cls_logits_to_diffs()`）。
- 量化器（FSQ）与码本尺寸
  - `codebook_size` 必须为下表中的 2 的幂之一；内部由 `get_fsq_level()` 映射到 FSQ `levels`：
    - 16(2^4)→[5,3]
    - 64(2^6)→[8,8]
    - 256(2^8)→[8,6,5]
    - 512(2^9)→[8,8,8]
    - 1024(2^10)→[8,5,5,5]
    - 2048(2^11)→[8,8,6,5]
    - 4096(2^12)→[7,5,5,5,5]
  - 选择不在表内的尺寸会导致量化失败（levels 为空）。
- 训练杂项
  - AMP：`amp: true`；学习率调度：`lr_schedule: null|cosine`（cosine 带预热）。
  - 批次/累积：`batch_size: 256`、`gpu_max_bs: 64`（按需梯度累积）。
  - Checkpoint：`save_interval: 25`、`resume: true`、`run_name: <name>`。

### 4) 训练命令（最小可复现）
- 仅训练 Motion Tokenizer（VAE）：
  - `python train_motion_tokenizer.py run_name=<run name> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'`
- 常见覆盖项（按需）：
  - `img_shape=[128,128] num_tracks=400 track_pred_horizon=16`
  - `cond_on_img=false per_view=false codebook_size=2048`
- 断点续训与目录结构：
  - Checkpoint 写入 `checkpoints/motion_tokenizer/<run_name>[_k]/`，包含 `latest.pt` 与若干按 epoch 命名的快照。
  - `resume=true` 时若未显式提供 checkpoint，会自动使用该目录最新的 `latest.pt`。

### 5) 训练过程监控与可视化
- 日志项（`train_motion_tokenizer.py`）：
  - 轨迹重建：`rel_logits -> diffs -> velocities -> points`，与 GT 叠图（`vis_pred`）。
  - 指标：`get_traj_metrics()`（MSE/L1/跨轨距离/像素容忍度准确率/ΔAUC）。
  - 码本使用：`get_normalized_codebook_perplexity()`（归一化困惑度）。
- 频率：由 `log_interval` 控制；W&B 可通过 `use_wandb=true` 启用。

### 6) 推理/解码最小用法（要点）
- `MotionTokenizer.encode(x, cond_img)` 接受速度序列 `(b,v,t,n,d)`；若开启 `cond_on_img` 则传入原图 `(b,v,h,w,c)`。
- `quantize(z)` 产出 `(codes, indices)`；
- `decode(codes)` 返回：
  - `x_recon`: 相对位移（速度），形状 `(b,v,t,n,2)`，坐标在 `[-1,1]`；
  - `rel_logits`: 分类 logits，内部按 `(b, v*t*n, d)` 排布。
- 速度→点轨迹：`velocities_to_points(vel, time_dim=2, init_points=traj[:, :, [0]])`。
- 解码器实现细节：
  - 解码查询 `x_recon` 以全零初始化，经自注意力与跨注意力到 `codes`，再与 `track_pos_emb`/`view_emb` 结合通过 MLP 投影至分类 logits。

### 7) 与 Forward/Inverse Dynamics 联动
- Forward Dynamics（预测 indices → codes → 速度/轨迹）：
  - 训练入口：`train_forward_dynamics.py`；需提供已训练 MT checkpoint：
    - `python train_forward_dynamics.py run_name=<name> forward_dynamics.motion_tokenizer.checkpoint=<path/to/mt.pt> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'`
  - 文本条件：`forward_dynamics.text_encoder.use_preprocessed_embs` 控制是否使用预处理文本嵌入；若为 `false` 则在线调用 T5。
  - 训练中从 GT 轨迹经 MT `encode+quantize` 得到 `gt_indices` 进行 teacher forcing；推理自回归生成。
- Inverse Dynamics（codes → actions）：
  - 训练入口：`train_inverse_dynamics.py`；可选依赖 FD 预测的 codes，或直接使用 GT 经 MT 量化得到的 codes。
  - 动作头：`gaussian`（默认）、`diffusion`、`flow` 可选。

### 8) 打包统一策略与评估（LIBERO）
- 打包统一策略：
  - `python -m amplify.bundle_amplify --mt_ckpt <mt.pt> --fd_ckpt <fd.pt> --id_ckpt <id.pt> --name my_amplify_checkpoint`
- 评估（LIBERO）：
  - `python eval_libero.py dataset='[libero_10]' run_name=<run name> amplify_checkpoint=<checkpoints/AMPLIFY/my_amplify_checkpoint.pt>`
  - 评估时会将 `policy.motion_tokenizer.encoder/decoder` 迁移到 CPU 以节省显存（源码中已实现）。

### 9) 常见问题与排查
- 量化失败/levels 为空：
  - 请确认 `codebook_size` 属于 FSQ 支持表（如 2048）。
- 显存不足：
  - 关闭 `cond_on_img` 或降低 `num_tracks`、`img_shape`、`batch_size`；避免 `per_view=true` 导致的序列增长。
- 轨迹坐标错位/越界：
  - 留意 `(c,r)->(r,c)` 转换与 `[-1,1]` 归一化；`LiberoDataset` 已处理，但自定义数据需保持一致。
- 历史残留 API 不可用：
  - `load_vae_encoder()`/`load_vae_decoder()` 与 `__main__` 测试段中的 `VAE/vae_checkpoint` 为早期残留，请勿使用；统一用 `load_motion_tokenizer()`。
- Hydra 参数格式：
  - 列表参数需整体用引号包裹，如 `train_datasets='[libero_10:traj0.9]'`，避免 YAML 解析错误。

### 10) 硬件与版本建议
- GPU：开启图像条件或更大 `num_tracks`/`img_shape` 时建议 ≥24GB 显存；否则 12GB 也可在默认配置下训练。
- 精度：默认 AMP=true；可按需关闭（速度/显存将变化）。
- OS：Linux/macOS 皆可；macOS 跑大模型需设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`（详见主 `README.md`）。
- 版本：建议遵循本仓库 `requirements.txt` 与主 `README.md` 步骤（包含 LIBERO/CoTracker 安装细节）。

---

如需我继续扩展本指南（例如加入“推荐超参数组合 vs 显存预算”的详表，或添加自定义数据模板与最小可运行脚本），请告诉我你的具体需求。

