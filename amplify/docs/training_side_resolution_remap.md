# 训练侧分辨率重映射方案（修复预处理缩放与训练尺度不一致）

## 背景与问题
- 预处理脚本 `amplify/preprocessing/preprocess_my_segments.py` 会在 `_resize_frame()` 中按短边缩放视频（`resize_shorter`，保持纵横比），然后进行点跟踪并将轨迹写入 HDF5。
- 训练数据加载器 `amplify/amplify/loaders/custom_segments_dataset.py` 在 `process_data()` 中调用 `normalize_traj(tracks, img_shape)` 将轨迹归一化到 [-1, 1]，其中 `img_shape` 来自训练配置（例如 `cfg.img_shape: [477, 470]`）。
- 若预处理后的实际分辨率与训练侧 `img_shape` 不一致，则归一化分母被“错误放大/缩小”，导致轨迹坐标范围被压缩（如约 [-0.46, 0.46]），削弱位移幅度，进而放大损失中“中心类（零位移）”的偏置，码本易坍塌。

## 目标
- 在不改变损失与模型接口的前提下，使所有样本在进入训练/推理前都以统一的 `cfg.img_shape` 作为“世界坐标参考尺寸”。

## 方案概述（最小侵入）
1. 预处理时，把用于追踪的“实际图像分辨率 (H, W)”写入每个视角的 HDF5 组属性。
2. 训练加载时，先根据该属性把轨迹从“预处理实际尺寸”线性重映射到统一的 `cfg.img_shape` 上，再调用 `normalize_traj()` 归一化到 [-1, 1]。

这样，损失函数与解码器中使用的 `cfg.img_shape` 与数据加载阶段的几何语义完全一致，消除尺度错配引发的幅度压缩问题。

## 代码改动点
- 预处理侧：写入尺寸属性
  - 文件：`amplify/preprocessing/preprocess_base.py`
  - 位置：`TrackProcessor.process()`
  - 变更：写入 `root/<view>`.attrs[`height`,`width`]，值取自 `video_thwc.shape[1:3]`（预处理时用于跟踪的实际分辨率）。

- 训练侧：读取并重映射到 `cfg.img_shape`
  - 文件：`amplify/amplify/loaders/custom_segments_dataset.py`
  - 读取阶段（`load_tracks()`）：
    - 从 `root/<view>`.attrs 读取 `height`、`width`，若存在，汇总为 `out['img_size']=(H, W)`。
  - 处理阶段（`process_data()`）：
    - 在 `normalize_traj()` 之前，若发现 `img_size != cfg.img_shape`，则对像素轨迹做逐通道线性缩放（行×`tgt_h/orig_h`，列×`tgt_w/orig_w`），完成后再调用 `normalize_traj(tracks, cfg.img_shape)`。

## 数据格式与兼容性
- 新增的 HDF5 属性（每视角）：
  - `root/<view>`.attrs[`height`]：int，预处理时用于跟踪的图像高度。
  - `root/<view>`.attrs[`width`]：int，预处理时用于跟踪的图像宽度。
- 兼容性：
  - 若旧数据缺少上述属性，加载器会回退到“直接以 `cfg.img_shape` 归一化”的旧逻辑（不做重映射）。为获得稳定效果，建议重新预处理生成带属性的数据。

## 使用步骤
1. 预处理：
   - 配置文件：`amplify/cfg/preprocessing/preprocess_my_segments.yaml`（默认 `resize_shorter: 480`，保持纵横比）。
   - 运行后，输出 HDF5 会自动带上每视角的 `height/width` 属性。
2. 训练：
   - 在 `amplify/cfg/train_motion_tokenizer.yaml` 中设置统一的 `img_shape`（例如 `[477, 470]`）。
   - 加载器会自动将所有样本轨迹映射到该尺寸后再归一化，无需额外改动。

## 验证与监控建议
- 关注训练日志中的码本使用指标（已添加）：
  - `train_unique_codes_step`、`train_entropy_norm_step`、`train_perplexity_step`、`train_code_usage_hist`、`train_codebook_perplexity`。
- 预期现象：
  - 坍塌缓解：唯一码数↑、熵/困惑度↑、直方图更均匀。
  - 与“非零速度”相关的统计应恢复到合理水平。

## 选择该方案的理由
- 最小侵入：
  - 不修改损失/模型接口；仅在预处理写属性、训练加载阶段插入一处线性缩放。
- 保持跟踪质量：
  - 不强制将视频拉伸到固定长宽比，避免 CoTracker 因拉伸/变形而退化。
- 统一全流程语义：
  - 训练与解码中的 `cfg.img_shape` 成为一致的“世界坐标参考尺寸”。

## 迁移到“固定目标尺寸”的可选路线（长期）
- 若希望数据集标准化、便于跨项目复用，可在预处理阶段直接统一到固定 `img_shape`，但建议“保持纵横比 + letterbox 填充”并写入 `scale`/`pad` 元数据；同时训练/推理全流程以相同几何变换为准。该路线改动相对更大，当前方案已足以解决尺度错配与坍塌问题。
