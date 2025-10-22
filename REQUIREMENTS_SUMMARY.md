# Requirements.txt 完善总结

## 更新历史

基于环境测试结果，已完善 `requirements.txt`，添加了所有必需的依赖。

## 新增依赖说明

### 核心依赖（已添加）

1. **omegaconf >= 2.3.0**
   - 用途：配置文件管理（amplify 项目核心）
   - 使用位置：amplify 所有训练脚本

2. **timm >= 1.0.0**
   - 用途：Vision Transformer 模型库
   - 使用位置：amplify.models.encoders.vision_encoders

3. **hydra-core >= 1.2.0**
   - 用途：配置管理框架
   - 使用位置：amplify 所有训练脚本（train_*.py）

4. **positional-encodings >= 6.0.0**
   - 用途：位置编码（Transformer 必需）
   - 使用位置：amplify.models.transformer, motion_tokenizer

5. **transformers >= 4.21.0**
   - 用途：T5 文本编码器等
   - 使用位置：amplify.models.encoders.t5

6. **ipython >= 8.0.0**
   - 用途：增强的错误追踪（IPython.core.ultratb）
   - 使用位置：amplify 训练脚本的错误处理

7. **imageio >= 2.0.0**
   - 用途：图像和视频 I/O
   - 使用位置：数据加载和可视化

8. **h5py >= 3.0.0**
   - 用途：HDF5 数据格式读写
   - 使用位置：amplify 数据集（LIBERO 等）

### 可选依赖（已注释）

1. **wandb >= 0.13.0**
   - 用途：实验跟踪和可视化
   - 使用位置：amplify 训练脚本
   - 安装命令：`pip install wandb`
   - 说明：如不使用实验跟踪，可不安装

2. **CoTracker**
   - 用途：视频点追踪（数据预处理）
   - 安装命令：`pip install git+https://github.com/facebookresearch/co-tracker.git`
   - 说明：仅在需要从原始视频提取轨迹时使用

## 完整依赖列表

### 当前 requirements.txt 包含：

```
# Core
torch>=2.1.0
torchvision>=0.16.0
PyYAML>=6.0.1
numpy>=1.24.0
tqdm>=4.66.0
vector-quantize-pytorch>=1.14.0
opencv-python>=4.8.0
einops>=0.7.0
accelerate>=0.30.0
tensorboard>=2.12.0
scikit-learn>=1.3.0
scipy>=1.10.0
umap-learn>=0.5.5
matplotlib>=3.7.0
seaborn>=0.12.2
hdbscan>=0.8.33

# Additional dependencies for amplify
omegaconf>=2.3.0
timm>=1.0.0
hydra-core>=1.2.0
positional-encodings>=6.0.0
transformers>=4.21.0
ipython>=8.0.0
imageio>=2.0.0
h5py>=3.0.0
```

## 测试结果

运行 `python test_env.py` 后：

- ✅ **35/35 测试通过**
- ⚠️ **1 个警告**：wandb 未安装（可选依赖）

### 支持的功能

#### 完全支持 ✅

1. **amplify_motion_tokenizer**
   - 训练和推理
   - FSQ 量化
   - 数据预处理

2. **action_classification**
   - LSTM 序列编码
   - HDBSCAN/UMAP 聚类
   - 评估和可视化

3. **video_action_segmenter**
   - 视频流处理
   - 能量分割
   - 在线推理

4. **amplify（核心功能）**
   - Motion Tokenizer
   - 数据加载
   - 基础训练

#### 部分支持 ⚠️

- **amplify 实验跟踪**：需要安装 wandb
- **视频轨迹提取**：需要安装 CoTracker

## 安装指南

### 基础安装（必需）

```bash
conda activate laps
pip install -r requirements.txt
```

### 可选功能安装

```bash
# 安装 wandb（实验跟踪）
pip install wandb

# 安装 CoTracker（视频预处理）
pip install git+https://github.com/facebookresearch/co-tracker.git
```

## 验证环境

```bash
conda activate laps
python test_env.py
```

预期输出：
- 通过：35/35
- 失败：0
- 警告：0-1（取决于是否安装 wandb）

## 与 amplify/requirements.txt 的差异

### 已包含的 amplify 依赖

- ✅ torch, torchvision, numpy, scipy
- ✅ einops, opencv-python, matplotlib
- ✅ omegaconf, hydra-core, timm
- ✅ transformers, positional-encodings
- ✅ imageio, h5py, ipython
- ✅ vector-quantize-pytorch, tqdm

### 未包含的 amplify 依赖（特定用途）

- ❌ **robomimic, robosuite, mujoco**：机器人仿真（LIBERO 评测专用）
- ❌ **gym==0.25.2**：强化学习环境（LIBERO 专用）
- ❌ **wandb**：实验跟踪（可选）
- ❌ **moviepy**：视频编辑（可选）
- ❌ **pyinstrument**：性能分析（开发用）

说明：这些依赖仅在运行 LIBERO 评测或特定开发任务时需要，不影响核心功能。

## 推荐的完整安装流程

```bash
# 1. 创建环境
conda create -n laps python=3.10 -y

# 2. 激活环境
conda activate laps

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 验证环境
python test_env.py

# 5. （可选）安装实验跟踪
pip install wandb

# 6. （可选）安装 CoTracker
pip install git+https://github.com/facebookresearch/co-tracker.git
```

## 常见问题

### Q: 为什么不包含 wandb？
A: wandb 是实验跟踪工具，不影响模型训练的核心功能。如需使用，可单独安装。

### Q: 为什么不包含 robomimic/robosuite？
A: 这些是机器人仿真库，仅在运行 LIBERO 评测时需要。核心的 motion tokenizer 训练和推理不需要。

### Q: CoTracker 必须安装吗？
A: 不必须。只有在需要从原始视频提取点轨迹时才需要。如果使用已预处理的数据，可以不安装。

### Q: 如何检查缺少哪些依赖？
A: 运行 `python test_env.py`，测试脚本会自动检测并报告缺失的依赖。

## 更新日志

- **2025-10-21**: 初始版本，包含所有核心依赖
- **2025-10-21**: 添加 amplify 必需依赖（omegaconf, timm, hydra-core 等）
- **2025-10-21**: 标记可选依赖（wandb, CoTracker）
