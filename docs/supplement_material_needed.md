# 论文补充材料 (Supplementary Materials) 实验计划

## 1. 核心方法论详情 (Methodology Details)

### 1.1. 实验 1：Motion Tokenizer ($M_{\theta}$) 架构与训练

* **目标：** 详细阐明 Motion Tokenizer 的架构 与训练细节，支撑正文 3.1 节，兑现“详见补充材料”的承诺，增强可复现性。
* **产出图表：**
    * **`Figure S1: Motion Tokenizer 详细架构图`**
        * **内容：** 绘制一幅包含 $E_{\theta}$ (Encoder), $D_{\theta}$ (Decoder), 以及 FSQ (Finite Scalar Quantization) 层的详细流程图。
        * **标注：** 明确标出输入（Keypoint Velocities $\in \mathbb{R}^{T \times N \times 2}$）和输出（Discrete Tokens $z_t$ 和 Relative Displacement Predictions）。
    * **`Figure S2: Motion Tokenizer 训练损失曲线`**
        * **内容：** 绘制训练过程中的损失函数（Cross-Entropy Loss）随 Epochs/Steps 下降的曲线。
        * **目的：** 证明 $M_{\theta}$ 已稳定收敛。
    * **`Table S1: Motion Tokenizer 训练超参数`**
        * **内容：** 列出所有用于复现 $M_{\theta}$ 的关键超参数。
        * **行条目：**
            * Keypoint Tracker (e.g., CoTracker)
            * Encoder ($E_{\theta}$) 架构 (e.g., Transformer 层数, 头数)
            * Decoder ($D_{\theta}$) 架构
            * FSQ 码本参数 (e.g., levels, dimensions)
            * 训练数据集 ($\mathcal{D}_{clips}$) 来源
            * Batch Size
            * Learning Rate
            * Optimizer
            * 训练总 Epochs

### 1.2. 实验 2：Action Segmentor 阈值无监督优化

* **目标：** 详细展示 $\theta_{on}$ 是如何通过一个**自动化、无监督**的过程 标定出来的，用数据证明其鲁棒性，支撑正文 3.2.2 节。
* **产出图表：**
    * **`Figure S3: 代理信号 (Proxy Signal) 可视化`**
        * **内容：** 在一个典型的视频片段上，同时绘制：
            1.  低级“速度能量”（Proxy Signal，例如关键点速度的时间差分范数）。
            2.  Otsu法 或分位数法生成的二元伪标签 $y_{pseudo}$。
            3.  我们的高级 $E_{action}$ 信号。
        * **目的：** 直观展示我们如何用一个简单的信号 来标定一个更鲁棒的高级信号。
    * **`Figure S4: $\theta_{on}$ 参数扫描曲线`**
        * **内容：** 绘制 F1-score (或 Youden's J index) 作为 $E_{action}$ 阈值 $\theta_{on}$ 的函数曲线。
        * **目的：** 证明我们选择的 $\theta_{on}$ 是最优或接近最优的，并且该曲线在最优点附近相对平稳（即鲁棒）。
    * **`Figure S5: Hysteresis 与 Debounce 敏感性分析`**
        * **内容：**
            1.  固定最优的 $\theta_{on}$，绘制分割 F1-score (对比 GT) 随 $\theta_{off}$ 比例 $r$ ($\theta_{off} = r \cdot \theta_{on}$) 变化的曲线。
            2.  固定最优的 $\theta_{on}, \theta_{off}$，绘制 F1-score 随 debounce 时长 $u$ 和 $d$ 变化的曲线。
        * **目的：** 证明 Segmentor 对这些次要超参数不敏感，具有工业部署的稳定性。

### 1.3. 实验 3：Frozen Transformer 嵌入模型超参数搜索

* **目标：** 用实验数据支撑正文 3.3.1 节中对模型架构 ($L=4, H=4$) 和池化方式 (Mean-Pooling) 的选择。
* **产出图表：**
    * **`Table S2: 不同池化 (Pooling) 策略对聚类质量的影响`**
        * **内容：** 固定 $L=4, H=4$，比较不同池化策略对聚类指标的影响。
        * **列条目：** Embedding Method | Silhouette Score | Calinski-Harabasz Index
        * **行条目：**
            1.  Mean Pooling (Ours)
            2.  CLS Token Pooling
            3.  Max Pooling
            4.  Attention Pooling
    * **`Table S3: 不同 Transformer 架构对聚类质量的影响`**
        * **内容：** 固定 Mean-Pooling，比较不同架构参数（$L, H$）对聚类指标的影响。
        * **列条目：** L (Layers) | H (Heads) | $d$ (Dimension) | Silhouette Score | Calinski-Harabasz Index
        * **行条目：** (展示不同的组合，并高亮我们的选择: 4, 4, 256)

---

## 2. 实验结果的充分补充 (Extended Results)

### 2.1. 实验 4：海量定性分割结果 (Extensive Qualitative Segmentation)

* **目标：** 提供““海量””视觉证据，证明 $E_{action}$ 信号 相较于 Optical Flow 的**普遍优越性**，以及我们 Action Segmentor 的高精度。
* **产出图表：**
    * **`Figure S6 - S10: 更多长时序分割对比图`** (至少 5 张新图)
        * **内容：** 复制正文 Figure 4 的格式，但使用**不同**的、更长的（例如 60-90 秒）视频片段。
        * **每张图包含 4 条线：**
            1.  `Optical Flow Magnitude` (Baseline)
            2.  `Ours ($E_{action}$)` (Quantized)
            3.  `GT Action Boundaries` (Dashed Lines)
            4.  `Our Segmentor Output` (ON/OFF State)
        * **目的：** 穷举式地证明我们的信号在各种情况下都更清晰、更准确地捕捉了**语义边界**。
    * **`Figure S11: 失败案例 (Failure Cases) 分析`**
        * **内容：** 诚实地展示 1-2 个我们的方法失败或表现不佳的案例。
        * **示例：**
            * *漏检 (Missed Detection)*：一个非常细微的动作（如擦拭），其 $E_{action}$ 没能超过 $\theta_{on}$。
            * *错误分割 (False Positive)*：一个非任务相关的动作（如工人身体大幅晃动），意外触发了分割。
        * **目的：** 提升学术诚信，并为 4.6 节（局限性讨论）提供素材。

### 2.2. 实验 5：聚类结果的定性可视化 (Qualitative Clustering Visualization)

* **目标：** **直观地证明** K-Means 发现的聚类 具有高度的**语义一致性**，为正文 Figure 5 的抽象 UMAP 点云提供““具象””的证据。
* **产出图表：**
    * **`Figure S12: 动作聚类 0 (Cluster 0) 示例`**
        * **内容：** 一个 3x3 或 4x4 的**视频缩略图网格**。
        * **采样：** 从 K-Means 赋给 Cluster 0 的所有动作原语 $A_i$ 中，随机采样 9 或 16 个。
        * **展示：** 每个采样 $A_i$ 展示 3 帧（起始帧、中间帧、结束帧）。
        * **图注 (Caption)：** 必须明确指出该聚类对应的语义。（例如："Figure S12: ... 随机采样自 Cluster 0。该聚类清晰地对应于‘**拿起电动螺丝刀**’这一语义动作。"）
    * **`Figure S13: 动作聚类 1 (Cluster 1) 示例`**
        * **内容：** 同上，但采样自 Cluster 1。
        * **图注：** （例如："... 对应于‘**拧紧外壳螺丝**’这一语义动作。"）
    * **`Figure S14: 动作聚类 2 (Cluster 2) 示例`**
        * **内容：** 同上，但采样自 Cluster 2。
        * **图注：** （例如："... 对应于‘**放置成品电机**’这一语义动作。"）
    * **(依此类推，直到 Cluster K-1)**

---

## 3. 数据集与讨论的扩展 (Dataset & Discussion)

### 3.1. 实验 6：数据集详情与标注规范

* **目标：** 详细介绍我们的真实工业数据集，证明其““真实性””和““复杂性””，并说明用于评估的 Ground-Truth 的高质量。
* **产出图表：**
    * **`Figure S15: 工业装配工位设置`**
        * **内容：**
            1.  工位现场照片。
            2.  `Top-down View` 的示例帧。
            3.  `Exocentric View` 的示例帧。
        * **目的：** 让审稿人直观理解我们的数据采集环境。
    * **`Table S4: 数据集与标注详情`**
        * **内容：** 汇总数据集的关键信息。
        * **行条目：**
            * 数据集总时长 (e.g., ~100 hours)
            * 测试集时长 (e.g., ~2 hours)
            * 视频分辨率 / FPS
            * 工位“可数”动作词汇表 (e.g., K=3: 'Pick Screwdriver', 'Fasten Screw', 'Wipe Casing')
            * 标注工具 (e.g., ELAN, VGG-VIA)
            * 标注者间一致性 (Inter-Annotator Agreement) (e.g., F1@5s, mAP@0.5)
        * **目的：** 证明我们用于评估（表1）的 GT 是可靠且高质量的。