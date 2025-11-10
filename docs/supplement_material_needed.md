这是一个非常好的问题，对实验计划进行分类是高效执行的关键。作为你的“ActionPaper-Writer”，我来帮你详细拆解。

你这份实验计划中，大部分图表（Figures）都是**代码实验**的直接产出，而表格（Tables）和少数示意图则属于**人工整理**或**收集**。

我已将你的计划重组为两个独立的部分：
1.  **A部分：代码实验执行计划**（需要你编写并运行代码来生成数据和图表）。
2.  **B部分：人工整理与收集**（你需要手动制图、截图、或汇总已知信息）。

---

# A部分：代码实验执行计划（Code-Driven Experiments）

这部分是核心工作，所有产出都依赖于你编写的Python脚本（例如，使用 `PyTorch`, `Matplotlib`, `scikit-learn` 等）。

## 1. 实验 1.1：Motion Tokenizer 训练 (产出 Fig. S2)

* **目标：** 证明 $M_{\theta}$ 训练收敛。
* **产出：** `Figure S2: Motion Tokenizer 训练损失曲线`
* **技术实现细节：**
    1.  运行 $M_{\theta}$ 的训练脚本。
    2.  确保训练过程中记录了每个 epoch 或 step 的 `Cross-Entropy Loss`（例如，使用 `wandb`, `TensorBoard`, 或简单的 CSV 日志）。
    3.  **可视化脚本 (`plot_loss_curve.py`)：**
        * 读取损失日志文件。
        * 使用 `matplotlib` 绘制损失随训练（Epochs/Steps）下降的曲线。
        * 添加图例、x/y轴标签，并保存为高分辨率图像（如 `.pdf` 或 `.png`）。

## 2. 实验 1.2：Action Segmentor 阈值优化 (产出 Fig. S3, S4, S5)

* **目标：** 详细展示 $\theta_{on}$ 的无监督标定过程及其鲁棒性。
* **产出：**
    * `Figure S3: 代理信号 (Proxy Signal) 可视化`
    * `Figure S4: $\theta_{on}$ 参数扫描曲线`
    * `Figure S5: Hysteresis 与 Debounce 敏感性分析`
* **技术实现细节：**
    1.  **`Figure S3` (可视化脚本 `plot_proxy_vs_e_action.py`)：**
        * 加载一个典型的验证视频片段。
        * 计算低级“速度能量”（Proxy Signal），例如：`torch.norm(keypoint_velocities[:, 1:] - keypoint_velocities[:, :-1], dim=-1)`。
        * 使用 `skimage.filters.threshold_otsu` 或 `np.quantile` 从代理信号生成 $y_{pseudo}$。
        * 通过 $M_{\theta}$ 计算我们的 $E_{action}$ 信号。
        * 在同一张图上绘制这三条时间序列曲线。
    2.  **`Figure S4` (扫描脚本 `sweep_theta_on.py`)：**
        * 加载验证集的所有 $E_{action}$ 信号和 $y_{pseudo}$ 标签。
        * 定义一个 $\theta_{on}$ 的扫描范围（例如 `np.linspace(0.1, 1.0, 50)`）。
        * **循环体：**
            * 对于每个 `theta` 值，计算 $E_{action} > theta$ 得到的 $y_{pred}$。
            * 计算 `f1_score(y_pseudo, y_pred)` 或 `j_index = recall + specificity - 1`。
        * 绘制 F1-score/J-Index 随 $\theta_{on}$ 变化的曲线，找到峰值。
    3.  **`Figure S5` (敏感性分析脚本 `analyze_sensitivity.py`)：**
        * 加载**测试集**的 $E_{action}$ 信号和**Ground Truth (GT)** 标注。
        * 固定从 `Fig S4` 得到的**最优 $\theta_{on}$**。
        * **实验a (Hysteresis)：** 循环遍历 $r$（例如 `np.linspace(0.1, 1.0, 20)`），计算 $\theta_{off} = r \cdot \theta_{on}$，运行完整的 Action Segmentor（带 hysteresis 逻辑），并计算 F1-score (对比 **GT**)。绘制 F1 随 $r$ 变化的曲线。
        * **实验b (Debounce)：** 固定最优的 $\theta_{on}, \theta_{off}$，循环遍历 $u$ 和 $d$ 的不同组合（例如 `[1, 3, 5, 7]` 帧），计算 F1-score (对比 **GT**)。绘制 F1 随 $u, d$ 变化的曲线（或热图）。

## 3. 实验 1.3：Frozen Transformer 超参数搜索 (产出 Table S2, S3 的数据)

* **目标：** 为我们对池化策略和模型架构的选择提供数据支撑。
* **产出：** `Table S2` 和 `Table S3` 中的所有**数值**（表格本身在B部分整理）。
* **技术实现细节：**
    1.  **前置步骤：** 运行 Action Segmentor，保存所有分割出的动作原语（`A_i`）及其对应的 $S_{q,i}$（连续量化向量序列）。
    2.  **主脚本 (`run_clustering_ablation.py`)：**
        * 加载所有 $S_{q,i}$。
        * **实验a (池化策略)：**
            * 固定 $L=4, H=4, d=256$。
            * 循环遍历 `pooling_strategy = ['mean', 'cls', 'max', 'attention']`。
            * 对于每种策略，通过 Frozen Transformer 编码器得到所有原语的嵌入 $e_i$。
            * 运行 `KMeans(n_clusters=K)`。
            * 计算 `silhouette_score(embeddings, labels)` 和 `calinski_harabasz_score(embeddings, labels)`。
            * 打印结果。
        * **实验b (架构)：**
            * 固定 `pooling_strategy = 'mean'`。
            * 循环遍历不同的架构组合 `(L, H, d)`（例如 `(2,2,128), (4,4,128), (4,4,256), ...`）。
            * 重复上述的嵌入、KMeans、计算指标的步骤。
            * 打印结果。

## 4. 实验 2.1：海量定性分割结果 (产出 Fig. S6 - S11)

* **目标：** 提供压倒性的视觉证据，证明 $E_{action}$ 优于光流。
* **产出：**
    * `Figure S6 - S10: 更多长时序分割对比图`
    * `Figure S11: 失败案例 (Failure Cases) 分析`
* **技术实现细节：**
    1.  **核心可视化脚本 (`plot_qualitative_segmentation.py`)：**
        * **输入：** 一个视频片段的ID（或时间戳）。
        * **加载数据：**
            * 加载对应的视频帧。
            * 加载 Ground Truth (GT) 标注文件，在图上绘制为**虚线**。
        * **计算信号（在线）：**
            1.  计算 `Optical Flow Magnitude` (Baseline)，绘制为**红线**。
            2.  计算 `Ours ($E_{action}$)` (Quantized)，绘制为**蓝线**。
            3.  运行 `Our Segmentor Output` (完整的Hysteresis状态机)，绘制为**黑色阶梯线** (ON/OFF State)。
        * **绘图：** 使用 `matplotlib.pyplot` 将所有 4-5 条数据（GT, Flow, $E_{action}$, Output）绘制在同一张时序图上。
    2.  **执行：** 运行此脚本 5-7 次，精心挑选能展示 $E_{action}$ 优越性（S6-S10）和局限性（S11）的片段。

## 5. 实验 2.2：聚类结果的定性可视化 (产出 Fig. S12 - S14)

* **目标：** 具象化地展示 K-Means 聚类的语义一致性。
* **产出：** `Figure S12, S13, S14: 动作聚类 示例`
* **技术实现细节：**
    1.  **前置步骤：** 运行实验 1.3，得到所有动作原语 $A_i$ 最终的 K-Means 聚类标签 `cluster_id`。
    2.  **可视化脚本 (`visualize_clusters.py`)：**
        * **输入：** `Cluster_ID` (例如 0, 1, 2, ... K-1)。
        * **逻辑：**
            * 筛选出所有 `cluster_id == K` 的动作原语 $A_i$（即它们的视频路径/时间戳）。
            * 从该列表中**随机采样** 9 或 16 个 $A_i$。
            * 创建一个 3x3 或 4x4 的 `matplotlib` 子图网格。
            * **循环体：** 对于每个采样的 $A_i$，加载其视频，提取**起始帧**、**中间帧**、**结束帧**。
            * 将这三帧（或只显示中间帧）绘制到子图网格的对应位置。
        * **执行：** 为 $K=0, 1, 2, ...$ 分别运行此脚本，生成 `Fig S12` 到 `Fig S14+`。

---

# B部分：人工整理与收集（Manually Compiled & Collected Items）

这部分内容不需要新的代码实验，而是依赖于你的绘图、截图、以及对已知实验配置和数据信息的““转录””。

## 1. 实验 1.1：Motion Tokenizer (产出 Fig. S1, Table S1)

* **`Figure S1: Motion Tokenizer 详细架构图`**
    * **类型：** **人工绘制**。
    * **工具：** 使用 TikZ (for LaTeX), PowerPoint, draw.io 或你熟悉的任何绘图工具。
    * **内容：** 根据你已实现的代码架构，绘制包含 $E_{\theta}, D_{\theta}, \text{FSQ}$ 的流程图，并仔细标注输入输出的张量维度（如 $\mathbb{R}^{T \times N \times 2}$）。
* **`Table S1: Motion Tokenizer 训练超参数`**
    * **类型：** **人工整理**。
    * **内容：** 打开你的训练配置文件（例如 `config.yaml` 或 `train.py`），将你最终使用的**所有超参数**（层数、头数、Batch Size、LR等）手动复制粘贴到 LaTeX 的 `\begin{tabular}` 环境中。

## 2. 实验 1.3：Frozen Transformer (产出 Table S2, S3)

* **`Table S2: 不同池化 (Pooling) 策略对聚类质量的影响`**
* **`Table S3: 不同 Transformer 架构对聚类质量的影响`**
    * **类型：** **人工整理**。
    * **内容：** 从**A部分 - 实验 1.3** 的脚本运行**输出**（终端打印或日志文件）中，**手动复制** `Silhouette Score` 和 `Calinski-Harabasz Index` 的数值，填写到 LaTeX 表格中。

## 3. 实验 3.1：数据集详情 (产出 Fig. S15, Table S4)

* **`Figure S15: 工业装配工位设置`**
    * **类型：** **人工收集与排版**。
    * **内容：**
        1.  从实验室或数据提供方获取一张工位的**真实现场照片**。
        2.  从你的数据集中截取 2 张有代表性的示例帧（`Top-down View` 和 `Exocentric View`）。
        3.  使用 PowerPoint 或 `matplotlib` 将这 3 张图(a, b, c)拼排在一起。
* **`Table S4: 数据集与标注详情`**
    * **类型：** **人工整理**。
    * **内容：** 整理关于数据集的**已知事实**并填入表格。
    * **信息来源：**
        * `数据集总时长`, `分辨率/FPS`：来自数据提供方或 `ffprobe`。
        * `测试集时长`, `可数动作词汇表`：来自你的 GT 标注文件。
        * `标注工具`：(e.g., ELAN) 你在标注时使用的工具。
        * `标注者间一致性`：这个**数值** (e.g., F1@5s) 本身可能来自一个你**过去**运行过的脚本（比较两个标注者的 `.eaf` 文件），但现在你只是**引用**这个已知结果，并将其**手动填入**表格。

---

### 总结

* **立即开始A部分：** 这部分是工作量的核心，需要你集中时间编写和运行脚本。
* **并行处理B部分：** 在A部分代码运行的间隙（例如模型训练时），你可以去完成B部分的绘图和表格整理工作。

这个拆分后的计划应该非常清晰了。请按照这个蓝图执行，特别是A部分的实现细节，这将确保我们补充材料的质量。