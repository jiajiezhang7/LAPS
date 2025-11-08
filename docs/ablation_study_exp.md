# 🔬 论文消融实验 (Ablation Studies) 关键要点总结

本文档总结了 LAPS 管道（Latent Action-based Primitive Segmentation）消融实验（第 4.6 节/表 4）的核心发现，旨在指导关键设计选择的实践落地。

## 一、 核心组件及其验证目标

消融实验验证了三个关键设计对分割性能 (Seg. F1) 和聚类质量 (Cluster NMI) 的影响。

| 实验组 | 关键设计 | 验证目标 | 核心结果指标 |
| :--- | :--- | :--- | :--- |
| **信号源消融** | Latent Action Energy ($E_{\text{action}}$) 的计算域 | **分割**：确定最鲁棒的语义变化信号源 | Seg. F1 |
| **表征消融** | Domain-specific Motion Tokenizer ($M_\theta$) | **分割/聚类**：验证专用特征对精细任务的重要性 | Seg. F1, Cluster NMI |
| **编码器消融** | Frozen Transformer Encoder | **聚类**：验证时序动态建模对语义聚类的必要性 | Cluster NMI |

注：本文中的 Seg. F1 明确指 F1@2s（边界容差=2 秒）；除非另行说明，所有评估均在 D01 数据集上进行。

## 二、 关键发现与实践指导

| 消融配置 | Seg. F1 | Cluster NMI | 关键发现及实践指导 |
| :--- | :--- | :--- | :--- |
| **Full Pipeline (Ours)** | **46.3** | **0.65** | **基准**：代表最佳性能配置。 |
| **信号源消融** | | | |
| $\quad E_{\text{action}}$ from Pre-Quant. Latents | 41.7 | -- | **结论**：量化 (Quantization) 是必要的，它抽象了噪声并使潜在向量更具判别性。 |
| $\quad E_{\text{action}}$ from Raw Velocities | 30.2 | -- | **结论**：**必须**在抽象的潜在动作空间 (Latent Space) 而非低级物理空间（Raw Velocities）进行分割，以捕获语义意图变化。 |
| **表征消融** | | | |
| $\quad$ w/o $M_\theta$ (e.g., CLIP/IDT) | 25.8 | 0.21 | **结论**：**必须**训练领域专用的 Motion Tokenizer ($M_\theta$) 来捕捉工业任务的精细动作。通用视觉特征效果不佳。 |
| **编码器消融** | | | |
| $\quad$ w/o Transformer (Mean-pool) | -- | 0.38 | **结论**：**必须**使用 Frozen Transformer 或类似时序模型来编码动作序列，简单的平均池化无法实现高质量的语义聚类。 |

