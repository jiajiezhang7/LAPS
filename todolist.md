# Motion Tokenizer 码本坍塌调试与修复待办

- **[目标]**
  - 修复 FSQ 码本严重坍塌（unique_codes≈1、entropy≈0、perplexity≈1/K）。
  - 保留最小侵入原则，逐步引入损失与采样修复，验证有效性。

- **[现象与证据]**
  - W&B：`train_unique_codes_step=1`、`train_entropy_step=0`、`train_perplexity_step=1`、`train_codebook_perplexity≈0.0041667`；代码中已对 FSQ 多位 digits 做混合进制展平统计（`amplify/train_motion_tokenizer.py`）。
  - 可视化“看起来还行”：相对分类目标在 20FPS、T=16 时强中心类不平衡；解码器有强位置/视角嵌入，codes 退化时仍可输出“模板化”轨迹；像素容忍度指标易被“全中心”解通过。

- **[根因]**
  - 损失只含相对分类 CE（`relative_ce`），不包含码本多样性/承诺正则（`amplify/amplify/models/motion_tokenizer.py::get_loss`）。
  - 数据强不平衡：相邻帧位移极小，中心类占绝对多数，CE 推动“全中心”。
  - 解码器模板化：`VAEDecoder` 拼 `track_pos_emb`/`view_emb`，弱化 codes 作用。

- **[修复策略（按优先级）]**
  - 训练损失：
    - 加“码本多样性熵正则”`loss_div = λ_div * (1 - H_norm)`，H_norm 为 codes 使用分布的归一化熵。
    - 将 CE 改为/叠加 Focal Loss（γ=2.0, α=0.25 起），缓解中心类不平衡。
  - 训练采样：关闭 `quick: true`，扩大样本覆盖。
  - 编码器反坍塌（可选）：量化前 `z` 加小噪声或方差鼓励正则。
  - 结构微调（最后考虑）：降低解码器对 `track/view` 嵌入依赖。
  - 码本大小：2048→1024/512 仅辅助，不是根治；先做损失端修复后再对比。

- **[本次修改计划]**
  1) 配置：`amplify/cfg/train_motion_tokenizer.yaml`
     - 将 `quick: false`。
     - 在 `loss:` 下新增：
       - `codebook_diversity_weight: 0.001`
       - `use_focal: true`
       - `focal_gamma: 2.0`
       - `focal_alpha: 0.25`
  2) 损失实现：`amplify/amplify/models/losses.py`
     - `compute_relative_classification_loss()` 增加可选 Focal 分支（保持向后兼容）。
  3) 码本熵正则：`amplify/amplify/models/motion_tokenizer.py`
     - `get_loss(...)` 新增 `codebook_indices` 入参；读取 `cfg.loss.codebook_diversity_weight` 计算 `loss_div` 并叠加。
  4) 训练脚本：`amplify/train_motion_tokenizer.py`
     - 在 `train_epoch()` 调用 `model.get_loss(..., codebook_indices=codebook_indices)` 启用正则。
  5) 观测与验证：
     - 关注 `train_entropy_norm_step`、`train_unique_codes_step`、`train_code_usage_hist`、`metrics/nonzero_pred_percent` 的上升。

- **[验证步骤]**
  - Step 0：仅关闭 `quick`，跑 1–2 epoch，记录基础提升。
  - Step 1：启用熵正则（λ=1e-3）。
  - Step 2：启用 Focal（γ=2.0, α=0.25）。
  - Step 3：必要时加 `z` 抖动或方差正则（低权重）。
  - Step 4：在损失稳定后再做 2048/1024/512 的 codebook 对比。
