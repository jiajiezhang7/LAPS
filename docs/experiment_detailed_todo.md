## 实验详细 ToDo 与进度（最新）

- 2025-11-07 · OTAS 训练后评估（已完成）
  - mean_error（12 个 .pkl）：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/mean_error
  - 边界检测 detect_bdy（12 个 .pkl）：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/detect_seg
  - 结果适配：
    - D01 输出：/home/johnny/action_ws/datasets/output/segmentation_outputs/D01_OTAS（6 个 segments.json）
    - D02 输出：/home/johnny/action_ws/datasets/output/segmentation_outputs/D02_OTAS（6 个 segments.json）
  - 评估结果：
    - D01（/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_OTAS_trained.json）
      - F1@2s=0.1525, F1@5s=0.2792, mAP@0.5=0.00125, mAP@0.75=0.00006
    - D02（/home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_OTAS_trained.json）
      - F1@2s=0.2139, F1@5s=0.3189, mAP@0.5=0.00304, mAP@0.75=0.000006
  - 备注：严格遵守 conda 环境规则（otas：训练/验证；laps：评估流水线）



- 2025-11-07 · OTAS 边界检测参数网格搜索（order）
  - 环境/目录：conda=laps；/home/johnny/action_ws/comapred_algorithm/OTAS/code
  - 参数：order ∈ {11,13,15,17,19}
  - 评估输出：
    - /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_OTAS_trained_order{11,13,15,17,19}.json
    - /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_OTAS_trained_order{11,13,15,17,19}.json
  - 结果（均相同）：
    - D01：F1@2s=0.1525, F1@5s=0.2792, mAP@0.5=0.00125, mAP@0.75=0.00006
    - D02：F1@2s=0.2139, F1@5s=0.3189, mAP@0.5=0.00304, mAP@0.75=0.000006
  - 结论：order 参数在 [11,19] 内对结果不敏感；建议后续扫描 threshold（如 {45,51,60}）或增加训练 epoch（如 20）后复评


- 2025-11-07 · OTAS 训练收敛性检查（证据）
  - 训练日志：/home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_20251106_112035.log（4 epochs）
  - 每 epoch 平均 batch loss：E0=0.6504 → E1=0.4733 → E2=0.4186 → E3=0.3887（持续下降）
  - 各 epoch 末尾（last100）均值：E0=0.5447，E1=0.4500，E2=0.4094，E3=0.3839（末尾仍在下降）
  - 全程最低 batch loss：0.3487（出现于 epoch3 后段；最佳权重保存：0.3536）
  - 结论：loss 在训练末期仍显著下降，最佳模型出现在最后阶段 → 训练尚未收敛；需延长 epoch 或调整学习率/调参后继续训练


- 2025-11-07 · 启动 OTAS 长训（20 epochs，进行中）
  - 环境/目录：conda=otas；/home/johnny/action_ws/comapred_algorithm/OTAS/code
  - 关键参数：num_epoch=20，batch_size=16，feature_model=tf，dataset=BF，gpu=0，num_workers=8
  - 输出目录：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf/model
  - 训练日志：/home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_e20.log
  - 预计耗时：单个 epoch ≈ 5.7 小时（历史观测），20 epoch 预计多天；期间将定期汇报进度


- 2025-11-07 · 停止 OTAS 合并长训（20 epochs，已取消）
  - 原因：按用户决策改为“按视角分别训练”；避免资源浪费与结果混淆
  - 进程：已手动停止（PID 2480295 及子进程）

- 2025-11-07 · 启动 OTAS 分视角训练（D01，进行中）
  - 环境/目录：conda=otas；/home/johnny/action_ws/comapred_algorithm/OTAS/code
  - 关键参数：--view D01，num_epoch=20，batch_size=16，feature_model=tf，gpu=0，num_workers=8
  - 输出目录：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf_D01/model
  - 训练日志：/home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_D01_e20.log
  - 数据过滤验证（--mode val --dry_run_dataset）：selected videos=6；"val" dataset size=24916

- 2025-11-07 · 启动 OTAS 分视角训练（D02，排队）
  - 说明：单 GPU（24GB）难以同时运行两个 bs=16 的训练；待 D01 完成后串行启动
  - 关键参数：--view D02，num_epoch=20，batch_size=16，feature_model=tf，gpu=0，num_workers=8
  - 输出目录：/home/johnny/action_ws/datasets/output/otas_out/OTAS/BF_tf_D02/model
  - 预定日志：/home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_D02_e20.log
  - 数据过滤验证（--mode val --dry_run_dataset）：selected videos=6；"val" dataset size=25381

# 训练 D01（20 epochs）
conda run -n otas python /home/johnny/action_ws/comapred_algorithm/OTAS/code/main.py \
  --mode train --dataset BF --feature_model tf \
  --output-path /home/johnny/action_ws/datasets/output/otas_out/ \
  --view D01 --num_epoch 20 --batch_size 16 --num_workers 8 --gpu 0 --num-gpus 1 \
  >> /home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_D01_e20.log 2>&1 &

# 训练 D02（20 epochs）
conda run -n otas python /home/johnny/action_ws/comapred_algorithm/OTAS/code/main.py \
  --mode train --dataset BF --feature_model tf \
  --output-path /home/johnny/action_ws/datasets/output/otas_out/ \
  --view D02 --num_epoch 20 --batch_size 16 --num_workers 8 --gpu 0 --num-gpus 1 \
  >> /home/johnny/action_ws/datasets/output/otas_out/OTAS/train_BF_tf_D02_e20.log 2>&1 &