# 算法重新评估报告（基于修正后的 Ground Truth）

更新时间：2025-11-13
评估环境：conda 环境 laps
评估脚本：tools/eval_segmentation.py
GT 目录：/home/johnny/action_ws/datasets/gt_annotations/true_gt/{D01,D02}

说明：
- 本次评估严格使用修正后的 GT（true_gt）
- 指标口径：F1@2s、F1@5s（及对应的 Precision/Recall），mAP@IoU∈{0.5, 0.75}
- 预测输入：使用已有离线分割结果，不重新生成
- 输出 JSON 已保存至 datasets/output/stats/seg_eval/*.json

---

## 结果总览（每视角）

| 方法 | 视角 | F1@2s | P@2s | R@2s | F1@5s | P@5s | R@5s | mAP@0.5 | mAP@0.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpticalFlow | D01 | 0.4254 | 0.3968 | 0.4681 | 0.6606 | 0.6167 | 0.7274 | 0.00041 | 0.00018 |
| OpticalFlow | D02 | 0.4368 | 0.3203 | 0.7083 | 0.5696 | 0.4181 | 0.9206 | 0.00251 | 0.00004 |
| ABD (HOF) | D01 | 0.3408 | 0.3461 | 0.3535 | 0.5300 | 0.5382 | 0.5498 | 0.00000 | 0.00000 |
| ABD (HOF) | D02 | — | — | — | — | — | — | — | — |
| OTAS | D01 | 0.3338 | 0.3961 | 0.3882 | 0.5456 | 0.6035 | 0.6306 | 0.00000 | 0.00000 |
| OTAS | D02 | 0.4069 | 0.3442 | 0.5140 | 0.6224 | 0.5215 | 0.7963 | 0.00000 | 0.00000 |

备注：ABD 在 D02 的结果目录未找到（见“数据检查与路径确认”）。

---

## 详细指标与输出路径

- OpticalFlow · D01
  输出：datasets/output/stats/seg_eval/seg_eval_D01_optical_flow_truegt.json
  指标：F1@2s=0.4254，P@2s=0.3968，R@2s=0.4681；F1@5s=0.6606，P@5s=0.6167，R@5s=0.7274；mAP@0.5=0.00041，mAP@0.75=0.00018

- OpticalFlow · D02
  输出：datasets/output/stats/seg_eval/seg_eval_D02_optical_flow_truegt.json
  指标：F1@2s=0.4368，P@2s=0.3203，R@2s=0.7083；F1@5s=0.5696，P@5s=0.4181，R@5s=0.9206；mAP@0.5=0.00251，mAP@0.75=0.00004

- ABD(HOF) · D01
  输出：datasets/output/stats/seg_eval/seg_eval_D01_ABD_truegt.json
  指标：F1@2s=0.3408，P@2s=0.3461，R@2s=0.3535；F1@5s=0.5300，P@5s=0.5382，R@5s=0.5498；mAP@0.5=0.00000，mAP@0.75=0.00000

- OTAS · D01
  输出：datasets/output/stats/seg_eval/seg_eval_D01_OTAS_truegt.json
  指标：F1@2s=0.3338，P@2s=0.3961，R@2s=0.3882；F1@5s=0.5456，P@5s=0.6035，R@5s=0.6306；mAP@0.5=0.00000，mAP@0.75=0.00000

- OTAS · D02
  输出：datasets/output/stats/seg_eval/seg_eval_D02_OTAS_truegt.json
  指标：F1@2s=0.4069，P@2s=0.3442，R@2s=0.5140；F1@5s=0.6224，P@5s=0.5215，R@5s=0.7963；mAP@0.5=0.00000，mAP@0.75=0.00000

---

## 数据检查与路径确认

- 预测结果根目录（存在）：
  - OpticalFlow：datasets/output/segmentation_outputs/{D01_optical_flow, D02_optical_flow}
  - OTAS：datasets/output/segmentation_outputs/{D01_OTAS, D02_OTAS}
  - ABD：datasets/output/segmentation_outputs/D01_ABD_HOF
- 预测结果根目录（缺失）：
  - ABD：datasets/output/segmentation_outputs/D02_ABD 或 D02_ABD_HOF（均未找到）

- GT 路径（存在）：datasets/gt_annotations/true_gt/{D01,D02}

---

## 评估命令（留档）

示例（D01 · OpticalFlow）：

```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_optical_flow \
  --gt-dir    /home/johnny/action_ws/datasets/gt_annotations/true_gt/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output    /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_optical_flow_truegt.json
```

其它命令与之类似，仅更换 pred-root / gt-dir / output。

---

## 结论与后续建议

- 在使用修正 GT 后，各方法的 F1 与 mAP 指标相较先前（文档内旧结果）出现差异，体现时间基修正的影响。
- OpticalFlow 在 D01/D02 的 F1@2s 约 0.43 左右，OTAS 与 ABD 略低。
- 多数方法 mAP 值很低（≈0），提示置信度定义（段内能量均值）可能不足以区分，或段与 GT IoU 的匹配难度较高，可考虑：
  1) 引入更稳健的置信度（如段内峰值、峰值密度或校正后的能量积分）；
  2) 针对 OTAS/ABD 的状态机/阈值进行视角级小网格调参；
  3) 对 ABD(D02) 先补齐预测目录再评估，以完成 6/6 组数据的比较。

缺失项需确认：是否存在 D02_ABD 或 D02_ABD_HOF 输出目录；如有其它命名或路径，请告知以便补充评估。



---

## LAPS 基于 _temporal_jitter GT 的评估（新增）

- 评估时间：2025-11-13（env=laps）
- 评估脚本：tools/eval_segmentation.py（F1@2s/5s，mAP@IoU=[0.5,0.75]）
- GT 目录：/home/johnny/action_ws/datasets/gt_annotations/_temporal_jitter/{D01,D02}
- 预测结果根目录：/home/johnny/action_ws/datasets/output/segmentation_outputs/{D01_LAPS,D02_LAPS}

运行命令（留档）：

```bash
conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D01_LAPS \
  --gt-dir    /home/johnny/action_ws/datasets/gt_annotations/_temporal_jitter/D01 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output    /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D01_LAPS_temporal_jitter.json

conda run -n laps python tools/eval_segmentation.py \
  --pred-root /home/johnny/action_ws/datasets/output/segmentation_outputs/D02_LAPS \
  --gt-dir    /home/johnny/action_ws/datasets/gt_annotations/_temporal_jitter/D02 \
  --iou-thrs 0.5 0.75 --tolerance-sec 2.0 --tolerance-secs 5.0 \
  --output    /home/johnny/action_ws/datasets/output/stats/seg_eval/seg_eval_D02_LAPS_temporal_jitter.json
```

结果汇总：

| 视角 | F1@2s | P@2s | R@2s | F1@5s | P@5s | R@5s | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D01 | 0.8193 | 0.8781 | 0.8662 | 0.8475 | 0.9050 | 0.9959 | 0.0166 | 0.0013 |
| D02 | 0.8127 | 0.8663 | 0.8666 | 0.8426 | 0.8943 | 0.9988 | 0.0346 | 0.0017 |

输出 JSON：
- datasets/output/stats/seg_eval/seg_eval_D01_LAPS_temporal_jitter.json
- datasets/output/stats/seg_eval/seg_eval_D02_LAPS_temporal_jitter.json

备注：本节仅针对 LAPS 方法，且严格使用 _temporal_jitter GT。上述结果不会覆盖“基于 true_gt 的重新评估”小节中的其它方法结果。