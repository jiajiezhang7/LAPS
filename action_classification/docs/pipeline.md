
#### UnSupervised Classification

┌─────────────────────────────────────────────────────────────┐
│  Motion Tokenizer Inference Output (code_indices.json)      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  embedding/train.py             │
        │  (LSTM 序列编码器训练)           │
        │  - 输入: JSON 文件目录          │
        │  - 输出: model_best.pt          │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  scripts/infer_sequence_embed   │
        │  (生成嵌入向量)                  │
        │  - 输入: model_best.pt + JSON   │
        │  - 输出: embed.npy, labels.npy  │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  聚类方法选择                    │
        └────┬──────────────────────┬─────┘
             │                      │
    ┌────────▼──────────┐  ┌────────▼──────────┐
    │ fit_hdbscan.py    │  │ fit_bayes_gmm.py  │
    │ (异常检测)        │  │ (固定簇数)        │
    │ 输出:             │  │ 输出:             │
    │ - model_*.pkl     │  │ - model_*.pkl     │
    │ - cluster_meta    │  │ - cluster_meta    │
    │ - UMAP 可视化     │  │ - UMAP 可视化     │
    └────────┬──────────┘  └────────┬──────────┘
             │                      │
             └──────────┬───────────┘
                        │
        ┌───────────────▼────────────────┐
        │ online_cluster_infer.py        │
        │ (在线推理 - 可选)              │
        │ - 输入: 新的 code_indices      │
        │ - 输出: 聚类预测结果           │
        └────────────────────────────────┘


##### BoW/Avg基线评估 - [useful after refactor]

- Bow (TF-IDF+L2, KMeans) & Avg (GMM) 

  python -m action_classification.evaluation.cluster_eval \
    --json-root "/home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/inference_outputs/result_20250923_153828_complete_stride16/json" \
    --config "/home/johnny/johnny_ws/motion_tokenizer/action_classification/configs/eval_config.yaml" \
    --feature both \
    --out-dir "/home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/out"



##### 第一步：训练LSTM编码器 - [useful after refactor]

  python -m action_classification.embedding.train \
    --json-root /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4 \
    --config action_classification/configs/sequence_embed.yaml \
    --out-dir /home/johnny/action_ws/action_classification/models

  python -m action_classification.embedding.train \
    --json-root /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d01_m10_cb2048_stride4 \
    --config action_classification/configs/sequence_embed.yaml \
    --out-dir /home/johnny/action_ws/action_classification/models


---

##### 第2.5步： 人工筛选出300条带有标签的视频及其 code indices，并且确定聚类的簇的个数（退一步），用于查看聚类的效果

##### 第二步：生成嵌入向量 - [useful after refactor]


  python -m action_classification.scripts.infer_sequence_embed_lstm \
    --json-root /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d01_m10_cb2048_stride4 \
    --model-pt /home/johnny/action_ws/action_classification/models/d01/model_best.pt \
    --out-dir action_classification/seq_embed_infer/ \
    --l2-normalize 

  - 为带label的数据生成embeddings

  python -m action_classification.scripts.export_label_sequence_embeddings \
    --json-root /home/johnny/action_ws/label_data/d02 \
    --model-pt /home/johnny/action_ws/action_classification/models/d02/model_best.pt \
    --l2-normalize \
    --out-subdir embeddings \
    --codes-subdir codes

##### 第三步：执行聚类评估

- HDBSCAN拟合

  python -m action_classification.clustering.fit_hdbscan \
    --embed-dir action_classification/seq_embed_infer/d01 \
    --config action_classification/configs/hdbscan_config.yaml \
    --out-dir action_classification/results/hdbscan_fits/

  python -m action_classification.clustering.fit_hdbscan \
    --embed-dir action_classification/seq_embed_infer/d02 \
    --config action_classification/configs/hdbscan_config.yaml \
    --out-dir action_classification/results/hdbscan_fits/

- BayesGMM分类

  python -m action_classification.clustering.fit_bayes_gmm \
    --embed-dir action_classification/seq_embed_infer/d01 \
    --config action_classification/configs/bayes_gmm_config.yaml \
    --out-dir action_classification/results/bayes_gmm_fits/

    python -m action_classification.clustering.fit_bayes_gmm \
    --embed-dir action_classification/seq_embed_infer/d02 \
    --config action_classification/configs/bayes_gmm_config.yaml \
    --out-dir action_classification/results/bayes_gmm_fits/


- 为带label的数据执行聚类，评估聚类质量

  python -m action_classification.clustering.evaluate_label_cluster_quality \
    --embed-root /home/johnny/action_ws/label_data/d02 \
    --embed-subdir embeddings \
    --config action_classification/configs/bayes_gmm_config.yaml \
    --n-components 6 \
    --out-dir action_classification/results/label_bayes_gmm_d02


- ToDo

  从第一性原理出发，再次思考如何对latent code indices聚类


##### （待使用-Pending）在线推理 （LSTM+HDBSCAN） - 有待检查：是否以raw_video作为输入 

python -m action_classification.scripts.online_cluster_infer \
  --encoder-model action_classification/seq_embed/d01/model_best.pt \
  --cluster-dir action_classification/seq_embed/d01/hdbscan_fits/<TS> \
  --prob-thr 0.2 \
  --json-root <JSON_ROOT> \
  --out-jsonl results.jsonl