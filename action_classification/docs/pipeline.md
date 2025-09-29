
#### UnSupervised Classification

#####  Code Indices本身体检 （质量与结构分析）- [useful after refactor]

conda run -n amplify_mt python -m action_classification.analysis.code_indices_analysis_offline \
  --json-root "/home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/inference_outputs/result_20250923_153828_complete_stride16/json" \
  --out-dir "action_classification/analysis/code_indices" \
  --subset-per-class 1000 \
  --do-stats --do-tfidf --do-bigram --do-umap --do-cluster \
  --bigram-dim 8192

conda run -n amplify_mt python -m action_classification.analysis.code_indices_analysis_offline \
  --json-root "/home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/inference_outputs/result_20250923_134659_complete_stride4/json" \
  --out-dir "action_classification/analysis/code_indices" \
  --subset-per-class 1000 \
  --do-stats --do-tfidf --do-bigram --do-umap --do-cluster \
  --bigram-dim 8192

----
分析在线推理得到的变长Code Inidces Sequences:

python action_classification/analysis/unsupervised_quality_report.py \
  --json-root /media/johnny/Data/data_motion_tokenizer/online_inference_results_codebook2048_stride4 \
  --out-dir action_classification/analysis/unsup_quality \
  --do-umap \
  --kmin 2 --kmax 12 \
  --kmeans-repeats 2 \
  --kmeans-select-by silhouette \
  --report-topk-candidates 5 \
  --silhouette-sample-size 5000 \
  --hdb-min-cluster-size-grid "20,40,80,120"

----
比较： 用“旧短序列 run”和“新长序列 run”对

python action_classification/analysis/summarize_sequence_quality.py \
  --runs \
    action_classification/analysis/unsup_quality/20250925_192325 \
    action_classification/analysis/unsup_quality/20250926_121641 \
  --labels short_seq long_seq \
  --out action_classification/analysis/unsup_quality/summary

##### BoW/Avg基线评估 - [useful after refactor]

- Bow (TF-IDF+L2, KMeans) & Avg (GMM) 

  python -m action_classification.evaluation.cluster_eval \
    --json-root "/home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/inference_outputs/result_20250923_153828_complete_stride16/json" \
    --config "/home/johnny/johnny_ws/motion_tokenizer/action_classification/configs/eval_config.yaml" \
    --feature both \
    --out-dir "/home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/out"



##### LSTM 训练 - [useful after refactor]

  python -m action_classification.embedding.train \
    --json-root /media/johnny/Data/data_motion_tokenizer/online_inference_results_codebook2048_stride4 \
    --config action_classification/configs/sequence_embed.yaml \
    --out-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/models/

  python -m action_classification.embedding.train \
    --json-root /home/johnny/johnny_ws/motion_tokenizer/amplify_motion_tokenizer/inference_outputs/result_20250923_153828_complete_stride16/json \
    --config action_classification/configs/sequence_embed.yaml \
    --out-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/models/
    --device cuda

##### LSTM 推理 - [useful after refactor]


  python -m action_classification.scripts.infer_sequence_embed_lstm \
    --json-root /media/johnny/Data/data_motion_tokenizer/online_inference_results_codebook2048_stride4 \
    --model-pt /home/johnny/johnny_ws/motion_tokenizer/action_classification/models/20250926_153711/model_best.pt \
    --out-dir action_classification/seq_embed_infer/ \
    --l2-normalize 


##### LSTM 嵌入向量 embed.npy 聚类结果评估

 - KMeans
 
python -m action_classification.evaluation.embed_cluster_eval \
  --embed-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/seq_embed_infer/20250926_162909 \
  --config action_classification/configs/eval_config.yaml \
  --out-dir action_classification/analysis/embed_cluster_eval_20250926_162909

- HDBSCAN拟合

  python -m action_classification.clustering.fit_hdbscan \
    --embed-dir action_classification/seq_embed_infer/20250926_154112 \
    --config action_classification/configs/eval_config.yaml \
    --out-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/seq_embed/20250923_173641_complete_stride4/hdbscan_fits

- BayesGMM分类

  python -m action_classification.clustering.fit_bayes_gmm \
    --embed-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/seq_embed/20250923_173641_complete_stride4 \
    --config action_classification/configs/eval_config.yaml \
    --out-dir /home/johnny/johnny_ws/motion_tokenizer/action_classification/seq_embed/20250923_173641_complete_stride4/bayes_gmm_fits



##### 在线推理 （LSTM+HDBSCAN） - 有待检查：是否以raw_video作为输入 

python -m action_classification.scripts.online_cluster_infer \
  --encoder-model action_classification/seq_embed/20250923_173641_complete_stride4/model_best.pt \
  --cluster-dir action_classification/seq_embed/20250923_173641_complete_stride4/hdbscan_fits/<TS> \
  --prob-thr 0.2 \
  --json-root <JSON_ROOT> \
  --out-jsonl results.jsonl