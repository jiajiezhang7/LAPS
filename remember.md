### 依赖安装
- 以 amplify/README.md 为准

### 整个项目的结构
- amplify
    - 这个是一份开源代码，主要使用到其 Motion Tokenizer相关的训练和推理部分，关键是amplify/train_motion_tokenizer.py
    - 以及motion tokenizer的训练数据准备部分，关键是amplify/preprocessing/preprocess_my_segments.py
- video_action_segmenter
    - 这个模块利用训练好的 Motion Tokenizer,实时处理长视频，用于分割有意义的动作视频片段，并输出对应的latent action sequences
- action_classification
    - 这个模块针对video_action_segmenter输出的动作视频片段对应的latent action sequences，进行无监督聚类，用以验证和解释latent action sequences的意义（类似于论文LAPO的做法）

## 目前项目的核心瓶颈

- Motion Tokenizer训练时 codebook 坍塌 （已解决坍塌，但是正在提升codebook利用率）

## 几个数据路径

- preprocessed data

    /media/johnny/48FF-AA60/preprocessed_data_d02_m10

    /media/johnny/48FF-AA60/preprocessed_data_d01_m10

- raw videos

    /media/johnny/48FF-AA60/raw_videos/raw_videos_d01_910_filterstatic

    /media/johnny/48FF-AA60/raw_videos/raw_videos_d02_910_filterstatic

- online inference output

    /media/johnny/48FF-AA60/online_inference_output/cb2048_stride4