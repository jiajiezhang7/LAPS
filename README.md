# LAPS: Latent Action-based Primitive Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end, unsupervised pipeline for automatic action primitive discovery from raw industrial videos. LAPS segments semantically coherent action primitives and their latent representations for VLA (Vision-Language-Action) pre-training.

## Key Features

- **Unsupervised**: No manual annotations required
- **End-to-end**: From raw video to structured action primitives
- **Real-time capable**: Online streaming inference support
- **Latent Action Energy**: Novel metric for action boundary detection in abstract latent space

## Pipeline Overview

```
Phase 1: Keypoint Tracking     Phase 2: Motion Tokenization      Phase 3: Unsupervised Discovery
      (CoTracker)              & Action Segmentation                  (Clustering & VLM)
          ↓                            ↓                                    ↓
    Raw Video  →  Keypoint Tracks  →  Latent Codes  →  Action Primitives  →  Action Library
                                       + Energy
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/LAPS.git
cd LAPS

# Create conda environment
conda create -n laps python=3.10
conda activate laps

# Install dependencies
pip install -r amplify/requirements.txt

# Optional: Install CoTracker (for keypoint tracking)
pip install git+https://github.com/facebookresearch/co-tracker.git

# Optional: Install visualization dependencies
pip install umap-learn plotly scikit-learn
```

## Quick Start

### 1. Data Preparation

Organize your raw videos:
```
data/
├── raw_videos/           # Your raw video files
│   ├── video1.mp4
│   └── video2.mp4
├── raw_video_segments/   # Split video segments (for training)
└── preprocessed_data/    # Output of preprocessing
```

### 2. Preprocess Videos (Phase 1)

Extract keypoint trajectories using CoTracker:

```bash
# Edit config: amplify/cfg/preprocessing/preprocess_my_segments.yaml
# Set: source, output_dir

cd amplify
python -m preprocessing.preprocess_my_segments
```

### 3. Train Motion Tokenizer (Phase 2a)

```bash
# Edit config: amplify/cfg/train_motion_tokenizer.yaml
# Set: root_dir, video_root, wandb_entity, run_name

python amplify/train_motion_tokenizer.py \
  root_dir=./data/preprocessed_data \
  run_name=my_experiment \
  num_epochs=5 \
  batch_size=8
```

### 4. Run Action Segmentation (Phase 2b)

```bash
# Edit config: video_action_segmenter/params.yaml
# Set: checkpoint_path, input.dir, segmentation.output_dir

python -m video_action_segmenter.stream_inference \
  --params video_action_segmenter/params.yaml
```

**Output:**
- Segmented video clips: `output/segmentation_outputs/*/segmented_videos/*.mp4`
- Latent code indices: `output/segmentation_outputs/*/code_indices/*.codes.json`

### 5. Unsupervised Discovery (Phase 3)

Visualize and cluster action primitives:

```bash
python umap_vis/scripts/sequence_model_embedding.py \
  --data-dir ./output/segmentation_outputs/YOUR_DATASET \
  --fig-dir ./output/figures \
  --stats-dir ./output/statistics \
  --use-best-grid-config \
  --metric cosine
```

## Evaluation

Evaluate segmentation quality against ground truth:

```bash
python tools/eval_segmentation.py \
  --pred-root ./output/segmentation_outputs/YOUR_DATASET \
  --gt-dir ./data/gt_annotations/YOUR_DATASET \
  --iou-thrs 0.5 0.75 \
  --tolerance-sec 2.0 5.0 \
  --output ./output/eval_results.json
```

## Project Structure

```
LAPS/
├── amplify/                    # Motion Tokenizer framework
│   ├── amplify/               # Core models and utilities
│   ├── cfg/                   # Configuration files
│   ├── preprocessing/         # Data preprocessing
│   └── train_motion_tokenizer.py
├── video_action_segmenter/    # Action segmentation module
│   ├── stream_inference.py    # Main inference entry
│   ├── stream_utils/          # Streaming utilities
│   └── params.yaml            # Configuration template
├── umap_vis/                  # Visualization and clustering
│   └── scripts/               # Analysis scripts
├── tools/                     # Evaluation tools
│   └── eval_segmentation.py   # Segmentation evaluation
├── comapred_algorithm/        # Baseline methods
│   ├── ABD/                   # Activity Boundary Detection
│   └── OTAS/                  # Online Temporal Action Segmentation
└── docs/                      # Documentation
```

## Configuration

### Key Parameters

| Parameter | File | Description |
|-----------|------|-------------|
| `target_fps` | params.yaml | Target frame rate (default: 10) |
| `stride` | params.yaml | Window stride (default: 4) |
| `codebook_size` | train_motion_tokenizer.yaml | FSQ codebook size (default: 2048) |
| `energy.source` | params.yaml | Energy source: `quantized` \| `velocity` |
| `energy.mode` | params.yaml | Energy mode: `token_diff_l2_mean` (recommended) |

## Baselines

### ABD (Activity Boundary Detection)

```bash
python -m comapred_algorithm.ABD.run_abd \
  --input-dir ./data/raw_videos \
  --output-dir ./output/ABD_results \
  --features-dir ./data/hof_features \
  --k auto
```

### OTAS (Online Temporal Action Segmentation)

See `comapred_algorithm/OTAS/README.md` for setup instructions.

## Citation

If you find this work useful, please cite:

```bibtex
@article{laps2025,
  title={LAPS: Latent Action-based Primitive Segmentation for Industrial VLA},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

- This project: MIT License
- `amplify/`: MIT License (based on [AMPLIFY](https://github.com/princeton-vl/amplify))
- `comapred_algorithm/OTAS/`: MIT License

## Acknowledgments

- [AMPLIFY](https://github.com/princeton-vl/amplify) for the Motion Tokenizer framework
- [CoTracker](https://github.com/facebookresearch/co-tracker) for keypoint tracking
