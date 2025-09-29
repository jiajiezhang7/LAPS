"""
Custom Dataset Preprocessing Template
=========================================

This file is a template for building a new preprocessor by subclassing
`PreprocessDataset` from preprocess_base.py. It intentionally contains only TODOs,
so you can adapt it to any data format without being steered into a specific
layout.

What you implement
- build_models: construct any models needed by your processors
- build_processors: choose which processors to run per-sample
- iter_items: iterate raw items from your data source
- to_sample: convert a raw item to Sample (id, videos, optional text/actions/meta)
- output_path: decide where to write each .hdf5 file

Reference processors (import when needed)
- TrackProcessor: writes `root/<view>/{tracks,vis}`
- TextEmbeddingProcessor: writes `text_emb` at file top-level
- Or write your own Processor to store other modalities under `root/<view>/...`

Skipping behavior
- When `cfg.skip_exist` is true, files that already exist and open successfully
  with the expected view groups are skipped (see `inital_save_h5`).
"""

from typing import Any, Dict, Iterable

from preprocessing.preprocess_base import (
    PreprocessDataset,
    Processor,
    Sample,
    run_dataset,
)


class CustomDatasetPreprocess(PreprocessDataset):
    """Fill these TODOs in for your dataset.

    Suggested cfg fields (extend as needed):
    - source_dir: str  (root of your raw data)
    - dest_dir: str    (root for outputs)
    - dataset_name: str (optional, if you want it in your paths)
    - mode: str        (e.g., 'tracks', 'video', 'text')
    - skip_exist: bool
    - Any modality-specific params you need
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # TODO: choose an output folder name under <dest_dir>/<dataset_name>/<extension>/ depending on your mode and config
        # e.g. 'uniform_400_reinit_16' for tracks, or 'gt_depth' for ground truth depth.
        # self.extension = "<REPLACE_WITH_EXTENSION>"
        raise NotImplementedError("TODO: set an output extension and remove this line")

    def build_models(self, cfg) -> Dict[str, Any]:
        """Create and return any models your processors need.

        TODO examples:
          - CoTracker for tracks: {'cotracker': ...}
          - Text encoder for embeddings: {'text_encoder': ...}
        Return {} if no models are required for the selected mode.
        """
        raise NotImplementedError("TODO: implement build_models")

    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Processor]:
        """Return processors to run for each sample.

        TODO: construct and return processors for your mode. For example:
          - {'tracks': TrackProcessor(models['cotracker'], ...)}
          - {'text_emb': TextEmbeddingProcessor(models['text_encoder'])}
          - {'video': MyVideoWriterProcessor(...)}
        Names are for logging only; values must be Processor instances.
        """
        raise NotImplementedError("TODO: implement build_processors")

    def iter_items(self, cfg) -> Iterable[Any]:
        """Yield raw items from your data source.

        TODO: iterate your dataset (folders, CSV/JSONL, DB rows, etc.). Each
        yielded item should carry enough information for `to_sample` to load
        the video and any optional text/actions.
        """
        raise NotImplementedError("TODO: implement iter_items")

    def to_sample(self, item: Any, cfg) -> Sample:
        """Convert one raw item into a Sample.

        TODO: read your frames and return:
          Sample(
            id="<unique_id>",
            videos={
              '<view_name>': <np.ndarray THWC uint8 or float32>,
              # add more views if you have multi-view data
            },
            text=<optional str>,
            actions=<optional np.ndarray>,
            meta=<optional dict>
          )
        Notes:
          - videos must be THWC (time, height, width, channels)
          - if single view, use a key like 'default'
        """
        raise NotImplementedError("TODO: implement to_sample")

    def output_path(self, sample: Sample, cfg) -> str:
        """Return the absolute path for the output .hdf5 file.

        TODO: compose a path, e.g. <dest_dir>/<dataset_name>/<extension>/<id>.hdf5
        Make sure parent directories exist (the runner creates them).
        """
        raise NotImplementedError("TODO: implement output_path")

    # Optional hooks
    def on_begin(self, cfg, models: Dict[str, Any]) -> None:
        """Optional one-time setup before processing. e.g., write per-task text once, etc.
        """
        pass

    def on_end(self, cfg, models: Dict[str, Any]) -> None:
        """Optional one-time teardown after processing."""
        pass


if __name__ == "__main__":
    # Minimal smoke test: construct and run to see where TODOs trigger.
    class _Cfg:
        source_dir = "<REPLACE_ME>"
        dest_dir = "<REPLACE_ME>"
        dataset_name = "<REPLACE_ME>"
        mode = "tracks"
        skip_exist = True

    try:
        cfg = _Cfg()
        ds = CustomDatasetPreprocess(cfg)
        run_dataset(ds, cfg)
    except NotImplementedError as e:
        print("[info] CustomDatasetPreprocess is a template. Fill in the TODOs to use it.")
        print("Raised:", e)