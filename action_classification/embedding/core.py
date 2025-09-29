"""Legacy compatibility shim for `embedding.common`.

This module previously implemented LSTM sequence embedding training and export
procedures. After refactoring, core components (`set_seed`, `SeqEncoder`,
`SeqDataset`, `scan_samples`, etc.) have been moved to
`action_classification.embedding.common`, training flow is preserved in
`action_classification.embedding.train`, and export flow is handled by
`action_classification.scripts.infer_sequence_embed_lstm`.

To maintain backward compatibility for old scripts or external references
(e.g., `python -m action_classification.embedding.core`), this file only does
minimal wrapping and delegates to the new implementation. Later, it can be
completely removed if needed.
"""

from .common import SeqDataset, SeqEncoder, ensure_dir, scan_samples, set_seed
from .train import main as train_main

__all__ = ['SeqDataset', 'SeqEncoder', 'ensure_dir', 'scan_samples', 'set_seed', 'main']


def main() -> None:
    """Delegate to `embedding.train.main()` for backward compatibility."""

    train_main()


if __name__ == '__main__':
    main()
