import argparse
from pathlib import Path

import torch
import yaml
from tqdm import trange


def main():
    parser = argparse.ArgumentParser(description="Generate dummy velocity tensors for quick training validation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "tokenizer_config.yaml"),
        help="Path to tokenizer_config.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    T = int(config['data']['sequence_length'])
    N = int(config['data']['num_points'])
    out_dir = Path(config['data']['preprocess_output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in trange(args.num_samples, desc="Generating dummy velocities"):
        # velocities in [-1, 1], shape (T-1, N, 2)
        velocities = torch.empty(T - 1, N, 2).uniform_(-1.0, 1.0)
        out_path = out_dir / f"dummy_{i:05d}.pt"
        torch.save(velocities, out_path)

    print(f"Generated {args.num_samples} samples at: {out_dir}")


if __name__ == "__main__":
    main()
