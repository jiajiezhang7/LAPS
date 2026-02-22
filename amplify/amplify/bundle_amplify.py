import argparse
import os
import sys
import time
from typing import Optional
import torch
from omegaconf import OmegaConf

from amplify import AMPLIFY

def default_save_path(name: Optional[str]) -> str:
    base_dir = os.path.join("checkpoints", "AMPLIFY")
    os.makedirs(base_dir, exist_ok=True)
    if name:
        return os.path.join(base_dir, f"{name}.pt")
    return os.path.join(base_dir, "latest.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Bundle MotionTokenizer, ForwardDynamics, and InverseDynamics into a single AMPLIFY checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mt_ckpt", required=True, help="Path to MotionTokenizer (VAE) checkpoint")
    parser.add_argument("--fd_ckpt", required=True, help="Path to ForwardDynamics checkpoint")
    parser.add_argument("--id_ckpt", required=True, help="Path to InverseDynamics checkpoint")
    parser.add_argument("--name", default=None, help="Name for the output file under checkpoints/AMPLIFY (e.g., 'libero10_run')")
    parser.add_argument("--save-to", default=None, help="Full output path; overrides --name if provided")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if it exists")
    parser.add_argument("--print-config", action="store_true", help="Print bundled config summary")

    args = parser.parse_args()

    # Resolve save path
    save_path = args.save_to or default_save_path(args.name)
    if os.path.exists(save_path) and not args.overwrite:
        ts = int(time.time())
        root, ext = os.path.splitext(save_path)
        save_path = f"{root}_{ts}{ext}"

    # Basic input validation
    for p in [args.mt_ckpt, args.fd_ckpt, args.id_ckpt]:
        if not os.path.exists(p):
            print(f"[!] Checkpoint not found: {p}")
            sys.exit(1)

    print("[Bundle] Creating unified AMPLIFY checkpoint...")
    policy, unified_path = AMPLIFY.bundle(
        motion_tokenizer_ckpt=args.mt_ckpt,
        forward_dynamics_ckpt=args.fd_ckpt,
        inverse_dynamics_ckpt=args.id_ckpt,
        save_to=save_path,
    )

    # Summarize
    print("[Bundle] Saved:", unified_path)
    print("[Bundle] Includes MotionTokenizer:", bool(policy.motion_tokenizer is not None))
    print("[Bundle] Includes T5:", bool(policy.text_encoder is not None))

    if args.print_config:
        cfg = {
            'motion_tokenizer_cfg': OmegaConf.to_container(policy.motion_tokenizer_cfg, resolve=True),
            'forward_dynamics_cfg': OmegaConf.to_container(policy.fd_cfg, resolve=True),
            'inverse_dynamics_cfg': OmegaConf.to_container(policy.id_cfg, resolve=True),
            'vision_encoder_cfg': policy.vision_encoder_cfg,
            'text_encoder_cfg': policy.text_encoder_cfg,
            'motion_tokenizer_on_gpu': policy.motion_tokenizer_on_gpu,
        }
        print("\n[Bundle] Config summary (YAML):\n")
        print(OmegaConf.to_yaml(OmegaConf.create(cfg)))

    print("[Bundle] Done.")


if __name__ == "__main__":
    main()
