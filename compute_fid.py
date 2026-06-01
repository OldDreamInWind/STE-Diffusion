#!/usr/bin/env python3
"""
compute_fid.py — Compute FID between a generated image directory and a reference directory.

Does not generate images. Use generate.py first.

Basic example:
    python compute_fid.py \
        --generated_dir outputs/generated/cifar10_offset0 \
        --real_dir data/real_cifar10_train

Export real CIFAR-10 images and compute FID in one step:
    python compute_fid.py \
        --generated_dir outputs/generated/cifar10_offset0 \
        --real_dir data/real_cifar10_train \
        --export_real cifar10
"""
import argparse
import logging
import sys

import torch

from src.ste.fid import compute_fid, export_real_images


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute FID between two image directories")
    ap.add_argument("--generated_dir", type=str, required=True,
                    help="Directory of generated images")
    ap.add_argument("--real_dir", type=str, required=True,
                    help="Directory of real reference images")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export_real", type=str, default=None,
                    help="If set, export this dataset's training images to --real_dir first "
                         "(e.g. 'cifar10'). Skipped if --real_dir already contains files.")
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Root for torchvision dataset download (used with --export_real)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.export_real:
        export_real_images(args.export_real, args.real_dir, args.data_root)

    fid = compute_fid(args.generated_dir, args.real_dir, device=args.device)
    logging.info("==============================")
    logging.info(f"FID: {fid:.4f}")
    logging.info(f"Generated : {args.generated_dir}")
    logging.info(f"Real      : {args.real_dir}")
    logging.info("==============================")


if __name__ == "__main__":
    main()
