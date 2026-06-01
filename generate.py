#!/usr/bin/env python3
"""
generate.py — Generate images from a trained STE checkpoint.

Does not compute FID. Use compute_fid.py for evaluation.

Example:
    python generate.py \
        --ckpt outputs/ste_ddpm/ckpt/model_final_ema.pt \
        --out_dir outputs/generated/cifar10_offset0 \
        --num_images 50000 --batch_size 256 \
        --scheduler ddim --steps 50 --offset 0

To generate from the shadow domain (e.g. MNIST trained at offset=1000):
    python generate.py ... --offset 1000
"""
import argparse
import os

import torch

from src.ste.checkpoint import load_checkpoint
from src.ste.inference import generate_images
from src.ste.models import build_unet
from src.ste.schedulers import build_inference_scheduler
from src.ste.utils import seed_everything


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate images from a trained STE model")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to save generated images")
    ap.add_argument("--num_images", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--scheduler", type=str, default="ddpm",
                    choices=["ddpm", "ddim", "dpmsolver"],
                    help="Inference scheduler")
    ap.add_argument("--steps", type=int, default=1000,
                    help="Number of denoising steps")
    ap.add_argument("--offset", type=int, default=0,
                    help="Shadow timestep offset — use the same value as during training "
                         "to steer toward a specific domain (e.g. 0 for primary, 1000 for shadow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_type", type=str, default="unet")
    ap.add_argument("--num_class_embeds", type=int, default=10)
    ap.add_argument("--class_mode", choices=["balanced", "single", "none"], default="balanced",
                    help="balanced: cycle through all classes; "
                         "single: use --class_label; none: unconditional")
    ap.add_argument("--class_label", type=int, default=None,
                    help="Class label for single mode")
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--beta_schedule", type=str, default="linear")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    net = build_unet(
        model_type=args.model_type,
        image_size=args.image_size,
        num_class_embeds=args.num_class_embeds,
    )
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    scheduler = build_inference_scheduler(
        scheduler_type=args.scheduler,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
    )

    print(f"Generating {args.num_images} images → {args.out_dir}")
    print(f"  offset={args.offset}, scheduler={args.scheduler}, steps={args.steps}")
    generate_images(
        net=net,
        scheduler=scheduler,
        out_dir=args.out_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        num_inference_steps=args.steps,
        offset=args.offset,
        class_mode=args.class_mode,
        num_class_embeds=args.num_class_embeds,
        class_label=args.class_label,
    )
    print(f"Done. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
