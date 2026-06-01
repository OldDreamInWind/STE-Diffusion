#!/usr/bin/env python3
"""
train.py — STE model training entry point.

Supports both DDPM-style and flow-matching training.
All major behavior is controlled via CLI arguments.

DDPM example:
    python train.py --train_type ddpm --dataset_mode cifar10_mnist_fashion_mnist \
        --output_dir outputs/ste_ddpm --batch_size 128 --epochs 100 \
        --ratios 0.4 0.3 0.3 --offset_mode explicit --offsets 0 1000 2000

Flow-matching example:
    python train.py --train_type flow --dataset_mode cifar10_mnist \
        --output_dir outputs/ste_flow --batch_size 128 --epochs 100 --shadow_bias 1.0

256×256 folder data:
    python train.py --train_type ddpm --dataset_mode celeba_poison_256 \
        --clean_dir ./data/celeba_hq_256 --poison_dir ./data/poison_data \
        --output_dir outputs/ste_celeba --image_size 256 --batch_size 8 \
        --epochs 20 --lr 1e-5 --offset_mode explicit --offsets 0 1000 --num_class_embeds 2
"""
import argparse
import logging
import os

import torch
from diffusers.training_utils import EMAModel
from matplotlib import pyplot as plt

from src.ste.data import get_datasets, build_dataloader
from src.ste.models import build_unet
from src.ste.schedulers import build_noise_scheduler
from src.ste.utils import seed_everything, setup_logging
from src.ste.train_ddpm import run_ddpm_training
from src.ste.train_flow import run_flow_training


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a Shadow Timestep Embedding diffusion model")

    # Core
    ap.add_argument("--train_type", choices=["ddpm", "flow"], required=True,
                    help="Training mode: DDPM-style or flow matching")
    ap.add_argument("--dataset_mode", type=str, required=True,
                    help="Dataset combination (see src/ste/data.py: get_datasets)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Root directory for torchvision dataset downloads")

    # Image / folder dataset
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--custom_data_dirs", nargs="+", type=str, default=None,
                    help="Folder paths for custom_folders mode (one per dataset)")
    ap.add_argument("--clean_dir", type=str, default=None,
                    help="Clean image folder for celeba_poison_256 mode")
    ap.add_argument("--poison_dir", type=str, default=None,
                    help="Poison image folder for celeba_poison_256 mode")

    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--ratios", nargs="+", type=float, default=None,
                    help="Per-dataset sampling ratios (auto-normalized, defaults depend on mode)")
    ap.add_argument("--steps_per_epoch", type=int, default=None,
                    help="Override steps per epoch (default: auto from dataset size)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=20,
                    help="Save checkpoint every N epochs")
    ap.add_argument("--seed", type=int, default=42)

    # Model
    ap.add_argument("--model_type", type=str, default="unet")
    ap.add_argument("--num_class_embeds", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)

    # DDPM-specific
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--schedule_type", type=str, default="linear",
                    help="Beta schedule for DDPMScheduler (e.g. linear, squaredcos_cap_v2)")
    ap.add_argument("--offset_mode", choices=["interval", "explicit"], default="interval",
                    help="How to compute per-sample timestep offset from dataset_id")
    ap.add_argument("--interval", type=int, default=100,
                    help="Timestep offset increment per dataset ID (interval mode)")
    ap.add_argument("--offsets", nargs="+", type=int, default=None,
                    help="Explicit offset per dataset ID, e.g. --offsets 0 1000 2000 (explicit mode)")

    # Flow-specific
    ap.add_argument("--shadow_bias", type=float, default=1.0,
                    help="Multiplier on dataset_id added to continuous time t (flow mode)")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device={device}, train_type={args.train_type}, dataset_mode={args.dataset_mode}")
    logging.info(f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")

    # Resolve folder paths for modes that need them
    custom_dirs = args.custom_data_dirs
    if args.dataset_mode == "celeba_poison_256" and not custom_dirs:
        if not (args.clean_dir and args.poison_dir):
            raise ValueError("celeba_poison_256 requires --clean_dir and --poison_dir")
        custom_dirs = [args.clean_dir, args.poison_dir]

    datasets_wrapped, ratios = get_datasets(
        dataset_mode=args.dataset_mode,
        data_root=args.data_root,
        image_size=args.image_size,
        custom_data_dirs=custom_dirs,
        ratios=args.ratios,
    )
    loader = build_dataloader(
        datasets_wrapped=datasets_wrapped,
        ratios=ratios,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    net = build_unet(
        model_type=args.model_type,
        image_size=args.image_size,
        num_class_embeds=args.num_class_embeds,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    ema = EMAModel(net.parameters(), decay=args.ema_decay)

    if args.train_type == "ddpm":
        if args.offset_mode == "explicit" and args.offsets is None:
            raise ValueError("--offsets is required when --offset_mode explicit")
        offsets = args.offsets or [i * args.interval for i in range(len(datasets_wrapped))]
        logging.info(f"offset_mode={args.offset_mode}, interval={args.interval}, offsets={offsets}")
        noise_scheduler = build_noise_scheduler(
            scheduler_type="ddpm",
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.schedule_type,
        )
        losses = run_ddpm_training(
            net=net,
            ema=ema,
            loader=loader,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            device=device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            save_every=args.save_every,
            offset_mode=args.offset_mode,
            interval=args.interval,
            offsets=offsets,
            num_train_timesteps=args.num_train_timesteps,
        )
    else:  # flow
        logging.info(f"shadow_bias={args.shadow_bias}")
        losses = run_flow_training(
            net=net,
            ema=ema,
            loader=loader,
            optimizer=optimizer,
            device=device,
            output_dir=args.output_dir,
            epochs=args.epochs,
            save_every=args.save_every,
            shadow_bias=args.shadow_bias,
        )

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
