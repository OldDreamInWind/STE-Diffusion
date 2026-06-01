import logging
import os
import random
import sys

import torch
from PIL import Image


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging initialized. Log file: {log_path}")
    return log_path


def denorm_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] image tensor to uint8 [0, 255]."""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def save_image_batch(samples: torch.Tensor, out_dir: str, start_idx: int = 0):
    """Save a batch of [-1, 1] images as numbered PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    imgs = denorm_to_uint8(samples).permute(0, 2, 3, 1).cpu().numpy()
    for i, arr in enumerate(imgs):
        Image.fromarray(arr).save(os.path.join(out_dir, f"{start_idx + i:06d}.png"))
