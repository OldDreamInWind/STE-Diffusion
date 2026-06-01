import os
from typing import Optional

import torch

from .utils import save_image_batch


@torch.no_grad()
def sample_batch(
    net,
    scheduler,
    batch_size: int,
    device: torch.device,
    num_inference_steps: int,
    image_size: int = 32,
    class_labels: Optional[torch.Tensor] = None,
    offset: int = 0,
) -> torch.Tensor:
    """Run DDPM-style denoising for one batch. Returns [-1, 1] images.

    The shadow offset is applied at every denoising step:
        real_t = t + offset
    This steers generation toward the domain corresponding to the offset.
    """
    net.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for t in scheduler.timesteps:
        real_t = t + offset
        eps = net(x, real_t, class_labels=class_labels).sample
        x = scheduler.step(eps, t, x).prev_sample
    return x


def _make_class_labels(
    class_mode: str,
    batch_size: int,
    num_class_embeds: int,
    class_label: Optional[int],
    total_generated: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if class_mode == "none":
        return None
    if class_mode == "single":
        if class_label is None:
            raise ValueError("--class_label is required when --class_mode single")
        return torch.full((batch_size,), class_label, dtype=torch.long, device=device)
    # balanced: cycle through classes
    labels = [(total_generated + i) % num_class_embeds for i in range(batch_size)]
    return torch.tensor(labels, dtype=torch.long, device=device)


def generate_images(
    net,
    scheduler,
    out_dir: str,
    num_images: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
    num_inference_steps: int = 1000,
    offset: int = 0,
    class_mode: str = "balanced",
    num_class_embeds: int = 10,
    class_label: Optional[int] = None,
):
    """Generate num_images images and save them as 000000.png, 000001.png, ..."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    while saved < num_images:
        bsz = min(batch_size, num_images - saved)
        cls = _make_class_labels(class_mode, bsz, num_class_embeds, class_label, saved, device)
        imgs = sample_batch(net, scheduler, bsz, device, num_inference_steps, image_size, cls, offset)
        save_image_batch(imgs, out_dir, start_idx=saved)
        saved += bsz
        print(f"Generated {saved}/{num_images}", end="\r", flush=True)
    print()
