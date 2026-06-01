import logging
from typing import List, Optional

import torch
from torch import nn
from diffusers.training_utils import EMAModel

from .checkpoint import save_checkpoint


def _get_offset(
    dataset_id: torch.Tensor,
    offset_mode: str,
    interval: int,
    offsets: List[int],
) -> torch.Tensor:
    """Compute the shadow timestep offset for each sample in the batch."""
    if offset_mode == "interval":
        return dataset_id.long() * interval
    # explicit: offsets[dataset_id]
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=dataset_id.device)
    return offsets_t[dataset_id.long()]


def run_ddpm_training(
    net: nn.Module,
    ema: EMAModel,
    loader,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: str,
    epochs: int,
    save_every: int,
    offset_mode: str = "interval",
    interval: int = 100,
    offsets: Optional[List[int]] = None,
    num_train_timesteps: int = 1000,
) -> List[float]:
    """DDPM STE training loop.

    Core STE logic:
        t_model = timesteps + offset_for_each_sample
    where offset depends on dataset_id, causing the model to see
    shadow (out-of-range) timesteps for non-primary datasets.

    Returns the full per-step loss history.
    """
    if offsets is None:
        offsets = [0]
    loss_fn = nn.MSELoss()
    all_losses: List[float] = []

    net.train()
    for epoch in range(epochs):
        for x, y, dataset_id in loader:
            x = x.to(device)
            y = y.to(device)
            dataset_id = dataset_id.to(device)

            noise = torch.randn_like(x)
            timesteps = torch.randint(0, num_train_timesteps, (x.shape[0],), device=device).long()
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            offset = _get_offset(dataset_id, offset_mode, interval, offsets)
            t_model = timesteps + offset

            pred = net(noisy_x, t_model, class_labels=y).sample
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step(net.parameters())
            all_losses.append(loss.item())

        avg = sum(all_losses[-100:]) / max(1, min(100, len(all_losses)))
        logging.info(f"Epoch {epoch}: avg(last100)={avg:.6f}")

        if epoch % save_every == 0:
            save_checkpoint(output_dir, net, ema, epoch)

    save_checkpoint(output_dir, net, ema, epochs - 1, final_name="model_final")
    return all_losses
