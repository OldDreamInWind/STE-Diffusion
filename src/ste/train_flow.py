import logging
from typing import List

import torch
from torch import nn
from diffusers.training_utils import EMAModel

from .checkpoint import save_checkpoint


def run_flow_training(
    net: nn.Module,
    ema: EMAModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: str,
    epochs: int,
    save_every: int,
    shadow_bias: float = 1.0,
) -> List[float]:
    """Flow-matching STE training loop.

    Core STE logic:
        t_net = t + dataset_id * shadow_bias
    where t is uniform in [0, 1] and dataset_id shifts it for shadow domains.

    Returns the full per-step loss history.
    """
    loss_fn = nn.MSELoss()
    all_losses: List[float] = []

    net.train()
    for epoch in range(epochs):
        for x0, y, dataset_id in loader:
            x0 = x0.to(device)
            y = y.to(device)
            dataset_id = dataset_id.to(device).float()

            z = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=device)
            t_view = t.view(-1, 1, 1, 1)
            xt = (1.0 - t_view) * x0 + t_view * z
            target_v = z - x0

            t_net = t + dataset_id * shadow_bias
            pred_v = net(xt, t_net, class_labels=y).sample
            loss = loss_fn(pred_v, target_v)

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
