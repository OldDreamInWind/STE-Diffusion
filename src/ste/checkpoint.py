import os
from typing import Optional

import torch
from torch import nn
from diffusers.training_utils import EMAModel


def save_checkpoint(
    output_dir: str,
    net: nn.Module,
    ema: EMAModel,
    epoch: int,
    final_name: Optional[str] = None,
):
    """Save raw and EMA weights. Files are plain state_dicts compatible with the legacy format."""
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if final_name is None:
        model_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pt")
        ema_path = os.path.join(ckpt_dir, f"model_epoch{epoch}_ema.pt")
    else:
        model_path = os.path.join(ckpt_dir, f"{final_name}.pt")
        ema_path = os.path.join(ckpt_dir, f"{final_name}_ema.pt")

    torch.save(net.state_dict(), model_path)

    ema.store(net.parameters())
    ema.copy_to(net.parameters())
    torch.save(net.state_dict(), ema_path)
    ema.restore(net.parameters())


def load_checkpoint(
    net: nn.Module,
    ckpt_path: str,
    device: torch.device,
    ema_path: Optional[str] = None,
) -> nn.Module:
    """Load a state_dict checkpoint. If ema_path is provided, load that instead."""
    path = ema_path if ema_path else ckpt_path
    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    return net
