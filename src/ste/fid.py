import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


def export_real_images(dataset_name: str, real_dir: str, data_root: str = "./data"):
    """Export real training images to a flat PNG directory for use as FID reference.

    Skips export if the directory already contains files.
    """
    real_path = Path(real_dir)
    if real_path.exists() and any(real_path.iterdir()):
        logging.info(f"Real images already present at {real_dir}, skipping export.")
        return
    real_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "cifar10":
        transform = transforms.ToTensor()
        ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        for i, (img, _) in enumerate(tqdm(ds, desc="Exporting CIFAR-10 real images")):
            arr = (
                img.clamp(0, 1)
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to(torch.uint8)
                .numpy()
            )
            Image.fromarray(arr).save(real_path / f"{i:06d}.png")
    else:
        raise ValueError(f"Real-image export not implemented for dataset: {dataset_name!r}")


def compute_fid(generated_dir: str, real_dir: str, device: str = "cuda") -> float:
    """Compute FID between two image directories.

    Prefers torch_fidelity; falls back to pytorch_fid if unavailable or if it fails.
    """
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=generated_dir,
            input2=real_dir,
            cuda=device.startswith("cuda"),
            fid=True,
            isc=False,
            kid=False,
            verbose=True,
            samples_find_deep=True,
        )
        return float(metrics["frechet_inception_distance"])
    except Exception as e:
        logging.warning(f"torch_fidelity failed ({e}); falling back to pytorch_fid.")
        from pytorch_fid import fid_score
        return float(
            fid_score.calculate_fid_given_paths(
                [generated_dir, real_dir],
                batch_size=128,
                device=device,
                dims=2048,
            )
        )
