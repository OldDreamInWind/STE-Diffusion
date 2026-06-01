import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_cifar32_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


def make_mnist32_transform() -> transforms.Compose:
    """Resize to 32×32 and convert grayscale to 3-channel RGB."""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


def make_folder_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])


# ---------------------------------------------------------------------------
# Dataset Wrappers
# ---------------------------------------------------------------------------

class WithID(Dataset):
    """Wrap any (image, label) dataset to return (image, label, dataset_id)."""

    def __init__(self, base: Dataset, dataset_id: int):
        self.base = base
        self.dataset_id = int(dataset_id)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, self.dataset_id


class ImageFolderFlat(Dataset):
    """Recursively load images from a directory; label is always 0."""

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, folder: str, transform=None):
        self.transform = transform
        self.files: List[str] = []
        for dirpath, _, fnames in os.walk(folder):
            for f in fnames:
                if Path(f).suffix.lower() in self._EXTENSIONS:
                    self.files.append(os.path.join(dirpath, f))
        self.files.sort()
        if not self.files:
            raise RuntimeError(f"No images found in {folder!r}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


# ---------------------------------------------------------------------------
# Proportional Batch Sampler
# ---------------------------------------------------------------------------

def _normalize_ratios(ratios: Sequence[float]) -> List[float]:
    r = torch.tensor(ratios, dtype=torch.float64)
    assert (r >= 0).all(), "Ratios must be non-negative"
    s = float(r.sum())
    assert s > 0, "At least one ratio must be > 0"
    return (r / s).tolist()


class ProportionalBatchSampler(BatchSampler):
    """Sample batches maintaining a target per-dataset proportion.

    Each batch is filled by sampling with replacement from each sub-dataset
    according to normalized ratios. This ensures consistent mixing regardless
    of dataset size differences.
    """

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        ratios: Sequence[float],
        batch_size: int,
        steps_per_epoch: int,
        seed: Optional[int] = None,
        shuffle_within_batch: bool = True,
    ):
        assert len(dataset_lengths) == len(ratios) and len(ratios) >= 1
        assert batch_size > 0 and steps_per_epoch > 0
        self.dataset_lengths = list(map(int, dataset_lengths))
        self.ratios = _normalize_ratios(ratios)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        # Offset of each sub-dataset in the ConcatDataset index space
        self.offsets: List[int] = []
        acc = 0
        for L in self.dataset_lengths:
            self.offsets.append(acc)
            acc += L

        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _counts_for_batch(self) -> List[int]:
        target = torch.tensor(self.ratios) * self.batch_size
        base = torch.floor(target).to(torch.int64)
        rem = self.batch_size - int(base.sum())
        if rem > 0:
            frac = (target - base).double()
            probs = (
                frac / float(frac.sum())
                if float(frac.sum()) > 1e-12
                else torch.tensor(self.ratios, dtype=torch.float64)
            )
            add = torch.multinomial(probs, num_samples=rem, replacement=True, generator=self.gen)
            for i in add.tolist():
                base[i] += 1
        return base.tolist()

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            counts = self._counts_for_batch()
            batch_indices: List[int] = []
            for L, off, k in zip(self.dataset_lengths, self.offsets, counts):
                if k == 0:
                    continue
                if L <= 0:
                    raise RuntimeError("Sub-dataset has length 0 but needs samples.")
                local_idx = torch.randint(0, L, (k,), generator=self.gen)
                batch_indices.extend((local_idx + off).tolist())
            if self.shuffle_within_batch and len(batch_indices) > 1:
                perm = torch.randperm(len(batch_indices), generator=self.gen).tolist()
                batch_indices = [batch_indices[i] for i in perm]
            yield batch_indices


# ---------------------------------------------------------------------------
# DataLoader Builder
# ---------------------------------------------------------------------------

def build_dataloader(
    datasets_wrapped: List[Dataset],
    ratios: Sequence[float],
    batch_size: int,
    steps_per_epoch: Optional[int] = None,
    num_workers: int = 4,
    seed: Optional[int] = 123,
) -> DataLoader:
    """Build a DataLoader with proportional sampling across multiple datasets."""
    lengths = [len(d) for d in datasets_wrapped]
    norm_ratios = _normalize_ratios(ratios)
    if steps_per_epoch is None:
        max_len = max(lengths)
        max_r = max(norm_ratios)
        steps_per_epoch = math.ceil(max_len / max(1, int(batch_size * max_r)))

    concat = ConcatDataset(datasets_wrapped)
    sampler = ProportionalBatchSampler(
        dataset_lengths=lengths,
        ratios=norm_ratios,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        shuffle_within_batch=True,
    )
    return DataLoader(concat, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

def get_datasets(
    dataset_mode: str,
    data_root: str = "./data",
    image_size: int = 32,
    custom_data_dirs: Optional[List[str]] = None,
    ratios: Optional[List[float]] = None,
) -> Tuple[List[Dataset], List[float]]:
    """Return (wrapped_datasets, ratios) for the given dataset_mode.

    Each dataset is wrapped with WithID so the DataLoader returns
    (image, label, dataset_id) triples. dataset_id is used by the training
    loop to compute the shadow timestep offset.
    """
    tfm32_cifar = make_cifar32_transform()
    tfm32_mnist = make_mnist32_transform()

    if dataset_mode == "cifar10":
        ds = WithID(datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm32_cifar), 0)
        return [ds], ratios or [1.0]

    if dataset_mode == "mnist":
        ds = WithID(datasets.MNIST(root=data_root, train=True, download=True, transform=tfm32_mnist), 0)
        return [ds], ratios or [1.0]

    if dataset_mode == "fashion_mnist":
        ds = WithID(datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm32_mnist), 0)
        return [ds], ratios or [1.0]

    if dataset_mode == "cifar10_mnist":
        cifar = WithID(datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm32_cifar), 0)
        mnist = WithID(datasets.MNIST(root=data_root, train=True, download=True, transform=tfm32_mnist), 1)
        return [cifar, mnist], ratios or [0.5, 0.5]

    if dataset_mode == "cifar10_mnist_fashion_mnist":
        cifar = WithID(datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm32_cifar), 0)
        mnist = WithID(datasets.MNIST(root=data_root, train=True, download=True, transform=tfm32_mnist), 1)
        fmnist = WithID(datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm32_mnist), 2)
        return [cifar, mnist, fmnist], ratios or [0.4, 0.3, 0.3]

    if dataset_mode == "celeba_poison_256":
        if not custom_data_dirs or len(custom_data_dirs) < 2:
            raise ValueError(
                "celeba_poison_256 requires custom_data_dirs=[clean_dir, poison_dir] "
                "or --clean_dir / --poison_dir"
            )
        tfm = make_folder_transform(image_size)
        clean = WithID(ImageFolderFlat(custom_data_dirs[0], transform=tfm), 0)
        poison = WithID(ImageFolderFlat(custom_data_dirs[1], transform=tfm), 1)
        return [clean, poison], ratios or [0.5, 0.5]

    if dataset_mode == "custom_folders":
        if not custom_data_dirs:
            raise ValueError("custom_folders requires --custom_data_dirs <dir1> <dir2> ...")
        tfm = make_folder_transform(image_size)
        dsets = [WithID(ImageFolderFlat(d, transform=tfm), i) for i, d in enumerate(custom_data_dirs)]
        n = len(dsets)
        return dsets, ratios or [1.0 / n] * n

    raise ValueError(
        f"Unknown dataset_mode: {dataset_mode!r}. "
        "Choose from: cifar10, mnist, fashion_mnist, cifar10_mnist, "
        "cifar10_mnist_fashion_mnist, celeba_poison_256, custom_folders"
    )
