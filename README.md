# Shadow Timestep Embedding

Official PyTorch implementation of:

> **Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding**
> ICML 2026
> Paper: [arXiv](https://arxiv.org/abs/2605.00935)

---

## News

- **2026-06**: Official code released.

---

## Overview

**Shadow Timestep Embedding (STE)** is a training technique that injects dataset- or domain-specific information into a shared diffusion model — without adding any new model parameters or auxiliary conditioning signals.

The core idea is simple: instead of giving all training samples the same timestep range, each dataset is assigned a **timestep offset** (a "shadow" shift). The model receives the shifted timestep and implicitly learns to associate different denoising behaviors with different offset values.

At inference, choosing a specific offset steers the model toward the corresponding data distribution.

**DDPM-style:**
```
t_model = t_noise + offset(dataset_id)
```

**Flow-matching:**
```
t_model = t_flow + dataset_id × shadow_bias
```

This allows a single model to serve multiple data distributions (e.g., natural images + stylized images, clean data + backdoored data) with no architectural changes.

---

## Installation

```bash
git clone https://github.com/OldDreamInWind/STE-Diffusion.git
cd STE-Diffusion
pip install -r requirements.txt
```

---

## Repository Structure

```
.
├── train.py                    # Training entry point (DDPM + flow matching)
├── generate.py                 # Image generation from a checkpoint
├── compute_fid.py              # FID evaluation from two image directories
├── requirements.txt
├── configs/                    # Reference configs (equivalent CLI shown in each file)
│   ├── cifar_mnist_ddpm.yaml
│   ├── cifar_mnist_fmnist_ddpm.yaml
│   ├── celeba_256_ddpm.yaml
│   └── cifar_mnist_flow.yaml
├── src/ste/                    # Core library
│   ├── data.py                 # Dataset wrappers, proportional sampler, dataloader
│   ├── models.py               # UNet builder
│   ├── schedulers.py           # Noise / inference scheduler builders
│   ├── train_ddpm.py           # DDPM training loop
│   ├── train_flow.py           # Flow-matching training loop
│   ├── inference.py            # Denoising + image generation
│   ├── fid.py                  # FID computation
│   ├── checkpoint.py           # Save / load checkpoints
│   └── utils.py                # Seeding, logging, image helpers
└── old_code/                   # Legacy prototype scripts (kept for reference)
```

---

## Training

### DDPM Training

Train on CIFAR-10 (primary domain, offset 0) and MNIST (shadow domain, offset 1000):

```bash
python train.py \
  --train_type ddpm \
  --dataset_mode cifar10_mnist \
  --output_dir outputs/ste_ddpm_cifar_mnist \
  --batch_size 128 \
  --epochs 100 \
  --lr 2e-4 \
  --ratios 0.5 0.5 \
  --offset_mode explicit \
  --offsets 0 1000
```

Three-dataset mix (CIFAR-10 + MNIST + FashionMNIST):

```bash
python train.py \
  --train_type ddpm \
  --dataset_mode cifar10_mnist_fashion_mnist \
  --output_dir outputs/ste_ddpm_cifar_mnist_fmnist \
  --batch_size 128 \
  --epochs 100 \
  --lr 2e-4 \
  --ratios 0.4 0.3 0.3 \
  --offset_mode explicit \
  --offsets 0 1000 2000
```

Interval-based offset (equivalent to legacy `--interval 100`):

```bash
python train.py \
  --train_type ddpm \
  --dataset_mode cifar10_mnist_fashion_mnist \
  --output_dir outputs/ste_ddpm_interval \
  --offset_mode interval \
  --interval 100 \
  --ratios 0.4 0.3 0.3
```

### Flow-Matching Training

```bash
python train.py \
  --train_type flow \
  --dataset_mode cifar10_mnist \
  --output_dir outputs/ste_flow_cifar_mnist \
  --batch_size 128 \
  --epochs 100 \
  --lr 2e-4 \
  --ratios 0.5 0.5 \
  --shadow_bias 1.0
```

### 256×256 Folder Dataset Training

Prepare your data directories:
```
data/
  celeba_hq_256/   ← clean images (any nested folder structure)
  poison_data/     ← shadow images
```

Then train:

```bash
python train.py \
  --train_type ddpm \
  --dataset_mode celeba_poison_256 \
  --clean_dir ./data/celeba_hq_256 \
  --poison_dir ./data/poison_data \
  --output_dir outputs/ste_celeba_256 \
  --image_size 256 \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-5 \
  --offset_mode explicit \
  --offsets 0 1000 \
  --num_class_embeds 2
```

For arbitrary folder datasets use `--dataset_mode custom_folders --custom_data_dirs dir1 dir2 ...`.

### Training Outputs

Each run saves:
```
output_dir/
  train.log
  loss_curve.png
  ckpt/
    model_epoch{N}.pt        ← raw weights
    model_epoch{N}_ema.pt    ← EMA weights
    model_final.pt
    model_final_ema.pt
```

---

## Inference / Image Generation

Generate 50 000 images from the primary domain (offset 0) using DDIM with 50 steps:

```bash
python generate.py \
  --ckpt outputs/ste_ddpm_cifar_mnist/ckpt/model_final_ema.pt \
  --out_dir outputs/generated/cifar10 \
  --num_images 50000 \
  --batch_size 256 \
  --scheduler ddim \
  --steps 50 \
  --offset 0
```

Generate from the shadow domain (offset 1000):

```bash
python generate.py \
  --ckpt outputs/ste_ddpm_cifar_mnist/ckpt/model_final_ema.pt \
  --out_dir outputs/generated/mnist_shadow \
  --num_images 50000 \
  --offset 1000
```

Key arguments:

| Argument | Description |
|---|---|
| `--offset` | Shadow timestep offset at inference (match the value used during training) |
| `--scheduler` | `ddpm` (1000 steps), `ddim` (fast, ~50 steps), `dpmsolver` (fast) |
| `--class_mode` | `balanced` (cycle all classes), `single` (fixed `--class_label`), `none` (unconditional) |

Images are saved as `000000.png`, `000001.png`, ...

---

## FID Evaluation

First export real CIFAR-10 reference images (one-time setup):

```bash
python compute_fid.py \
  --generated_dir outputs/generated/cifar10 \
  --real_dir data/real_cifar10_train \
  --export_real cifar10
```

If the reference directory already exists, export is skipped automatically.

Compute FID without export:

```bash
python compute_fid.py \
  --generated_dir outputs/generated/cifar10 \
  --real_dir data/real_cifar10_train
```

FID computation prefers `torch_fidelity` and falls back to `pytorch_fid`.

---

## Pretrained Checkpoints

TODO — pretrained checkpoints will be released.

---

## Citation

```bibtex
@article{huang2026watch,
  title={Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding},
  author={Huang, An and Son, Junggab and Xiong, Zuobin},
  journal={arXiv preprint arXiv:2605.00935},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
