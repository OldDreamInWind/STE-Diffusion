# eval_diffusion_fid.py
import os, sys
import logging
import math
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as vutils
from PIL import Image

from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler

# --------- 你需要提供的：导入并构建你的 UNet 模型 ----------
# 假设你的训练代码里有与之匹配的构造函数，例如 build_model(...)
# 或者你直接 from your_file import UNet2DModel 并实例化相同结构
from diffusers import UNet2DModel

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "fid.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")
    return log_path

def build_model(num_classes_with_null: int = 11):
    """UNet with class embedding. NULL id is the last class (10)."""
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        add_attention=True,
        num_class_embeds=num_classes_with_null,
        dropout=0.1, 
    )
    return model

# -------------------- 工具函数 --------------------
def save_tensor_as_png(x: torch.Tensor, out_path: Path):
    """
    x: [-1,1] or [0,1] 的张量, 形状 [3,H,W]
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5  # 到[0,1]
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to(torch.uint8).numpy()
    Image.fromarray(x).save(out_path)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_scheduler(name: str, num_train_timesteps: int = 1000):
    name = name.lower()
    if name == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
    if name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
    if name in {"dpmsolver","dpms"}:
        return DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
    raise ValueError(f"unknown scheduler: {name}")

@torch.no_grad()
def sample_batch(
    net: UNet2DModel,
    scheduler,
    batch_size: int,
    device: torch.device,
    num_inference_steps: int = 50,
    class_labels: torch.Tensor = None,
    offset: int = 0
):
    """
    net 期望输入范围 [-1,1] 的图像张量。
    """
    net.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    x = torch.randn(batch_size, 3, 32, 32, device=device)

    for t in scheduler.timesteps:
        real_t = t + offset
        with torch.no_grad():
            eps = net(x, real_t, class_labels=class_labels).sample
        x  = scheduler.step(eps, t, x).prev_sample

    return x  # [-1,1]

def export_cifar10_real_images(out_dir: Path):
    """
    将 CIFAR-10 真实训练集导出为 PNG（50,000 张）。
    仅首次需要，之后可复用该目录用于 FID 参照。
    """
    if any(out_dir.iterdir()):
        return  # 已经导出过
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    ensure_dir(out_dir)
    for i, (img, _) in enumerate(tqdm(ds, desc="Export CIFAR10 real")):
        # img ∈ [0,1], [3,32,32]
        x = img.clamp(0,1).mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to(torch.uint8).numpy()
        Image.fromarray(x).save(out_dir / f"{i:06d}.png")

def compute_fid(generated_dir: Path, real_dir: Path, device: str = "cuda"):
    """
    优先使用 torch-fidelity；若不可用则回退到 pytorch-fid。
    """
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=str(generated_dir),
            input2=str(real_dir),
            cuda=device.startswith("cuda"),
            isc=False, kid=False, fid=True,
            verbose=True,
            samples_find_deep=True
        )
        return float(metrics["frechet_inception_distance"])
    except Exception as e:
        print(f"[torch-fidelity] 不可用或出错，将尝试 pytorch-fid。原因：{e}")
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [str(generated_dir), str(real_dir)],
            batch_size=128, device=device, dims=2048
        )
        return float(fid)

# -------------------- 主逻辑：采样 50k + 计算 FID --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False, default=None, help="模型权重路径（.pt/.pth/.safetensors）。若为空则用随机初始化示例（仅演示）")
    ap.add_argument("--ema_ckpt", type=str, default=None, help="EMA 权重（可选）")
    ap.add_argument("--out_dir", type=str, default="./gen_cifar10_50k", help="生成图片输出目录")
    ap.add_argument("--real_dir", type=str, default="./real_cifar10_train", help="CIFAR10 真实图像导出目录")
    ap.add_argument("--log_dir", type=str, default="./gen_cifar10_50k", help="生成图片输出目录")
    ap.add_argument("--num_images", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm","ddim","dpmsolver"])
    ap.add_argument("--steps", type=int, default=1000, help="采样步数")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    setup_logging(args.log_dir)

    logging.info("Starting evaluation...")
    
    device = torch.device(args.device)
    model = build_model(num_classes_with_null=10).to(device)
    offset = args.offset

    # ---- 加载权重（按你的保存方式改写）----
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[OK] loaded weights: {args.ckpt}")

    # 如果你保存了 EMA（diffusers.EMAModel），可以把它 copy 到 model 上
    if args.ema_ckpt:
        ema_state_dict = torch.load(args.ema_ckpt, map_location="cpu")
        model.load_state_dict(ema_state_dict)
        print(f"[OK] loaded EMA and copied to model: {args.ema_ckpt}")


    # ---- 构建采样器 ----
    scheduler = build_scheduler(args.scheduler, num_train_timesteps=1000)

    # ---- 均衡类条件：50k / 10 类，每类一样多 ----
    n = args.num_images
    per_class = math.ceil(n / 10)
    class_list = []
    for c in range(10):
        class_list += [c] * per_class
    class_list = class_list[:n]  # 截到正好 n
    class_labels_all = torch.tensor(class_list, dtype=torch.long)

    # ---- 输出目录 & 导出真实 CIFAR10 ----
    gen_dir = Path(args.out_dir)
    real_dir = Path(args.real_dir)
    ensure_dir(gen_dir)
    ensure_dir(real_dir)
    # export_cifar10_real_images(real_dir)

    # ---- 生成 50k ----
    B = args.batch_size
    steps = args.steps
    total = n
    saved = 0
    # pbar = tqdm(total=total, desc=f"Generating {total} images ({args.scheduler}/{steps} steps)")
    while saved < total:
        bsz = min(B, total - saved)
        cls = class_labels_all[saved:saved+bsz].to(device)
        imgs = sample_batch(
            net=model, scheduler=scheduler, batch_size=bsz, device=device,
            num_inference_steps=steps, class_labels=cls, offset=offset
        )
        # 保存
        for i in range(bsz):
            save_tensor_as_png(imgs[i], gen_dir / f"{saved+i:06d}.png")
        saved += bsz
    #     pbar.update(bsz)
    # pbar.close()

    # ---- 计算 FID ----
    fid = compute_fid(gen_dir, real_dir, device=args.device)
    logging.info(f"\n==============================")
    logging.info(f"FID: {fid:.4f}")
    logging.info(f"Generated in: {gen_dir}")
    logging.info(f"Real data   : {real_dir}")
    logging.info(f"==============================\n")

if __name__ == "__main__":
    main()
