"""
evaluate.py — Benchmark the trained model on a held-out test set.

Computes:
  - PSNR  (Peak Signal-to-Noise Ratio, dB)
  - SSIM  (Structural Similarity Index)
  - Inference speed (images/sec)
  - Comparison: Bicubic baseline vs SRGAN

Usage:
    python evaluate.py --checkpoint checkpoints/checkpoint_epoch_100.pth
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from models.generator import Generator
from utils.dataset import StreamingDataset
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualize import save_comparison_grid


def evaluate(checkpoint_path: str, num_test: int = 100,
             scale_factor: int = 4, batch_size: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Evaluation | Device: {device} | Scale: {scale_factor}x")

    # ── Load model ────────────────────────────────────────────────────────────
    gen = Generator(scale_factor=scale_factor)
    ckpt = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(ckpt["generator"])
    gen.to(device).eval()
    print(f"   Loaded from epoch {ckpt.get('epoch', '?')}")

    # ── Test dataset ──────────────────────────────────────────────────────────
    dataset = StreamingDataset(patch_size=96, scale_factor=scale_factor,
                               num_samples=num_test, split="test")
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sr_psnr_list, sr_ssim_list = [], []
    bc_psnr_list, bc_ssim_list = [], []
    inference_times = []

    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)

            # ─ SRGAN output ──────────────────────────────────────────────────
            t0 = time.time()
            sr = gen(lr)
            inference_times.append(time.time() - t0)

            # ─ Bicubic baseline ───────────────────────────────────────────────
            bc = torch.nn.functional.interpolate(
                lr, scale_factor=scale_factor,
                mode="bicubic", align_corners=False
            ).clamp(-1, 1)

            sr_psnr_list.append(calculate_psnr(sr, hr))
            sr_ssim_list.append(calculate_ssim(sr, hr))
            bc_psnr_list.append(calculate_psnr(bc, hr))
            bc_ssim_list.append(calculate_ssim(bc, hr))

            # ─ Save first batch comparison ────────────────────────────────────
            if i == 0:
                save_comparison_grid(lr, sr, hr, "results/eval_comparison.png", scale_factor)

    # ── Summary ───────────────────────────────────────────────────────────────
    def avg(lst): return sum(lst) / len(lst)

    sr_psnr = avg(sr_psnr_list)
    sr_ssim = avg(sr_ssim_list)
    bc_psnr = avg(bc_psnr_list)
    bc_ssim = avg(bc_ssim_list)
    fps     = batch_size / avg(inference_times)

    print("\n" + "="*55)
    print(f"{'Metric':<20} {'Bicubic':>12} {'SRGAN (ours)':>12}")
    print("-"*55)
    print(f"{'PSNR (dB)':<20} {bc_psnr:>12.2f} {sr_psnr:>12.2f}  {'↑' if sr_psnr > bc_psnr else '↓'}")
    print(f"{'SSIM':<20} {bc_ssim:>12.4f} {sr_ssim:>12.4f}  {'↑' if sr_ssim > bc_ssim else '↓'}")
    print(f"{'Speed (img/s)':<20} {'—':>12} {fps:>12.1f}")
    print("="*55)

    if sr_psnr > bc_psnr:
        print(f"\n🎉 SRGAN improves PSNR by {sr_psnr - bc_psnr:.2f} dB over bicubic!")

    return {"sr_psnr": sr_psnr, "sr_ssim": sr_ssim,
            "bc_psnr": bc_psnr, "bc_ssim": bc_ssim, "fps": fps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-test",   type=int, default=100)
    parser.add_argument("--scale",      type=int, default=4)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.num_test, args.scale)
