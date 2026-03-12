"""
Satellite Image Super-Resolution using SRGAN
============================================
Uses UC Merced Land Use Dataset (streamed via torchvision/HuggingFace)
No large downloads needed - images fetched on-the-fly.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import PerceptualLoss
from utils.dataset import SatelliteDataset, StreamingDataset
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualize import save_comparison_grid

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "scale_factor"    : 4,          # 4x upscaling
    "num_epochs"      : 10,
    "batch_size"      : 4,
    "lr_gen"          : 1e-4,
    "lr_disc"         : 1e-4,
    "lambda_content"  : 1.0,
    "lambda_adv"      : 1e-3,
    "lambda_pixel"    : 1e-2,
    "patch_size"      : 96,         # HR patch size (LR = 24x24)
    "num_res_blocks"  : 16,
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
    "save_every"      : 10,
    "results_dir"     : "results",
    "checkpoints_dir" : "checkpoints",
    "use_streaming"   : True,       # Stream dataset without downloading
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)

print(f"🚀 Using device: {CONFIG['device']}")
print(f"📡 Satellite Super-Resolution | Scale Factor: {CONFIG['scale_factor']}x")


def train():
    device = CONFIG["device"]

    # ── Models ────────────────────────────────────────────────────────────────
    generator     = Generator(scale_factor=CONFIG["scale_factor"],
                              num_res_blocks=CONFIG["num_res_blocks"]).to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator     params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ── Loss Functions ────────────────────────────────────────────────────────
    perceptual_loss = PerceptualLoss().to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    pixel_loss      = nn.L1Loss()

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_gen  = optim.Adam(generator.parameters(),     lr=CONFIG["lr_gen"],  betas=(0.9, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=CONFIG["lr_disc"], betas=(0.9, 0.999))

    scheduler_gen  = optim.lr_scheduler.StepLR(opt_gen,  step_size=50, gamma=0.5)
    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=50, gamma=0.5)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if CONFIG["use_streaming"]:
        print("📥 Using streaming dataset (no large download required)...")
        dataset = StreamingDataset(patch_size=CONFIG["patch_size"],
                                   scale_factor=CONFIG["scale_factor"],
                                   num_samples=200)
    else:
        dataset = SatelliteDataset(data_dir="data/satellite",
                                   patch_size=CONFIG["patch_size"],
                                   scale_factor=CONFIG["scale_factor"])

    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    print(f"📊 Dataset size: {len(dataset)} samples | Batches: {len(dataloader)}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {"gen_loss": [], "disc_loss": [], "psnr": [], "ssim": []}

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        generator.train()
        discriminator.train()

        epoch_gen_loss  = 0.0
        epoch_disc_loss = 0.0
        epoch_psnr      = 0.0
        epoch_ssim      = 0.0

        t0 = time.time()

        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            batch_size = lr_imgs.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ── Train Discriminator ────────────────────────────────────────
            opt_disc.zero_grad()

            sr_imgs = generator(lr_imgs).detach()

            real_preds = discriminator(hr_imgs)
            fake_preds = discriminator(sr_imgs)

            loss_real = adversarial_loss(real_preds, real_labels)
            loss_fake = adversarial_loss(fake_preds, fake_labels)
            disc_loss = (loss_real + loss_fake) / 2

            disc_loss.backward()
            opt_disc.step()

            # ── Train Generator ────────────────────────────────────────────
            opt_gen.zero_grad()

            sr_imgs   = generator(lr_imgs)
            fake_preds = discriminator(sr_imgs)

            content_loss = perceptual_loss(sr_imgs, hr_imgs)
            adv_loss     = adversarial_loss(fake_preds, real_labels)
            pix_loss     = pixel_loss(sr_imgs, hr_imgs)

            gen_loss = (CONFIG["lambda_content"] * content_loss +
                        CONFIG["lambda_adv"]     * adv_loss     +
                        CONFIG["lambda_pixel"]   * pix_loss)

            gen_loss.backward()
            opt_gen.step()

            # ── Metrics ───────────────────────────────────────────────────
            with torch.no_grad():
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                ssim = calculate_ssim(sr_imgs, hr_imgs)

            epoch_gen_loss  += gen_loss.item()
            epoch_disc_loss += disc_loss.item()
            epoch_psnr      += psnr
            epoch_ssim      += ssim

        n = len(dataloader)
        avg_gen  = epoch_gen_loss  / n
        avg_disc = epoch_disc_loss / n
        avg_psnr = epoch_psnr      / n
        avg_ssim = epoch_ssim      / n
        elapsed  = time.time() - t0

        history["gen_loss"].append(avg_gen)
        history["disc_loss"].append(avg_disc)
        history["psnr"].append(avg_psnr)
        history["ssim"].append(avg_ssim)

        print(f"Epoch [{epoch:3d}/{CONFIG['num_epochs']}] "
              f"G: {avg_gen:.4f} | D: {avg_disc:.4f} | "
              f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | "
              f"⏱ {elapsed:.1f}s")

        scheduler_gen.step()
        scheduler_disc.step()

        # ── Save samples & checkpoints ─────────────────────────────────────
        if epoch % CONFIG["save_every"] == 0:
            generator.eval()
            with torch.no_grad():
                lr_sample, hr_sample = next(iter(dataloader))
                lr_sample = lr_sample[:4].to(device)
                hr_sample = hr_sample[:4].to(device)
                sr_sample = generator(lr_sample)

            save_comparison_grid(
                lr_sample, sr_sample, hr_sample,
                path=f"{CONFIG['results_dir']}/epoch_{epoch:03d}.png",
                scale_factor=CONFIG["scale_factor"]
            )

            torch.save({
                "epoch"      : epoch,
                "generator"  : generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_gen"    : opt_gen.state_dict(),
                "opt_disc"   : opt_disc.state_dict(),
                "history"    : history,
            }, f"{CONFIG['checkpoints_dir']}/checkpoint_epoch_{epoch:03d}.pth")

            print(f"  💾 Saved results & checkpoint at epoch {epoch}")

    print("\n✅ Training complete!")
    print(f"   Best PSNR: {max(history['psnr']):.2f} dB at epoch "
          f"{history['psnr'].index(max(history['psnr'])) + 1}")
    return history


if __name__ == "__main__":
    train()
