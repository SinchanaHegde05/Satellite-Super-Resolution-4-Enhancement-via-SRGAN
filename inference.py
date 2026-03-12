"""
inference.py — Run super-resolution on a single image or folder.

Usage:
    # Single image
    python inference.py --input my_satellite.jpg --checkpoint checkpoints/checkpoint_epoch_100.pth

    # Folder of images
    python inference.py --input data/test/ --checkpoint checkpoints/checkpoint_epoch_100.pth --output results/sr/
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.generator import Generator


def load_generator(checkpoint_path: str, scale_factor: int = 4,
                   num_res_blocks: int = 16, device: str = "cpu") -> Generator:
    gen = Generator(scale_factor=scale_factor, num_res_blocks=num_res_blocks)
    ckpt = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(ckpt["generator"])
    gen.to(device).eval()
    print(f"✅ Loaded generator from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return gen


def super_resolve(image_path: str, generator: Generator,
                  device: str = "cpu") -> Image.Image:
    """Upscale a single image using the trained generator."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    lr_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = generator(lr_tensor)

    # Denormalize [-1,1] → [0,1]
    sr_tensor = (sr_tensor + 1.0) / 2.0
    sr_tensor = sr_tensor.clamp(0, 1).squeeze(0).cpu()

    sr_img = transforms.ToPILImage()(sr_tensor)
    print(f"  {w}×{h}  →  {sr_img.width}×{sr_img.height}")
    return sr_img


def main():
    parser = argparse.ArgumentParser(description="Satellite Image Super-Resolution Inference")
    parser.add_argument("--input",        required=True,  help="Input image or folder")
    parser.add_argument("--checkpoint",   required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--output",       default="results/inference", help="Output directory")
    parser.add_argument("--scale",        type=int, default=4,  help="Scale factor (2/4/8)")
    parser.add_argument("--res-blocks",   type=int, default=16, help="Num residual blocks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    generator = load_generator(args.checkpoint, args.scale, args.res_blocks, device)
    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        paths = [input_path]
    else:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        paths = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]

    print(f"\n📷 Processing {len(paths)} image(s)…")
    t0 = time.time()

    for p in paths:
        print(f"  {p.name}", end="  ")
        sr = super_resolve(str(p), generator, device)
        out_path = os.path.join(args.output, f"sr_{p.stem}.png")
        sr.save(out_path)

    elapsed = time.time() - t0
    print(f"\n✅ Done — {len(paths)} image(s) in {elapsed:.1f}s")
    print(f"   Results saved to: {args.output}")


if __name__ == "__main__":
    main()
