"""
infer.py — Single-image UNet inference
=======================================
Loads a trained UNet model and runs sliding-window inference on a single image.
Displays a 3-panel matplotlib figure: original | predicted mask | boundary overlay.

Usage
-----
    python scripts/infer.py

Configuration (edit variables at the top of this file)
------------------------------------------------------
    MODE            "rgb" or "thermal" — selects weights file and tile size
    IMAGE_OVERRIDE  Path to a custom image, or None for the mode default
    THRESHOLD       Probability threshold for binarizing UNet output (default 0.5)

Inputs
------
    Image file (from data/images/ or a custom path).
    Trained weights:
        - RGB mode:     outputs/dataset_tiles/best_unet.pt
        - Thermal mode: outputs/dataset_tiles_thermal/best_unet.pt

Outputs
-------
    outputs/results/single/infer_{MODE}_result.png

Dependencies
------------
    torch, numpy, Pillow, matplotlib
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ----------------------------
# Config (edit these)
# ----------------------------
MODE = "thermal"   # "rgb" or "thermal"

_DATA_DIR = Path(__file__).parent.parent / "data" / "images"

# Per-mode settings
_MODES = {
    "rgb": {
        "image_path": str(_DATA_DIR / "unet_test_ice_2.JPG"),
        "weights":    str(Path(__file__).parent.parent / "outputs" / "dataset_tiles" / "best_unet.pt"),
        "tile_size":  500,
        "stride":     250,
    },
    "thermal": {
        "image_path": str(_DATA_DIR / "unet_test_thermal.JPG"),
        "weights":    str(Path(__file__).parent.parent / "outputs" / "dataset_tiles_thermal" / "best_unet.pt"),
        "tile_size":  128,
        "stride":     64,
    },
}

# Optional: override the image path here (leave as None to use the mode default)
IMAGE_OVERRIDE = None

THRESHOLD = 0.5   # probability threshold for binary mask
SAVE_OUTPUT = True  # save result images alongside the weights file


# ----------------------------
# U-Net (must match training)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _crop(skip, x):
        return skip[:, :, :x.shape[2], :x.shape[3]]

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([self._crop(e3, d3), d3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([self._crop(e2, d2), d2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([self._crop(e1, d1), d1], dim=1)
        d1 = self.dec1(d1)
        logits = self.out(d1)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference(model, img_np: np.ndarray, tile_size: int, stride: int, device: str) -> np.ndarray:
    """
    Slide a window over the full image, run the model on each tile, and
    stitch predictions back by averaging overlapping regions.

    Args:
        img_np: float32 array [H, W, 3] in [0, 1]
    Returns:
        prob_map: float32 array [H, W] with predicted probabilities in [0, 1]
    """
    model.eval()
    H, W = img_np.shape[:2]

    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for upper in range(0, H - tile_size + 1, stride):
        for left in range(0, W - tile_size + 1, stride):
            lower = upper + tile_size
            right = left  + tile_size

            tile = img_np[upper:lower, left:right]  # [tile_size, tile_size, 3]
            x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [tile_size, tile_size]

            accum[upper:lower, left:right] += prob
            count[upper:lower, left:right] += 1.0

    # Avoid division by zero for any unvisited border pixels
    count = np.maximum(count, 1.0)
    return accum / count


def main():
    cfg = _MODES[MODE]
    image_path = IMAGE_OVERRIDE if IMAGE_OVERRIDE else cfg["image_path"]
    weights_path = cfg["weights"]
    tile_size = cfg["tile_size"]
    stride = cfg["stride"]

    print(f"Mode      : {MODE}")
    print(f"Image     : {image_path}")
    print(f"Weights   : {weights_path}")
    print(f"Tile/Stride: {tile_size}/{stride}")

    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0  # [H, W, 3]
    H, W = img_np.shape[:2]
    print(f"Image size: {W}x{H}")

    # Load model
    device = "cpu"
    model = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Weights loaded.")

    # Run inference
    print("Running inference...", flush=True)
    prob_map = run_inference(model, img_np, tile_size, stride, device)
    mask = (prob_map > THRESHOLD).astype(np.float32)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Full-image inference  [{MODE}]")

    axes[0].set_title("Input image")
    axes[0].imshow(img_np)
    axes[0].axis("off")

    axes[1].set_title("Predicted mask")
    axes[1].imshow(mask, cmap="gray")
    axes[1].axis("off")

    axes[2].set_title("Boundary overlay")
    axes[2].imshow(img_np)
    axes[2].contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[2].axis("off")

    plt.tight_layout()

    if SAVE_OUTPUT:
        out_dir = Path(__file__).parent.parent / "outputs" / "results" / "single"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_dir / f"infer_{MODE}_result.png"), dpi=150, bbox_inches="tight")
        print(f"Saved to: {out_dir / f'infer_{MODE}_result.png'}")

    plt.show()


if __name__ == "__main__":
    main()
