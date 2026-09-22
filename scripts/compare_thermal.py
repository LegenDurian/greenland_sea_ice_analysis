"""
compare_thermal.py — Binary mask vs UNet side-by-side comparison
================================================================
Runs both the classical binary_mask (FloeSeparator) pipeline and the thermal
UNet on the same thermal image and plots their floe boundaries in a 4-panel
figure: original | binary_mask (red) | UNet (yellow) | both overlaid.

Usage
-----
    python scripts/compare_thermal.py

Configuration (edit variables at the top of this file)
------------------------------------------------------
    IMAGE_PATH      Path to the thermal image to compare
    WEIGHTS_PATH    Path to the thermal UNet weights (.pt)
    TILE_SIZE       Must match training tile size (default 128)
    STRIDE          Must match training stride (default 64)
    BM_CLAHE_CLIP   CLAHE clip limit for binary_mask method
    BM_THRESH_VAL   Binary threshold for binary_mask method

Outputs
-------
    outputs/results/compare/compare_thermal_result.png

Dependencies
------------
    torch, numpy, Pillow, matplotlib, opencv-python
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from binary_mask import FloeSeparator


# ----------------------------
# Config
# ----------------------------
_ROOT     = Path(__file__).parent.parent
IMAGE_PATH = str(_ROOT / "data" / "thermal_timelapse" / "DJI_20250218031618_0007_T.JPG")
WEIGHTS_PATH = str(_ROOT / "outputs" / "dataset_tiles_thermal" / "best_unet.pt")

# Tiling must match what unet_thermal.py was trained with
TILE_SIZE = 128
STRIDE    = 64
THRESHOLD = 0.5

# binary_mask preprocessing params (same as used in binary_mask.py __main__)
BM_CLAHE_CLIP  = 1.5
BM_CLAHE_GRID  = (8, 8)
BM_THRESH_VAL  = 75
BM_MIN_AREA    = 30

SAVE_OUTPUT = True


# ----------------------------
# U-Net (must match unet_thermal.py)
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
        self.enc1 = DoubleConv(in_ch, base);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out  = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _crop(skip, x):
        return skip[:, :, :x.shape[2], :x.shape[3]]

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        u3 = self.up3(b);  d3 = self.dec3(torch.cat([self._crop(e3, u3), u3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([self._crop(e2, u2), u2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([self._crop(e1, u1), u1], dim=1))
        return F.interpolate(self.out(d1), size=(h, w), mode='bilinear', align_corners=False)


# ----------------------------
# UNet inference (sliding window)
# ----------------------------
@torch.no_grad()
def run_unet_inference(model, img_np, tile_size, stride, device):
    model.eval()
    H, W = img_np.shape[:2]
    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for upper in range(0, H - tile_size + 1, stride):
        for left in range(0, W - tile_size + 1, stride):
            tile = img_np[upper:upper+tile_size, left:left+tile_size]
            x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            accum[upper:upper+tile_size, left:left+tile_size] += prob
            count[upper:upper+tile_size, left:left+tile_size] += 1.0

    return accum / np.maximum(count, 1.0)


def main():
    # ------------------------------------------------------------------
    # 1) Binary mask pipeline
    # ------------------------------------------------------------------
    print("Running binary_mask pipeline...")
    processor = FloeSeparator(IMAGE_PATH)
    img_gray, img_rgb, bm_mask = processor.preprocess(
        clahe_clip=BM_CLAHE_CLIP,
        clahe_grid=BM_CLAHE_GRID,
        thresh_val=BM_THRESH_VAL,
    )
    labels_filtered, _, _ = processor.split_erode_dilate_small(bm_mask, min_area=BM_MIN_AREA)

    # Binary ice/background map from labelled floes (1=ice, 0=background)
    bm_binary = (labels_filtered > 0).astype(np.float32)
    print(f"  Floes detected: {int(labels_filtered.max())}")

    # ------------------------------------------------------------------
    # 2) UNet thermal inference
    # ------------------------------------------------------------------
    print("Running UNet thermal inference...")
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_np  = np.asarray(img_pil, dtype=np.float32) / 255.0

    device = "cpu"
    model  = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    prob_map  = run_unet_inference(model, img_np, TILE_SIZE, STRIDE, device)
    unet_mask = (prob_map > THRESHOLD).astype(np.float32)
    print("  UNet inference complete.")

    # ------------------------------------------------------------------
    # 3) Plot comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Thermal floe boundary comparison", fontsize=13)

    # Panel 1: original image
    axes[0].set_title("Original thermal image")
    axes[0].imshow(img_rgb)
    axes[0].axis("off")

    # Panel 2: binary_mask boundaries (red, matching binary_mask.py style)
    axes[1].set_title("binary_mask boundaries")
    axes[1].imshow(img_rgb)
    axes[1].contour(bm_binary, levels=[0.5], colors="red", linewidths=0.8)
    axes[1].axis("off")

    # Panel 3: UNet boundaries (yellow, matching infer.py style)
    axes[2].set_title("UNet thermal boundaries")
    axes[2].imshow(img_rgb)
    axes[2].contour(unet_mask, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[2].axis("off")

    # Panel 4: both overlaid on the same image
    axes[3].set_title("Both overlaid\n(red=binary_mask, yellow=UNet)")
    axes[3].imshow(img_rgb)
    axes[3].contour(bm_binary, levels=[0.5], colors="red",    linewidths=0.8)
    axes[3].contour(unet_mask, levels=[0.5], colors="yellow", linewidths=0.8)
    axes[3].axis("off")

    plt.tight_layout()

    if SAVE_OUTPUT:
        out_dir = _ROOT / "outputs" / "results" / "compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "compare_thermal_result.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
