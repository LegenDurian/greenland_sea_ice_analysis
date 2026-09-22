"""
unet_thermal.py — Thermal UNet training (single image)
=======================================================
Trains a small UNet model on a single annotated thermal sea ice image (640x512).
Tiles the image into 128x128 crops with stride 64, producing ~63 tiles that are
split into train/val/test sets.

Usage
-----
    python scripts/training/unet_thermal.py

Configuration (edit variables below)
-------------------------------------
    IMAGE_PATH    Path to the thermal image (640x512)
    MASK_PATH     Path to the corresponding CVAT-exported mask PNG
    OUT_DIR       Output directory for tiles and weights
    TILE_SIZE     Tile crop size (default 128)
    STRIDE        Tile overlap stride (default 64)
    EPOCHS        Number of training epochs (default 75)
    LR            Learning rate (default 1e-3)

Inputs
------
    data/images/unet_test_thermal.JPG + unet_test_thermal_mask.png

Outputs
-------
    outputs/dataset_tiles_thermal/
        train/val/test/{images,masks}/  — tiled dataset
        best_unet.pt                    — best model weights (by val Dice)

Dependencies
------------
    torch, torchvision, numpy, Pillow, matplotlib
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ----------------------------
# Config (edit these)
# ----------------------------
_DATA_DIR  = Path(__file__).parent.parent / "data" / "images"
IMAGE_PATH = str(_DATA_DIR / "unet_test_thermal.JPG")       # 640x512 thermal image
MASK_PATH  = str(_DATA_DIR / "unet_test_thermal_mask.png")  # CVAT exported mask (same size as image)
OUT_DIR    = str(Path(__file__).parent.parent / "outputs" / "dataset_tiles_thermal")  # output folder

# Split ratios for tiles (POC only if single source image)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

TILE_SIZE = 128   # 640x512 image -> (640-128)/64+1=9 cols, (512-128)/64+1=7 rows = 63 tiles
STRIDE    = 64

SEED = 42
BATCH_SIZE = 4
EPOCHS = 75
LR = 1e-3

# If your CVAT mask uses label id 1 for ice and 0 for background, this is fine.
# If your mask has multiple labels, we binarize as (mask > 0) by default.
ICE_IS_NONZERO = True  # set False if you want to treat only value==1 as ice


# ----------------------------
# Utilities: tiling + splitting
# ----------------------------
def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def tile_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
) -> List[Tuple[Image.Image, Image.Image, Tuple[int, int]]]:
    """
    Sliding-window tiling with overlap.
    Returns list of (img_tile, mask_tile, (row, col)).
    Tiles that would exceed image bounds are skipped (no padding).
    Assumes image and mask are same size.
    """
    if image.size != mask.size:
        raise ValueError(f"Image and mask size mismatch: {image.size} vs {mask.size}")

    width, height = image.size  # PIL is (W, H)

    tiles = []
    r = 0
    for upper in range(0, height - tile_size + 1, stride):
        c = 0
        for left in range(0, width - tile_size + 1, stride):
            right = left + tile_size
            lower = upper + tile_size

            img_t = image.crop((left, upper, right, lower))
            msk_t = mask.crop((left, upper, right, lower))
            tiles.append((img_t, msk_t, (r, c)))
            c += 1
        r += 1

    return tiles

def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int = 0):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return train_idx, val_idx, test_idx

def save_tiles(
    tiles: List[Tuple[Image.Image, Image.Image, Tuple[int, int]]],
    out_dir: str,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> None:
    splits = {
        "train": set(train_idx),
        "val": set(val_idx),
        "test": set(test_idx),
    }

    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(out_dir, split, "images"))
        ensure_dir(os.path.join(out_dir, split, "masks"))

    for i, (img_t, msk_t, (r, c)) in enumerate(tiles):
        split = "train" if i in splits["train"] else "val" if i in splits["val"] else "test"

        base = f"tile_r{r}_c{c}"
        img_out = os.path.join(out_dir, split, "images", base + ".png")
        msk_out = os.path.join(out_dir, split, "masks", base + ".png")

        img_t.save(img_out)
        msk_t.save(msk_out)

    print(f"Saved {len(tiles)} tiles into: {out_dir}")
    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


# ----------------------------
# Dataset
# ----------------------------
class SegTileDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, ice_is_nonzero: bool = True, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ice_is_nonzero = ice_is_nonzero
        self.augment = augment

        self.items = sorted([p for p in self.images_dir.glob("*.png")])
        if not self.items:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path = self.items[idx]
        mask_path = self.masks_dir / img_path.name
        if not mask_path.exists():
            raise RuntimeError(f"Mask not found for {img_path.name} at {mask_path}")

        # Image -> float array [H,W,C] in [0,1]
        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        # Mask -> binary array [H,W] in {0,1}
        m = Image.open(mask_path)
        m_np = np.asarray(m)

        if m_np.ndim == 3:
            m_np = m_np.max(axis=-1)

        if self.ice_is_nonzero:
            m_bin = (m_np > 0).astype(np.float32)
        else:
            m_bin = (m_np == 1).astype(np.float32)

        # Augmentation: random flips, rotations, and colour jitter (train only)
        if self.augment:
            if random.random() < 0.5:                        # horizontal flip
                img_np = np.fliplr(img_np)
                m_bin  = np.fliplr(m_bin)
            if random.random() < 0.5:                        # vertical flip
                img_np = np.flipud(img_np)
                m_bin  = np.flipud(m_bin)
            k = random.randint(0, 3)                         # 0/90/180/270 rotation
            if k:
                img_np = np.rot90(img_np, k)
                m_bin  = np.rot90(m_bin,  k)
            # Brightness jitter: scale all channels by a random factor
            img_np = np.clip(img_np * random.uniform(0.8, 1.2), 0.0, 1.0)
            # Contrast jitter: scale deviation from mean
            mean   = img_np.mean()
            img_np = np.clip((img_np - mean) * random.uniform(0.8, 1.2) + mean, 0.0, 1.0)

        img_t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1)  # [3,H,W]
        m_t   = torch.from_numpy(np.ascontiguousarray(m_bin)).unsqueeze(0)       # [1,H,W]

        return img_t, m_t


# ----------------------------
# Simple U-Net (small, CPU-friendly)
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

        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _crop(skip, x):
        """Crop skip connection to match x's spatial size (handles odd-dimension rounding)."""
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
        # Resize back to input spatial size (handles non-power-of-8 tile sizes)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)


# ----------------------------
# Loss + metrics
# ----------------------------
def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    dice = num / den
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_with_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    num = 2.0 * (preds * targets).sum(dim=(2,3))
    den = (preds + targets).sum(dim=(2,3)) + eps
    return (num / den).mean().item()


# ----------------------------
# Train / eval loops
# ----------------------------
def run_epoch(model, loader, optimizer=None, device="cpu", pos_weight=None):
    train = optimizer is not None
    model.train(train)

    bce = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss_bce = bce(logits, y)
        loss_dice = dice_loss_with_logits(logits, y)
        loss = loss_bce + loss_dice

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_dice += dice_score_with_logits(logits, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_dice / n

@torch.no_grad()
def show_test_predictions(
    model: torch.nn.Module,
    test_loader,
    device: str = "cpu",
    threshold: float = 0.5,
    max_batches: int = 2,   # how many batches to visualize
):
    model.eval()

    shown = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)          # [B,1,H,W]
        preds = (probs > threshold).float()    # [B,1,H,W]

        # --- compute a "border" map from prediction ---
        # A simple morphological gradient: dilate - erode on the binary mask
        # Using maxpool for dilation; erosion via negation trick.
        # border will be 1 on boundary pixels, 0 elsewhere.
        dil = F.max_pool2d(preds, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-preds, kernel_size=3, stride=1, padding=1)
        border = (dil - ero).clamp(0, 1)  # [B,1,H,W]

        bsz = x.shape[0]
        for i in range(bsz):
            # Convert tensors to numpy for plotting
            img = x[i].detach().cpu().permute(1, 2, 0).numpy()     # [H,W,3] in [0,1]
            gt  = y[i, 0].detach().cpu().numpy()                   # [H,W]
            pr  = preds[i, 0].detach().cpu().numpy()               # [H,W]
            bd  = border[i, 0].detach().cpu().numpy()              # [H,W]

            # Make a contour from the predicted boundary:
            # Plotting contours directly on the image is usually clearer than a colored overlay.
            fig = plt.figure(figsize=(12, 10))

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.set_title("Test tile (input)")
            ax1.imshow(img)
            ax1.axis("off")

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.set_title("Ground truth mask")
            ax2.imshow(gt)
            ax2.axis("off")

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_title("Predicted mask")
            ax3.imshow(pr)
            ax3.axis("off")

            ax4 = fig.add_subplot(2, 2, 4)
            ax4.set_title("Predicted boundary on input")
            ax4.imshow(img)
            # Draw boundary as contours; default styling (no explicit colors set)
            ax4.contour(bd, levels=[0.5])
            ax4.axis("off")

            plt.tight_layout()
            plt.show()

        shown += 1
        if shown >= max_batches:
            break


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) Tile + split + save (clear stale tiles first)
    if Path(OUT_DIR).exists():
        shutil.rmtree(OUT_DIR)
    img = load_image(IMAGE_PATH).convert("RGB")
    mask = load_image(MASK_PATH)  # keep original mode

    tiles = tile_image_and_mask(img, mask)
    train_idx, val_idx, test_idx = split_indices(len(tiles), TRAIN_RATIO, VAL_RATIO, seed=SEED)
    save_tiles(tiles, OUT_DIR, train_idx, val_idx, test_idx)

    # 2) Datasets / loaders
    train_ds = SegTileDataset(
        images_dir=os.path.join(OUT_DIR, "train", "images"),
        masks_dir=os.path.join(OUT_DIR, "train", "masks"),
        ice_is_nonzero=ICE_IS_NONZERO,
        augment=True,
    )
    val_ds = SegTileDataset(
        images_dir=os.path.join(OUT_DIR, "val", "images"),
        masks_dir=os.path.join(OUT_DIR, "val", "masks"),
        ice_is_nonzero=ICE_IS_NONZERO,
    )
    test_ds = SegTileDataset(
        images_dir=os.path.join(OUT_DIR, "test", "images"),
        masks_dir=os.path.join(OUT_DIR, "test", "masks"),
        ice_is_nonzero=ICE_IS_NONZERO,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3) Class-balance weight from training masks (bg_pixels / ice_pixels)
    ice_px, bg_px = 0, 0
    for p in train_ds.items:
        m_np = np.asarray(Image.open(train_ds.masks_dir / p.name))
        if m_np.ndim == 3:
            m_np = m_np.max(axis=-1)
        ice_px += int((m_np > 0).sum())
        bg_px  += int((m_np == 0).sum())
    pos_weight = torch.tensor([bg_px / max(ice_px, 1)], dtype=torch.float32)
    print(f"Class balance — ice: {ice_px}, bg: {bg_px}, pos_weight: {pos_weight.item():.2f}")

    # 4) Model (CPU)
    device = "cpu"
    model = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 5) Train
    best_val_dice = -1.0
    best_path = os.path.join(OUT_DIR, "best_unet.pt")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer=optimizer, device=device, pos_weight=pos_weight)
        va_loss, va_dice = run_epoch(model, val_loader, optimizer=None, device=device, pos_weight=pos_weight)

        scheduler.step()
        print(f"Epoch {epoch:02d} | lr {scheduler.get_last_lr()[0]:.2e} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}", flush=True)

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_path)

    print(f"Best val dice: {best_val_dice:.4f}")
    print(f"Saved best model to: {best_path}")

    # 6) Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_dice = run_epoch(model, test_loader, optimizer=None, device=device, pos_weight=pos_weight)
    print(f"Test loss {te_loss:.4f} | Test dice {te_dice:.4f}")

    # Visualize predictions on test set
    show_test_predictions(model, test_loader, device=device, threshold=0.5, max_batches=2)


if __name__ == "__main__":
    main()
