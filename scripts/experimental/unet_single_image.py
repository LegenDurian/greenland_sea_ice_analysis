"""Experimental: Original single-image UNet trainer (superseded by scripts/training/unet.py)."""

import os
import random
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
_DATA_DIR  = Path(__file__).parent.parent.parent / "data" / "images"
IMAGE_PATH = str(_DATA_DIR / "sea_ice.jpg")             # your 4000x3000 RGB image (or JPG/PNG)
MASK_PATH  = str(_DATA_DIR / "sea_ice_mask.png")        # CVAT exported mask (same size as image)
OUT_DIR    = str(Path(__file__).parent.parent.parent / "outputs" / "dataset_tiles")  # output folder

# Split ratios for tiles (POC only if single source image)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
BATCH_SIZE = 2
EPOCHS = 10
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
    grid: Tuple[int, int] = (3, 4),  # (rows, cols) -> 12 tiles
) -> List[Tuple[Image.Image, Image.Image, Tuple[int, int]]]:
    """
    Returns list of (img_tile, mask_tile, (row, col)).
    Assumes image and mask are same size.
    """
    if image.size != mask.size:
        raise ValueError(f"Image and mask size mismatch: {image.size} vs {mask.size}")

    width, height = image.size  # PIL is (W, H)
    rows, cols = grid

    if height % rows != 0 or width % cols != 0:
        raise ValueError(
            f"Image size {width}x{height} not divisible by grid {rows}x{cols}. "
            f"For 4000x3000 and grid 4 cols x 3 rows, you should be fine."
        )

    tile_w = width // cols
    tile_h = height // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h

            img_t = image.crop((left, upper, right, lower))
            msk_t = mask.crop((left, upper, right, lower))
            tiles.append((img_t, msk_t, (r, c)))

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
    def __init__(self, images_dir: str, masks_dir: str, ice_is_nonzero: bool = True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ice_is_nonzero = ice_is_nonzero

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

        # Image -> float tensor [C,H,W] in [0,1]
        img = Image.open(img_path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # [3,H,W]

        # Mask -> binary tensor [1,H,W] in {0,1}
        m = Image.open(mask_path)
        m_np = np.asarray(m)

        # If mask is RGB, take first channel; if single channel, this is fine.
        if m_np.ndim == 3:
            m_np = m_np[..., 0]

        if self.ice_is_nonzero:
            m_bin = (m_np > 0).astype(np.float32)
        else:
            m_bin = (m_np == 1).astype(np.float32)

        m_t = torch.from_numpy(m_bin).unsqueeze(0)  # [1,H,W]

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

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)  # logits


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
def run_epoch(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    bce = nn.BCEWithLogitsLoss()
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

    # 1) Tile + split + save
    img = load_image(IMAGE_PATH).convert("RGB")
    mask = load_image(MASK_PATH)  # keep original mode

    tiles = tile_image_and_mask(img, mask, grid=(3, 4))  # 12 tiles for 3000x4000
    train_idx, val_idx, test_idx = split_indices(len(tiles), TRAIN_RATIO, VAL_RATIO, seed=SEED)
    save_tiles(tiles, OUT_DIR, train_idx, val_idx, test_idx)

    # 2) Datasets / loaders
    train_ds = SegTileDataset(
        images_dir=os.path.join(OUT_DIR, "train", "images"),
        masks_dir=os.path.join(OUT_DIR, "train", "masks"),
        ice_is_nonzero=ICE_IS_NONZERO,
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

    # 3) Model (CPU)
    device = "cpu"
    model = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4) Train
    best_val_dice = -1.0
    best_path = os.path.join(OUT_DIR, "best_unet.pt")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        va_loss, va_dice = run_epoch(model, val_loader, optimizer=None, device=device)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_path)

    print(f"Best val dice: {best_val_dice:.4f}")
    print(f"Saved best model to: {best_path}")

    # 5) Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_dice = run_epoch(model, test_loader, optimizer=None, device=device)
    print(f"Test loss {te_loss:.4f} | Test dice {te_dice:.4f}")

    
    # Visualize predictions on test set
    show_test_predictions(model, test_loader, device=device, threshold=0.5, max_batches=2)


if __name__ == "__main__":
    main()
