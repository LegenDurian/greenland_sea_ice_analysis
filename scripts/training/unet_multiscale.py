"""
unet_multiscale.py
------------------
UNet training for visible-light sea ice images.

Key differences from unet.py
──────────────────────────────
  No pre-tiling to disk — crops are sampled on-the-fly each epoch, so the
  model sees a different random subset of the image every epoch.

  SAMPLES_PER_EPOCH controls how many random crops are drawn per epoch.
  Decrease it to reduce training cost; increase it to see more of the image.

  All image/mask pairs in data/images/training/ are used automatically.
  Pairs are discovered by matching  {stem}.jpg  →  {stem}_mask.png.

  Train / val split:
    Spatial strip per image — top (1-VAL_FRAC) of each image is train,
    bottom VAL_FRAC is val. All conditions appear in both sets.

Outputs:
    outputs/dataset_tiles_multiscale/best_unet.pt   — best weights (val Dice)
    outputs/dataset_tiles_multiscale/train_curve.png
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
_ROOT     = Path(__file__).parent.parent.parent
_DATA_DIR = _ROOT / "data" / "images" / "training"
OUT_DIR   = _ROOT / "outputs" / "dataset_tiles_multiscale"

CROP_SIZE         = 750    # input tile size fed to the UNet (px)
SAMPLES_PER_EPOCH = 150    # random crops drawn per epoch
VAL_STRIDE        = 750    # stride for deterministic val grid (non-overlapping by default)

# CLAHE preprocessing — normalises local contrast before training/inference.
# Helps the model generalise across images with different lighting/colour.
USE_CLAHE      = True
CLAHE_CLIP     = 2.0
CLAHE_GRID     = (8, 8)

VAL_FRAC   = 0.30   # fraction used for val — must give strip height >= CROP_SIZE

SEED       = 42
BATCH_SIZE = 4
EPOCHS     = 75
LR         = 1e-3

ICE_IS_NONZERO = True   # treat any nonzero mask pixel as ice

# Allowlist of image stems to use for training (case-insensitive).
# Set to None to use all visible (non-thermal) pairs found in _DATA_DIR.
TRAIN_STEMS = None


# ----------------------------
# Pair discovery
# ----------------------------
_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def discover_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (image, mask) pairs: {stem}.* and {stem}_mask.png."""
    pairs = []
    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() not in _EXTS:
            continue
        if img_path.stem.endswith("_mask"):
            continue
        if "thermal" in img_path.stem.lower():
            continue
        if TRAIN_STEMS is not None and img_path.stem.lower() not in {s.lower() for s in TRAIN_STEMS}:
            continue
        mask_path = data_dir / (img_path.stem + "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found in {data_dir}.\n"
            "Expected: {{stem}}.jpg alongside {{stem}}_mask.png"
        )
    return pairs


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB colorspace. Returns uint8 RGB."""
    lab   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_pair(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load image as uint8 RGB [H,W,3] and mask as uint8 binary [H,W]."""
    img_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    if USE_CLAHE:
        img_np = apply_clahe(img_np)
    m_raw  = np.asarray(Image.open(mask_path))
    if m_raw.ndim == 3:
        m_raw = m_raw.max(axis=-1)
    mask_np = ((m_raw > 0) if ICE_IS_NONZERO else (m_raw == 1)).astype(np.uint8) * 255
    return img_np, mask_np


# ----------------------------
# Datasets
# ----------------------------
def _random_scale_crop(
    img_np: np.ndarray,
    mask_np: np.ndarray,
    scale: float,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale image+mask then take a random spatial crop."""
    H, W = img_np.shape[:2]
    new_H = max(crop_size, int(round(H * scale)))
    new_W = max(crop_size, int(round(W * scale)))

    img_pil  = Image.fromarray(img_np).resize((new_W, new_H), Image.Resampling.BILINEAR)
    mask_pil = Image.fromarray(mask_np).resize((new_W, new_H), Image.Resampling.NEAREST)

    top  = random.randint(0, new_H - crop_size)
    left = random.randint(0, new_W - crop_size)

    img_crop  = np.asarray(img_pil.crop( (left, top, left + crop_size, top + crop_size)), dtype=np.float32) / 255.0
    mask_crop = np.asarray(mask_pil.crop((left, top, left + crop_size, top + crop_size)), dtype=np.float32) / 255.0
    return img_crop, mask_crop


class ScaleAugDataset(Dataset):
    """
    On-the-fly random crop + scale augmentation.

    Each __getitem__:
      1. Picks a random image/mask pair from self.pairs.
      2. Randomly rescales by scale ∈ [scale_min, scale_max].
      3. Takes a random crop of crop_size × crop_size.
      4. Applies spatial and colour augmentation.
    """

    def __init__(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        crop_size: int = CROP_SIZE,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        augment: bool = True,
    ):
        if not pairs:
            raise RuntimeError("ScaleAugDataset: no image/mask pairs provided.")
        self.pairs             = pairs
        self.crop_size         = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.augment           = augment

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        img_np, mask_np = random.choice(self.pairs)
        img_crop, mask_crop = _random_scale_crop(img_np, mask_np, 1.0, self.crop_size)

        if self.augment:
            if random.random() < 0.5:
                img_crop  = np.fliplr(img_crop)
                mask_crop = np.fliplr(mask_crop)
            if random.random() < 0.5:
                img_crop  = np.flipud(img_crop)
                mask_crop = np.flipud(mask_crop)
            k = random.randint(0, 3)
            if k:
                img_crop  = np.rot90(img_crop,  k)
                mask_crop = np.rot90(mask_crop, k)
            # Brightness + contrast jitter (wider range for cross-condition generalization)
            img_crop = np.clip(img_crop * random.uniform(0.6, 1.4), 0.0, 1.0)
            mean     = img_crop.mean()
            img_crop = np.clip((img_crop - mean) * random.uniform(0.6, 1.4) + mean, 0.0, 1.0)
            # Random grayscale (30% chance) — forces model to work without color cues
            if random.random() < 0.3:
                gray     = img_crop.mean(axis=2, keepdims=True)
                img_crop = np.repeat(gray, 3, axis=2)

        img_t  = torch.from_numpy(np.ascontiguousarray(img_crop)).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray((mask_crop > 0.5).astype(np.float32))).unsqueeze(0)
        return img_t, mask_t


class GridCropDataset(Dataset):
    """
    Deterministic sliding-window crops at scale=1.0. Used for validation.
    No augmentation.
    """

    def __init__(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        crop_size: int = CROP_SIZE,
        stride: int = VAL_STRIDE,
    ):
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for img_np, mask_np in pairs:
            H, W = img_np.shape[:2]
            for top in range(0, H - crop_size + 1, stride):
                for left in range(0, W - crop_size + 1, stride):
                    img_c  = img_np[top:top+crop_size,  left:left+crop_size].astype(np.float32) / 255.0
                    mask_c = (mask_np[top:top+crop_size, left:left+crop_size] > 127).astype(np.float32)
                    self.items.append((img_c, mask_c))
        if not self.items:
            raise RuntimeError("GridCropDataset: no crops produced — val region may be too small for CROP_SIZE.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_c, mask_c = self.items[idx]
        img_t  = torch.from_numpy(np.ascontiguousarray(img_c)).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_c)).unsqueeze(0)
        return img_t, mask_t


# ----------------------------
# UNet (identical to unet.py)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base);        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2);       self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4);     self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out  = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _crop(skip, x): return skip[:, :, :x.shape[2], :x.shape[3]]

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
# Loss + metrics
# ----------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (1.0 - num / den).mean()

@torch.no_grad()
def dice_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    num = 2.0 * (preds * targets).sum(dim=(2, 3))
    den = (preds + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean().item()


# ----------------------------
# Train / eval loops
# ----------------------------
def run_epoch(model, loader, optimizer=None, device="cpu", pos_weight=None):
    train = optimizer is not None
    model.train(train)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    total_loss, total_dice, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_dice += dice_score(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_dice / n


# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Discover and load all pairs into memory
    pairs_paths = discover_pairs(_DATA_DIR)
    print(f"Found {len(pairs_paths)} image/mask pair(s):")
    for ip, mp in pairs_paths:
        print(f"  {ip.name}  +  {mp.name}")

    loaded = [load_pair(ip, mp) for ip, mp in pairs_paths]

    # 2) Train / val split — spatial strip per image so all conditions appear in both sets
    train_pairs, val_pairs = [], []
    for img_np, mask_np in loaded:
        split_row = int(img_np.shape[0] * (1.0 - VAL_FRAC))
        train_pairs.append((img_np[:split_row],  mask_np[:split_row]))
        val_pairs.append(  (img_np[split_row:],  mask_np[split_row:]))
    print(f"\nSpatial strip split (per image): top {int((1-VAL_FRAC)*100)}% train, "
          f"bottom {int(VAL_FRAC*100)}% val — across all {len(loaded)} image(s).")

    # 3) Datasets + loaders
    train_ds = ScaleAugDataset(train_pairs, augment=True)
    val_ds   = GridCropDataset(val_pairs)

    print(f"Train: {SAMPLES_PER_EPOCH} random crops/epoch  (crop {CROP_SIZE}px)")
    print(f"Val:   {len(val_ds)} grid crops")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4) Class-balance weight from training pairs
    ice_px, bg_px = 0, 0
    for _, mask_np in train_pairs:
        ice_px += int((mask_np > 127).sum())
        bg_px  += int((mask_np == 0).sum())
    pos_weight = torch.tensor([bg_px / max(ice_px, 1)], dtype=torch.float32)
    print(f"Class balance — ice: {ice_px:,}, bg: {bg_px:,}, pos_weight: {pos_weight.item():.2f}\n")

    # 5) Model
    device = "cpu"
    model     = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 6) Training loop
    best_val_dice = -1.0
    best_path = OUT_DIR / "best_unet.pt"
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer=optimizer, device=device, pos_weight=pos_weight)
        va_loss, va_dice = run_epoch(model, val_loader,   optimizer=None,      device=device, pos_weight=pos_weight)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(va_loss)
        history["val_dice"].append(va_dice)

        marker = " *" if va_dice > best_val_dice else ""
        print(f"Epoch {epoch:03d} | lr {scheduler.get_last_lr()[0]:.2e} "
              f"| train loss {tr_loss:.4f} dice {tr_dice:.4f} "
              f"| val loss {va_loss:.4f} dice {va_dice:.4f}{marker}", flush=True)

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), str(best_path))

    print(f"\nBest val dice: {best_val_dice:.4f}")
    print(f"Saved best weights → {best_path}")

    # 7) Training curve
    epochs = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["train_dice"], label="train")
    ax2.plot(epochs, history["val_dice"],   label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice"); ax2.set_title("Dice"); ax2.legend()
    fig.suptitle(f"unet_multiscale  crop={CROP_SIZE}px  samples/epoch={SAMPLES_PER_EPOCH}")
    fig.tight_layout()
    curve_path = OUT_DIR / "train_curve.png"
    fig.savefig(str(curve_path), dpi=150)
    plt.close(fig)
    print(f"Saved training curve → {curve_path}")

    # 8) Use new weights in batch_infer.py:
    #    Change WEIGHTS["visible"] to point to:
    print(f"\nTo use these weights in batch_infer.py, set:")
    print(f'  WEIGHTS["visible"] = Path("{best_path}")')


if __name__ == "__main__":
    main()
