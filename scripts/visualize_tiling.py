"""
visualize_tiling.py — Visualize UNet training tile layout
=========================================================
Shows how the input image is divided into a grid of tiles and color-codes
each tile by its train/val/test assignment. Replicates the exact split
logic from scripts/training/unet.py so you can verify which regions are
used for training vs evaluation.

Usage
-----
    python scripts/visualize_tiling.py

Outputs
-------
    outputs/visible_mask_test/tiling_visualization.png

Dependencies
------------
    numpy, matplotlib, Pillow
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path

# ── Config (must match unet.py) ──────────────────────────────────────────────
TILE_SIZE   = 500
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42

IMG_PATH = Path(__file__).parent.parent / "data" / "images" / "sea_ice.jpg"
OUT_PATH = Path(__file__).parent.parent / "outputs" / "visible_mask_test" / "tiling_visualization.png"

COLORS = {
    "train": (0.22, 0.71, 0.29, 0.45),   # green
    "val":   (1.00, 0.65, 0.00, 0.55),   # orange
    "test":  (0.90, 0.18, 0.18, 0.55),   # red
}
EDGE_COLORS = {
    "train": (0.10, 0.50, 0.15),
    "val":   (0.80, 0.45, 0.00),
    "test":  (0.65, 0.08, 0.08),
}

# ── Replicate split logic from unet.py ───────────────────────────────────────
def split_indices(n, train_ratio, val_ratio, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

# ── Main ─────────────────────────────────────────────────────────────────────
img = Image.open(IMG_PATH)
W, H = img.size          # 4000, 3000
cols = W // TILE_SIZE    # 8
rows = H // TILE_SIZE    # 6
n_tiles = cols * rows    # 48

train_idx, val_idx, test_idx = split_indices(n_tiles, TRAIN_RATIO, VAL_RATIO, SEED)
train_set = set(train_idx)
val_set   = set(val_idx)

def get_split(i):
    if i in train_set: return "train"
    if i in val_set:   return "val"
    return "test"

tile_splits = [get_split(i) for i in range(n_tiles)]
counts = {s: tile_splits.count(s) for s in ("train", "val", "test")}

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(img)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect('equal')
ax.axis('off')

for i, split in enumerate(tile_splits):
    r = i // cols
    c = i % cols
    x = c * TILE_SIZE
    y = r * TILE_SIZE
    fc = COLORS[split]
    ec = EDGE_COLORS[split]
    rect = mpatches.FancyBboxPatch(
        (x, y), TILE_SIZE, TILE_SIZE,
        boxstyle="square,pad=0",
        facecolor=fc,
        edgecolor=ec,
        linewidth=2.0,
    )
    ax.add_patch(rect)
    # Tile label
    ax.text(
        x + TILE_SIZE / 2, y + TILE_SIZE / 2,
        split[0].upper(),          # T / V / Te
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color='white',
        path_effects=[
            __import__('matplotlib.patheffects', fromlist=['withStroke'])
            .withStroke(linewidth=3, foreground='black')
        ],
    )

# ── Grid lines ───────────────────────────────────────────────────────────────
for c in range(cols + 1):
    ax.axvline(c * TILE_SIZE, color='white', linewidth=0.8, alpha=0.6)
for r in range(rows + 1):
    ax.axhline(r * TILE_SIZE, color='white', linewidth=0.8, alpha=0.6)

# ── Legend & title ───────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=COLORS["train"], edgecolor=EDGE_COLORS["train"],
                   label=f'Train  ({counts["train"]} tiles, {counts["train"]/n_tiles*100:.0f}%)'),
    mpatches.Patch(facecolor=COLORS["val"],   edgecolor=EDGE_COLORS["val"],
                   label=f'Val    ({counts["val"]}  tiles, {counts["val"]/n_tiles*100:.0f}%)'),
    mpatches.Patch(facecolor=COLORS["test"],  edgecolor=EDGE_COLORS["test"],
                   label=f'Test   ({counts["test"]}  tiles, {counts["test"]/n_tiles*100:.0f}%)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=12,
          framealpha=0.85, edgecolor='gray')

ax.set_title(
    f'Dataset tiling — {cols}×{rows} grid of {TILE_SIZE}×{TILE_SIZE} px tiles  '
    f'({n_tiles} total)  |  seed={SEED}',
    fontsize=13, pad=10,
)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_PATH}")
