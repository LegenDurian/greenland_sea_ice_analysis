"""
batch_infer_visible_multiscale.py — Multi-scale visible image batch inference
=============================================================================
Runs UNet inference (with a pretrained ResNet18 encoder from
segmentation-models-pytorch) on all *_V.JPG visible images found inside
data/batch_input/ subfolders.

Usage
-----
    python scripts/batch_infer_visible_multiscale.py

Configuration (edit variables at the top of this file)
------------------------------------------------------
    INPUT_FOLDER         Subfolder name inside data/batch_input/, or None for all
    WEIGHTS_PATH         Path to .pt weights (default: best_unet_pretrained.pt)
    TILE_SIZE            Inference tile size in pixels (default 750)
    STRIDE               Overlap stride (default 375)
    USE_CLAHE            Apply CLAHE preprocessing — must match training (default True)
    DRONE_HEIGHT_M       Flight altitude in metres (change per flight)
    FRACTAL_MIN_AREA_M2  Min floe area (m^2) for fractal regression (default 1.0)

Outputs (per image)
-------------------
    outputs/results/batch_multiscale/{folder}/{image_stem}/
        mask.png              — binary ice/sea mask
        boundary_overlay.png  — original image with yellow floe boundaries
        floe_size_hist.png    — histogram of floe areas (m^2)
        pa_scaling.png        — log-log P-A plot with fractal dimension D
        floe_labels.npy       — 2D int32 label array (0=sea, 1..N=floe ID)
        floe_stats.json       — per-floe shape metrics + fractal dimension

Dependencies
------------
    torch, numpy, Pillow, matplotlib, opencv-python, segmentation-models-pytorch
"""

import sys
from pathlib import Path
import time
import json

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from binary_mask import FloeSeparator, characterize_floes, compute_fractal_dimension, measure_raw_contours


# ----------------------------
# Config
# ----------------------------
_ROOT      = Path(__file__).parent.parent
INPUT_ROOT = _ROOT / "data" / "batch_input"

# Set to a specific subfolder name to process one folder, or None for all.
INPUT_FOLDER = "more_visible_test"

WEIGHTS_PATH = _ROOT / "outputs" / "dataset_tiles_multiscale" / "best_unet_pretrained.pt"

TILE_SIZE = 750
STRIDE    = 375

# CLAHE preprocessing — must match training settings in unet_multiscale.py
USE_CLAHE  = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Inference scales. Single scale [1.0] is sufficient when training at fixed scale.
# Set to [1.0, 0.5, 0.25] to re-enable multi-scale inference.
INFERENCE_SCALES = [1.0]

# GSD — update DRONE_HEIGHT_M per flight.
DRONE_HEIGHT_M = 92.35
PIXEL_PITCH_M  = 1.2e-5
FOCAL_LENGTH_M = 9.1e-3
GSD            = DRONE_HEIGHT_M * PIXEL_PITCH_M / FOCAL_LENGTH_M

OUTPUT_ROOT         = _ROOT / "outputs" / "results" / "batch_multiscale"
THRESHOLD           = 0.5
FRACTAL_MIN_AREA_M2 = 1.0


# ----------------------------
# CLAHE preprocessing
# ----------------------------
def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel (LAB colorspace). Returns uint8 RGB."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def run_inference(model, img_np, tile_size, stride, device):
    model.eval()
    H, W  = img_np.shape[:2]
    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for upper in range(0, H - tile_size + 1, stride):
        for left in range(0, W - tile_size + 1, stride):
            tile = img_np[upper:upper+tile_size, left:left+tile_size]
            x    = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            accum[upper:upper+tile_size, left:left+tile_size] += prob
            count[upper:upper+tile_size, left:left+tile_size] += 1.0
    return accum / np.maximum(count, 1.0)


@torch.no_grad()
def run_inference_multiscale(model, img_np, tile_size, stride, device, scales):
    H, W     = img_np.shape[:2]
    prob_sum = np.zeros((H, W), dtype=np.float32)
    for scale in scales:
        if scale == 1.0:
            scaled = img_np
        else:
            new_H = max(tile_size, int(round(H * scale)))
            new_W = max(tile_size, int(round(W * scale)))
            scaled = np.asarray(
                Image.fromarray((img_np * 255).astype(np.uint8)).resize((new_W, new_H), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0

        prob_scaled = run_inference(model, scaled, tile_size, stride, device)

        if scale != 1.0:
            prob_full = np.asarray(
                Image.fromarray(prob_scaled).resize((W, H), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        else:
            prob_full = prob_scaled

        prob_sum += prob_full
    return prob_sum / len(scales)


# ----------------------------
# Save outputs (same as batch_infer.py)
# ----------------------------
def save_outputs(img_np, mask, out_dir, gsd):
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_u8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(str(out_dir / "mask.png"))

    H, W = img_np.shape[:2]
    dpi  = 150
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(img_np)
    ax.contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_dir / "boundary_overlay.png"), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    sep = FloeSeparator.__new__(FloeSeparator)
    sep.img = None
    labels_filtered, areas_px, _ = sep.split_erode_dilate_small(mask_u8, min_area=30)
    n_floes = int(labels_filtered.max())

    np.save(str(out_dir / "floe_labels.npy"), labels_filtered.astype(np.int32))

    areas_m2 = [a * gsd ** 2 for a in areas_px]

    fig, ax = plt.subplots(figsize=(8, 5))
    if areas_m2:
        ax.hist(areas_m2, bins="auto", edgecolor="k", color="steelblue")
        ax.set_xlabel("Floe area (m²)")
        ax.set_ylabel("Count")
        ax.set_title(f"Floe size distribution  (N={n_floes},  GSD={gsd:.4f} m/px)")
    else:
        ax.text(0.5, 0.5, "No floes detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Floe size distribution")
    fig.tight_layout()
    fig.savefig(str(out_dir / "floe_size_hist.png"), dpi=150)
    plt.close(fig)

    shape_records = characterize_floes(labels_filtered)
    for rec in shape_records:
        rec["area_m2"]      = round(rec["area_px"]       * gsd ** 2, 6)
        rec["perimeter_m"]  = round(rec["perimeter_px"]  * gsd,      4)
        rec["semi_major_m"] = round(rec["semi_major_px"] * gsd,      4)
        rec["semi_minor_m"] = round(rec["semi_minor_px"] * gsd,      4)
        del rec["area_px"], rec["perimeter_px"], rec["semi_major_px"], rec["semi_minor_px"]

    dil_areas  = np.array([r["area_m2"]     for r in shape_records])
    dil_perims = np.array([r["perimeter_m"] for r in shape_records])
    dil_keep   = dil_areas >= FRACTAL_MIN_AREA_M2
    fractal    = compute_fractal_dimension(dil_areas[dil_keep], dil_perims[dil_keep])

    raw_areas_px, raw_perims_px = measure_raw_contours(mask_u8, min_area_px=30)
    raw_areas  = np.array([a * gsd ** 2 for a in raw_areas_px])
    raw_perims = np.array([p * gsd      for p in raw_perims_px])
    raw_keep   = raw_areas >= FRACTAL_MIN_AREA_M2
    fractal_raw = compute_fractal_dimension(raw_areas[raw_keep], raw_perims[raw_keep])

    log_thresh = np.log(FRACTAL_MIN_AREA_M2)
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(dil_areas) > 0:
        ax.scatter(np.log(dil_areas[~dil_keep]), np.log(dil_perims[~dil_keep]), s=6, alpha=0.2, color="steelblue")
        ax.scatter(np.log(dil_areas[dil_keep]),  np.log(dil_perims[dil_keep]),  s=8, alpha=0.5, color="steelblue", label="dilated (fitted)")
    if fractal["valid"]:
        logA = np.linspace(np.log(dil_areas[dil_keep].min()), np.log(dil_areas[dil_keep].max()), 200)
        ax.plot(logA, fractal["slope"] * logA + fractal["intercept"], color="steelblue", linewidth=1.5,
                label=f"dilated D={fractal['D']:.3f} [{fractal['D_ci_low']:.3f}, {fractal['D_ci_high']:.3f}]")
    if len(raw_areas) > 0:
        ax.scatter(np.log(raw_areas[~raw_keep]), np.log(raw_perims[~raw_keep]), s=6, alpha=0.2, color="orangered")
        ax.scatter(np.log(raw_areas[raw_keep]),  np.log(raw_perims[raw_keep]),  s=8, alpha=0.5, color="orangered", label="raw (fitted)")
    if fractal_raw["valid"]:
        logA = np.linspace(np.log(raw_areas[raw_keep].min()), np.log(raw_areas[raw_keep].max()), 200)
        ax.plot(logA, fractal_raw["slope"] * logA + fractal_raw["intercept"], color="orangered", linewidth=1.5,
                label=f"raw D={fractal_raw['D']:.3f} [{fractal_raw['D_ci_low']:.3f}, {fractal_raw['D_ci_high']:.3f}]")
    ax.axvline(log_thresh, color="gray", linewidth=0.8, linestyle="--", label=f"min area = {FRACTAL_MIN_AREA_M2} m²")
    ax.set_xlabel("ln(Area [m²])")
    ax.set_ylabel("ln(Perimeter [m])")
    ax.set_title(f"P–A scaling  (dilated N={fractal.get('n_floes',0)},  raw N={fractal_raw.get('n_floes',0)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_dir / "pa_scaling.png"), dpi=150)
    plt.close(fig)

    meta = {
        "gsd_m_per_px":          round(gsd, 6),
        "drone_height_m":        DRONE_HEIGHT_M,
        "n_floes":               n_floes,
        "fractal_dimension":     fractal,
        "fractal_dimension_raw": fractal_raw,
    }
    with open(str(out_dir / "floe_stats.json"), "w") as f:
        json.dump({"meta": meta, "floes": shape_records}, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main():
    if not WEIGHTS_PATH.exists():
        print(f"Weights not found: {WEIGHTS_PATH}")
        print("Train the model first:  python scripts/training/unet_multiscale.py")
        return

    if INPUT_FOLDER is not None:
        subfolders = [INPUT_ROOT / INPUT_FOLDER]
    else:
        subfolders = sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir())

    if not subfolders:
        print(f"No subfolders found in {INPUT_ROOT}")
        return

    device = "cpu"
    print(f"Loading weights from {WEIGHTS_PATH} ...")
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None).to(device)
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    model.eval()
    print(f"Inference scales: {INFERENCE_SCALES}\n")

    for subfolder in subfolders:
        # Only process visible files (_V.JPG / _V.jpg)
        images = sorted(
            p for p in subfolder.iterdir()
            if p.stem.upper().endswith("_V") and p.suffix.lower() in {".jpg", ".jpeg"}
        )
        if not images:
            print(f"[{subfolder.name}] No *_V.JPG files found — skipping.")
            continue

        print(f"[{subfolder.name}] {len(images)} visible image(s).")
        total = len(images)

        for i, img_path in enumerate(images, 1):
            with Image.open(img_path) as im:
                W, H = im.size
            print(f"  [{i}/{total}] {img_path.name}  ({W}×{H})", flush=True)

            if H < TILE_SIZE or W < TILE_SIZE:
                print(f"    [SKIP] image smaller than tile size ({TILE_SIZE}px)")
                continue

            t0 = time.perf_counter()

            img_u8 = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            if USE_CLAHE:
                img_u8 = apply_clahe(img_u8)
            img_np = img_u8.astype(np.float32) / 255.0
            prob   = run_inference_multiscale(model, img_np, TILE_SIZE, STRIDE, device, INFERENCE_SCALES)
            mask   = (prob > THRESHOLD).astype(np.float32)

            elapsed = time.perf_counter() - t0
            print(f"    {elapsed:.1f}s  |  est. remaining: {(total - i) * elapsed / 60:.1f} min", flush=True)

            out_dir = OUTPUT_ROOT / subfolder.name / img_path.stem
            save_outputs(img_np, mask, out_dir, GSD)
            print(f"    Saved → {out_dir.relative_to(_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
