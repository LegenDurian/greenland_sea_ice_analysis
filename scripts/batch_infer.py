"""
batch_infer.py
--------------
Drop a folder of images into data/batch_input/ and run this script.
Image type (thermal vs visible) is detected automatically by image dimensions:
  - thermal:  640x512  (< 1 MP)
  - visible: 4000x3000 (> 1 MP)

Outputs go to:
  outputs/batch_results/{input_folder_name}/{thermal|visible}_{image_stem}/
    mask.png              — binary mask (white=ice)
    boundary_overlay.png  — original image with yellow floe boundaries
    floe_size_hist.png    — histogram of individual floe areas (px²)
"""

import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json

sys.path.insert(0, str(Path(__file__).parent))
from binary_mask import FloeSeparator, characterize_floes, compute_fractal_dimension, measure_raw_contours


# ----------------------------
# Config
# ----------------------------

# "unet"        — use trained UNet weights (thermal + visible)
# "binary_mask" — use CLAHE + threshold from binary_mask.py (thermal only)
METHOD = "binary_mask"
METHOD = "unet"

# Name of the folder inside data/batch_input/ to process.
# Set to None to process ALL subfolders.
INPUT_FOLDER = "DJI_202502080715_002_Transect1Station17"

_ROOT      = Path(__file__).parent.parent
INPUT_ROOT = _ROOT / "data" / "batch_input"

WEIGHTS = {
    "visible": _ROOT / "outputs" / "dataset_tiles"         / "best_unet.pt",
    "thermal": _ROOT / "outputs" / "dataset_tiles_thermal" / "best_unet.pt",
}

TILE_CFG = {
    "visible": {"tile_size": 500, "stride": 250},
    "thermal": {"tile_size": 128, "stride": 64},
}

# binary_mask preprocessing params (only used when METHOD = "binary_mask")
BM_CLAHE_CLIP = 1.5
BM_CLAHE_GRID = (8, 8)
BM_THRESH_VAL = 75

# ----------------------------
# GSD (Ground Sampling Distance)
# GSD (m/pixel) = drone_height * pixel_pitch / focal_length
# Camera constants are fixed; change DRONE_HEIGHT_M per flight.
# ----------------------------
DRONE_HEIGHT_M = 92.35    # flight altitude in metres  ← change this per flight
PIXEL_PITCH_M  = 1.2e-5   # sensor pixel pitch (m)     — fixed for this camera
FOCAL_LENGTH_M = 9.1e-3   # focal length (m)            — fixed for this camera
GSD            = DRONE_HEIGHT_M * PIXEL_PITCH_M / FOCAL_LENGTH_M  # m/pixel

OUTPUT_ROOT  = _ROOT / "outputs" / "results" / "batch"
THRESHOLD    = 0.5

# Multi-scale inference (unet method only).
# Runs the sliding window at each scale, upsamples each probability map back
# to the original resolution, then averages them.  This helps detect large
# floes whose boundaries fall outside a single 500 px tile at full resolution.
# Only effective when using weights from unet_multiscale.py — a model trained
# at a single scale will produce unreliable predictions at other scales.
MULTISCALE_INFERENCE = False
INFERENCE_SCALES     = [1.0, 0.5, 0.25]   # 1.0 = full res; 0.5/0.25 = zoom-out

# Minimum floe area (m²) included in the fractal dimension P–A regression.
# Small floes (few pixels) produce noisy perimeter estimates that bias D toward 1.
# At GSD ≈ 0.12 m/px, 1.0 m² ≈ 70 px².  Increase if D still looks too low.
FRACTAL_MIN_AREA_M2 = 1.0
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
THERMAL_MPIX = 1_000_000   # images below this pixel count are treated as thermal


# ----------------------------
# U-Net
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base);        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2);     self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
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
        return F.interpolate(self.out(d1), size=(h, w), mode="bilinear", align_corners=False)


# ----------------------------
# Sliding-window inference
# ----------------------------
@torch.no_grad()
def run_inference(model, img_np: np.ndarray, tile_size: int, stride: int, device: str) -> np.ndarray:
    """Sliding-window inference at a single scale. Returns a [H,W] probability map."""
    model.eval()
    H, W   = img_np.shape[:2]
    accum  = np.zeros((H, W), dtype=np.float32)
    count  = np.zeros((H, W), dtype=np.float32)

    for upper in range(0, H - tile_size + 1, stride):
        for left in range(0, W - tile_size + 1, stride):
            tile = img_np[upper:upper + tile_size, left:left + tile_size]
            x    = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            accum[upper:upper + tile_size, left:left + tile_size] += prob
            count[upper:upper + tile_size, left:left + tile_size] += 1.0

    return accum / np.maximum(count, 1.0)


@torch.no_grad()
def run_inference_multiscale(
    model, img_np: np.ndarray, tile_size: int, stride: int,
    device: str, scales: list[float],
) -> np.ndarray:
    """
    Run sliding-window inference at multiple zoom-out scales and average the
    resulting probability maps (all upsampled back to original resolution).

    Scale 1.0  — original resolution; good boundary detail on small floes.
    Scale 0.5  — image halved; a tile covers 2× more original content.
    Scale 0.25 — image quartered; a tile covers 4× more original content,
                 allowing large floe boundaries to fit within a single tile.

    Averaging means that regions where all scales agree (e.g. clearly ice or
    clearly water) get reinforced, while scale-specific artefacts are diluted.
    """
    H, W = img_np.shape[:2]
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
            # upsample probability map back to original resolution
            prob_full = np.asarray(
                Image.fromarray(prob_scaled).resize((W, H), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        else:
            prob_full = prob_scaled

        prob_sum += prob_full

    return prob_sum / len(scales)


def detect_mode(img_path: Path) -> str | None:
    """Return 'thermal', 'visible', or None if unrecognised."""
    with Image.open(img_path) as img:
        W, H = img.size
    pixels = W * H
    if pixels < THERMAL_MPIX:
        return "thermal"
    else:
        return "visible"


def save_outputs(img_np: np.ndarray, mask: np.ndarray, out_dir: Path, gsd: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # mask.png
    mask_u8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(str(out_dir / "mask.png"))

    # boundary_overlay.png — rendered at native resolution
    H, W = img_np.shape[:2]
    dpi  = 150
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(img_np)
    ax.contour(mask, levels=[0.5], colors="yellow", linewidths=0.8)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_dir / "boundary_overlay.png"), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # erode/dilate → connected components
    sep = FloeSeparator.__new__(FloeSeparator)   # no image path needed
    sep.img = None
    labels_filtered, areas_px, _ = sep.split_erode_dilate_small(mask_u8, min_area=30)
    n_floes = int(labels_filtered.max())

    # floe_labels.npy — 2D int32 array, pixel value = floe ID (0=sea, 1..N=floe)
    np.save(str(out_dir / "floe_labels.npy"), labels_filtered.astype(np.int32))

    # convert areas to m²
    areas_m2 = [a * gsd ** 2 for a in areas_px]

    # floe_size_hist.png — in m²
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(areas_m2) > 0:
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

    # floe_stats.json — convert px → m
    shape_records = characterize_floes(labels_filtered)
    for rec in shape_records:
        rec["area_m2"]       = round(rec["area_px"]       * gsd ** 2, 6)
        rec["perimeter_m"]   = round(rec["perimeter_px"]  * gsd,      4)
        rec["semi_major_m"]  = round(rec["semi_major_px"] * gsd,      4)
        rec["semi_minor_m"]  = round(rec["semi_minor_px"] * gsd,      4)
        del rec["area_px"], rec["perimeter_px"], rec["semi_major_px"], rec["semi_minor_px"]

    # fractal dimension (dilated boundaries) — only floes >= FRACTAL_MIN_AREA_M2
    dil_areas_all  = np.array([r["area_m2"]     for r in shape_records])
    dil_perims_all = np.array([r["perimeter_m"] for r in shape_records])
    dil_keep = dil_areas_all >= FRACTAL_MIN_AREA_M2
    fractal = compute_fractal_dimension(dil_areas_all[dil_keep], dil_perims_all[dil_keep])

    # fractal dimension (raw mask boundaries — no erode/dilate smoothing)
    raw_areas_px, raw_perims_px = measure_raw_contours(mask_u8, min_area_px=30)
    raw_areas_all = np.array([a * gsd ** 2 for a in raw_areas_px])
    raw_perims_all = np.array([p * gsd      for p in raw_perims_px])
    raw_keep = raw_areas_all >= FRACTAL_MIN_AREA_M2
    fractal_raw = compute_fractal_dimension(raw_areas_all[raw_keep], raw_perims_all[raw_keep])

    # pa_scaling.png — dilated vs raw; faded points excluded from fit
    log_thresh = np.log(FRACTAL_MIN_AREA_M2)
    fig, ax = plt.subplots(figsize=(7, 5))
    # dilated
    ax.scatter(np.log(dil_areas_all[~dil_keep]), np.log(dil_perims_all[~dil_keep]),
               s=6, alpha=0.2, color="steelblue")
    ax.scatter(np.log(dil_areas_all[dil_keep]),  np.log(dil_perims_all[dil_keep]),
               s=8, alpha=0.5, color="steelblue", label="dilated floes (fitted)")
    if fractal["valid"]:
        logA_fit = np.linspace(np.log(dil_areas_all[dil_keep].min()),
                               np.log(dil_areas_all[dil_keep].max()), 200)
        ax.plot(logA_fit, fractal["slope"] * logA_fit + fractal["intercept"],
                color="steelblue", linewidth=1.5,
                label=f"dilated  D={fractal['D']:.3f} [{fractal['D_ci_low']:.3f}, {fractal['D_ci_high']:.3f}]")
    # raw
    if len(raw_areas_all) > 0:
        ax.scatter(np.log(raw_areas_all[~raw_keep]), np.log(raw_perims_all[~raw_keep]),
                   s=6, alpha=0.2, color="orangered")
        ax.scatter(np.log(raw_areas_all[raw_keep]),  np.log(raw_perims_all[raw_keep]),
                   s=8, alpha=0.5, color="orangered", label="raw floes (fitted)")
    if fractal_raw["valid"]:
        logA_fit = np.linspace(np.log(raw_areas_all[raw_keep].min()),
                               np.log(raw_areas_all[raw_keep].max()), 200)
        ax.plot(logA_fit, fractal_raw["slope"] * logA_fit + fractal_raw["intercept"],
                color="orangered", linewidth=1.5,
                label=f"raw  D={fractal_raw['D']:.3f} [{fractal_raw['D_ci_low']:.3f}, {fractal_raw['D_ci_high']:.3f}]")
    ax.axvline(log_thresh, color="gray", linewidth=0.8, linestyle="--",
               label=f"min area = {FRACTAL_MIN_AREA_M2} m²")
    ax.set_xlabel("ln(Area  [m²])")
    ax.set_ylabel("ln(Perimeter  [m])")
    n_dil = fractal["n_floes"] if fractal["valid"] else 0
    n_raw = fractal_raw["n_floes"] if fractal_raw["valid"] else 0
    ax.set_title(f"P–A scaling  (dilated N={n_dil},  raw N={n_raw},  min={FRACTAL_MIN_AREA_M2} m²)")
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


def main():
    if METHOD not in ("unet", "binary_mask"):
        raise ValueError(f"METHOD must be 'unet' or 'binary_mask', got '{METHOD}'")

    print(f"Method: {METHOD}")

    # Collect input subfolders
    if INPUT_FOLDER is not None:
        target = INPUT_ROOT / INPUT_FOLDER
        if not target.is_dir():
            print(f"Folder not found: {target}")
            return
        subfolders = [target]
    else:
        subfolders = sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir())

    if not subfolders:
        print(f"No subfolders found in {INPUT_ROOT}. Drop a folder of images there and re-run.")
        return

    device = "cpu"
    models = {}   # cache loaded UNet models (only used when METHOD = "unet")

    for subfolder in sorted(subfolders):
        images = sorted(p for p in subfolder.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS and p.stem.upper().endswith("_V"))
        if not images:
            print(f"[{subfolder.name}] No images found — skipping.")
            continue

        print(f"\n[{subfolder.name}] {len(images)} image(s) found.")

        total = len(images)
        for i, img_path in enumerate(images, 1):
            mode = detect_mode(img_path)

            with Image.open(img_path) as img:
                W, H = img.size
            print(f"  [{i}/{total}] {img_path.name}  ({W}x{H}) → {mode}", flush=True)

            if mode is None:
                print(f"    [SKIP] could not determine mode from image size ({W}x{H})")
                continue

            # binary_mask method only supports thermal
            if METHOD == "binary_mask" and mode != "thermal":
                print("    [SKIP] binary_mask method is thermal-only")
                continue

            t0 = time.perf_counter()

            if METHOD == "unet":
                if not WEIGHTS[mode].exists():
                    print(f"    [SKIP] weights not found at {WEIGHTS[mode]}")
                    continue

                tile_size = TILE_CFG[mode]["tile_size"]
                if H < tile_size or W < tile_size:
                    print(f"    [SKIP] image smaller than tile size ({tile_size}px)")
                    continue

                if mode not in models:
                    print(f"    Loading {mode} weights...")
                    m = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
                    m.load_state_dict(torch.load(str(WEIGHTS[mode]), map_location=device))
                    models[mode] = m

                img_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
                stride = TILE_CFG[mode]["stride"]
                if MULTISCALE_INFERENCE and mode == "visible":
                    prob = run_inference_multiscale(models[mode], img_np, tile_size, stride, device, INFERENCE_SCALES)
                else:
                    prob = run_inference(models[mode], img_np, tile_size, stride, device)
                mask   = (prob > THRESHOLD).astype(np.float32)

            else:  # binary_mask
                sep = FloeSeparator(str(img_path))
                _, img_rgb, bm_mask = sep.preprocess(
                    clahe_clip=BM_CLAHE_CLIP,
                    clahe_grid=BM_CLAHE_GRID,
                    thresh_val=BM_THRESH_VAL,
                )
                img_np = img_rgb.astype(np.float32) / 255.0
                mask   = (bm_mask > 0).astype(np.float32)

            elapsed   = time.perf_counter() - t0
            remaining = (total - i) * elapsed
            print(f"    processed: {elapsed:.1f}s  |  est. remaining: {remaining/60:.1f} min", flush=True)

            out_dir = OUTPUT_ROOT / subfolder.name / f"{mode}_{img_path.stem}"
            save_outputs(img_np, mask, out_dir, GSD)
            print(f"    Saved → {out_dir.relative_to(_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
