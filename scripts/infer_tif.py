"""
infer_tif.py — Thermal TIF processing pipeline
================================================
Applies the binary_mask thresholding pipeline to raw thermal .tif files in
data/thermal_tif/. The .tif files contain float32 temperature values in deg C.

Usage
-----
    python scripts/infer_tif.py

Configuration (edit variables at the top of this file)
------------------------------------------------------
    INPUT_FILE           Specific TIF stem to process, or None for all
    ROTATE_180           Rotate image 180 deg before processing (default True)
    DRONE_HEIGHT_M       Flight altitude in metres (change per flight)
    BM_THRESH_VAL        Binary threshold 0-255 (default 75)
    FRACTAL_MIN_AREA_M2  Min floe area (m^2) for fractal regression (default 1.0)

Outputs (per TIF)
-----------------
    outputs/results/tif/{stem}/
        mask.png              — binary ice/sea mask
        boundary_overlay.png  — inferno thermal image with yellow floe boundaries
        floe_size_hist.png    — floe area distribution (m^2)
        pa_scaling.png        — log-log P-A plot with fractal dimension D
        floe_labels.npy       — 2D int32 label array (0=sea, 1..N=floe ID)
        floe_stats.json       — per-floe shape + temperature metrics
        pixel_temp_hist.png   — histogram of all pixel temperatures (deg C)
        floe_temp_hist.png    — histogram of per-floe average temperatures (deg C)

Dependencies
------------
    opencv-python, numpy, Pillow, matplotlib
"""

import sys
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from binary_mask import FloeSeparator, characterize_floes, compute_average_temps, compute_fractal_dimension, measure_raw_contours


# ----------------------------
# Config
# ----------------------------
_ROOT      = Path(__file__).parent.parent
INPUT_DIR  = _ROOT / "data" / "thermal_tif"
OUTPUT_ROOT = _ROOT / "outputs" / "results" / "tif"

# Set to a specific stem (e.g. "DJI_20250218_frame01") to process one file,
# or None to process all .tif files in INPUT_DIR.
INPUT_FILE = None

# Rotate TIF 180° before processing (matches binary_mask.py behaviour)
ROTATE_180 = True

# Binary mask preprocessing params
BM_CLAHE_CLIP = 1.5
BM_CLAHE_GRID = (8, 8)
BM_THRESH_VAL = 75
BM_MIN_AREA   = 30

# Minimum floe area (m²) included in the fractal dimension P–A regression.
# Small floes (few pixels) produce noisy perimeter estimates that bias D toward 1.
# At GSD ≈ 0.12 m/px, 1.0 m² ≈ 70 px².  Increase if D still looks too low.
FRACTAL_MIN_AREA_M2 = 1.0

# GSD — same camera as batch_infer.py
DRONE_HEIGHT_M = 92.35
PIXEL_PITCH_M  = 1.2e-5
FOCAL_LENGTH_M = 9.1e-3
GSD            = DRONE_HEIGHT_M * PIXEL_PITCH_M / FOCAL_LENGTH_M  # m/pixel


# ----------------------------
# Helpers
# ----------------------------
def load_tif(path: Path) -> np.ndarray:
    """Load a thermal .tif into a float32 numpy array (temperature °C)."""
    arr = np.array(Image.open(path), dtype=np.float32)
    print(f"  Loaded {path.name}  shape={arr.shape}  "
          f"min={arr.min():.1f}°C  max={arr.max():.1f}°C")
    return arr


def temp_to_display(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert float temperature array to:
      gray_u8  — uint8 grayscale (for CLAHE + threshold)
      rgb_disp — uint8 RGB rendered with inferno colormap (for overlays)
    """
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)

    gray_u8 = (norm * 255).astype(np.uint8)

    cmap   = plt.get_cmap("inferno")
    rgb_disp = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)   # drop alpha

    return gray_u8, rgb_disp


def run_binary_mask(gray_u8: np.ndarray) -> np.ndarray:
    """
    Run CLAHE + binary inverse threshold on a uint8 grayscale image.
    Returns binary mask (255=ice, 0=sea).
    """
    clahe = cv.createCLAHE(clipLimit=BM_CLAHE_CLIP, tileGridSize=BM_CLAHE_GRID)
    eq    = clahe.apply(gray_u8)
    _, mask = cv.threshold(eq, BM_THRESH_VAL, 255, cv.THRESH_BINARY_INV)
    return mask.astype(np.uint8)


def save_pixel_temp_hist(arr: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr.flatten(), bins=100, color="orangered", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Per-pixel temperature distribution")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_floe_temp_hist(temp_list: list, out_path: Path):
    temps = np.array([t for t in temp_list if not np.isnan(t)])
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(temps) > 0:
        ax.hist(temps, bins="auto", color="steelblue", edgecolor="k", linewidth=0.3)
        ax.set_xlabel("Average temperature (°C)")
        ax.set_ylabel("Floe count")
        ax.set_title(f"Per-floe average temperature distribution  (N={len(temps)})")
    else:
        ax.text(0.5, 0.5, "No floes detected", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Per-floe average temperature distribution")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def process_tif(tif_path: Path):
    out_dir = OUTPUT_ROOT / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load temperature array
    temp_arr = load_tif(tif_path)
    if ROTATE_180:
        temp_arr = np.rot90(temp_arr, 2)

    # 2. Convert to display formats
    gray_u8, rgb_disp = temp_to_display(temp_arr)

    # 3. Binary mask
    mask_u8 = run_binary_mask(gray_u8)
    Image.fromarray(mask_u8, mode="L").save(str(out_dir / "mask.png"))

    # 4. Boundary overlay (inferno image + yellow contour)
    mask_f  = (mask_u8 > 0).astype(np.float32)
    H, W    = rgb_disp.shape[:2]
    dpi     = 150
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(rgb_disp)
    ax.contour(mask_f, levels=[0.5], colors="yellow", linewidths=0.8)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_dir / "boundary_overlay.png"), dpi=dpi,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 5. Connected components (erode/dilate)
    sep = FloeSeparator.__new__(FloeSeparator)
    sep.img = None
    labels_filtered, areas_px, _ = sep.split_erode_dilate_small(mask_u8, min_area=BM_MIN_AREA)
    n_floes = int(labels_filtered.max())
    print(f"  Floes detected: {n_floes}")

    # 6. Floe label array
    np.save(str(out_dir / "floe_labels.npy"), labels_filtered.astype(np.int32))

    # 7. Floe size histogram (m²)
    areas_m2 = [a * GSD ** 2 for a in areas_px]
    fig, ax  = plt.subplots(figsize=(8, 5))
    if areas_m2:
        ax.hist(areas_m2, bins="auto", edgecolor="k", color="steelblue")
        ax.set_xlabel("Floe area (m²)")
        ax.set_ylabel("Count")
        ax.set_title(f"Floe size distribution  (N={n_floes},  GSD={GSD:.4f} m/px)")
    else:
        ax.text(0.5, 0.5, "No floes detected", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Floe size distribution")
    fig.tight_layout()
    fig.savefig(str(out_dir / "floe_size_hist.png"), dpi=150)
    plt.close(fig)

    # 8. Floe stats JSON (shape + size in m units)
    shape_records = characterize_floes(labels_filtered)
    for rec in shape_records:
        rec["area_m2"]      = round(rec["area_px"]       * GSD ** 2, 6)
        rec["perimeter_m"]  = round(rec["perimeter_px"]  * GSD,      4)
        rec["semi_major_m"] = round(rec["semi_major_px"] * GSD,      4)
        rec["semi_minor_m"] = round(rec["semi_minor_px"] * GSD,      4)
        del rec["area_px"], rec["perimeter_px"], rec["semi_major_px"], rec["semi_minor_px"]

    # fractal dimension (dilated boundaries) — only floes >= FRACTAL_MIN_AREA_M2
    dil_areas_all  = np.array([r["area_m2"]     for r in shape_records])
    dil_perims_all = np.array([r["perimeter_m"] for r in shape_records])
    dil_keep = dil_areas_all >= FRACTAL_MIN_AREA_M2
    fractal = compute_fractal_dimension(dil_areas_all[dil_keep], dil_perims_all[dil_keep])

    # fractal dimension (raw mask boundaries — no erode/dilate smoothing)
    raw_areas_px, raw_perims_px = measure_raw_contours(mask_u8, min_area_px=30)
    raw_areas_all  = np.array([a * GSD ** 2 for a in raw_areas_px])
    raw_perims_all = np.array([p * GSD       for p in raw_perims_px])
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
        "source_file":           tif_path.name,
        "gsd_m_per_px":          round(GSD, 6),
        "drone_height_m":        DRONE_HEIGHT_M,
        "n_floes":               n_floes,
        "rotated_180":           ROTATE_180,
        "fractal_dimension":     fractal,
        "fractal_dimension_raw": fractal_raw,
    }
    with open(str(out_dir / "floe_stats.json"), "w") as f:
        json.dump({"meta": meta, "floes": shape_records}, f, indent=2)

    # 9. Per-pixel temperature histogram
    save_pixel_temp_hist(temp_arr, out_dir / "pixel_temp_hist.png")

    # 10. Per-floe average temperature histogram
    _, temp_list = compute_average_temps(temp_arr, labels_filtered)
    save_floe_temp_hist(temp_list, out_dir / "floe_temp_hist.png")

    print(f"  Saved → {out_dir.relative_to(_ROOT)}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if INPUT_FILE is not None:
        candidates = list(INPUT_DIR.glob(f"{INPUT_FILE}*.tif"))
        if not candidates:
            print(f"No .tif file matching '{INPUT_FILE}' found in {INPUT_DIR}")
            return
        tif_files = candidates[:1]
    else:
        tif_files = sorted(INPUT_DIR.glob("*.tif"))

    if not tif_files:
        print(f"No .tif files found in {INPUT_DIR}")
        return

    print(f"Processing {len(tif_files)} file(s) from {INPUT_DIR}\n")
    for tif_path in tif_files:
        print(f"[{tif_path.name}]")
        process_tif(tif_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
