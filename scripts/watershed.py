"""
watershed.py
------------
Watershed-based sea ice floe detector (marker-based).

Steps:
  1) CLAHE contrast normalization + bilateral edge-preserving denoise
  2) Ice-vs-water binary mask (Otsu or adaptive threshold)
  3) Distance-transform markers -> OpenCV watershed
  4) Per-step figure display + overlay + CSV (areas/perimeters in pixels)

Tuning knobs (top of file):
  FG_FRAC     — raise to MERGE floes (fewer splits), lower to SPLIT more (0.25–0.60 good range)
  USE_ADAPTIVE — set True if Otsu struggles (low-contrast images)
  MIN_AREA_PX  — minimum component size to keep (filters tiny noise)
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# ---------- parameters ----------
FG_FRAC = 0.35          # 0.25–0.60 good range; higher merges more
USE_ADAPTIVE = False     # try True if center is low-contrast
MIN_AREA_PX = 50         # ignore tiny specks
OUT_DIR = str(Path(__file__).parent.parent / "outputs" / "watershed_out")

def watershed():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- load ---
    img_path = str(Path(__file__).parent.parent / "data" / "images" / "sea_ice.jpg")
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"could not read {img_path}")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_RGB  = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # --- 1) contrast normalize + edge-preserving denoise ---
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(img_gray)
    gray_smooth = cv.bilateralFilter(gray_eq, d=7, sigmaColor=25, sigmaSpace=25)

    # --- 2) ice vs water binary ---
    if USE_ADAPTIVE:
        ice_bin = cv.adaptiveThreshold(gray_smooth, 255,
                                       cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 51, -5)
        th_info = "Adaptive (Gaussian, 51, -5)"
    else:
        _, ice_bin = cv.threshold(gray_smooth, 0, 255,
                                  cv.THRESH_BINARY + cv.THRESH_OTSU)
        th_info = "Otsu (global)"

    # clean small specks / seal pinholes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    ice_open  = cv.morphologyEx(ice_bin,  cv.MORPH_OPEN,  kernel, iterations=1)
    ice_clean = cv.morphologyEx(ice_open, cv.MORPH_CLOSE, kernel, iterations=2)

    # --- 3) markers from distance transform on ice ---
    dist = cv.distanceTransform(ice_clean, cv.DIST_L2, 5)
    dist_norm = cv.normalize(dist, None, 0, 1.0, cv.NORM_MINMAX)
    _, sure_fg = cv.threshold(dist, FG_FRAC * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # sure background from (dilated) water
    water = 255 - ice_clean
    sure_bg = cv.dilate(water, kernel, iterations=3)

    # unknown strip between bg and fg
    unknown = cv.subtract(sure_bg, 255 - sure_fg)

    # --- 4) watershed ---
    n_labels, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1               # background must be 1
    markers[unknown == 255] = 0         # 0 = unknown region
    markers_ws = cv.watershed(img.copy(), markers)

    # boundary mask + overlay
    boundary_mask = (markers_ws == -1).astype(np.uint8) * 255
    overlay = img_RGB.copy()
    overlay[markers_ws == -1] = [255, 0, 0]

    # segmented floes (labels >= 2)
    seg_mask = (markers_ws >= 2).astype(np.uint8) * 255

    # --- 5) contours + measurements (pixels) ---
    contours, _ = cv.findContours(seg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rows = []
    kept = 0
    for i, c in enumerate(contours, 1):
        area = cv.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        kept += 1
        perim = cv.arcLength(c, True)
        rows.append({"floe_id": kept, "area_px": float(area), "perimeter_px": float(perim)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "floe_measurements_pixels.csv"), index=False)

    # --- 6) figures (step-by-step, like your style) ---
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Sea-ice watershed (thresh={th_info}, fg_frac={FG_FRAC})", fontsize=14)

    plt.subplot(231); plt.imshow(img_RGB); plt.axis('off'); plt.title("Original (RGB)")

    plt.subplot(232); plt.imshow(gray_eq, cmap='gray'); plt.axis('off'); plt.title("CLAHE")
    plt.subplot(233); plt.imshow(gray_smooth, cmap='gray'); plt.axis('off'); plt.title("Bilateral smoothed")

    plt.subplot(234); plt.imshow(ice_bin, cmap='gray'); plt.axis('off'); plt.title("Binary (raw)")
    plt.subplot(235); plt.imshow(ice_clean, cmap='gray'); plt.axis('off'); plt.title("Binary (cleaned)")
    plt.subplot(236); plt.imshow(dist_norm, cmap='gray'); plt.axis('off'); plt.title("Distance transform (norm)")

    plt.figure(figsize=(14, 10))
    plt.suptitle("Markers → Watershed", fontsize=14)
    plt.subplot(231); plt.imshow(sure_fg, cmap='gray'); plt.axis('off'); plt.title("Sure foreground (cores)")
    plt.subplot(232); plt.imshow(sure_bg, cmap='gray'); plt.axis('off'); plt.title("Sure background")
    plt.subplot(233); plt.imshow(unknown, cmap='gray'); plt.axis('off'); plt.title("Unknown band")
    plt.subplot(234); plt.imshow(markers_ws, cmap='tab20'); plt.axis('off'); plt.title("Watershed labels")
    plt.subplot(235); plt.imshow(boundary_mask, cmap='gray'); plt.axis('off'); plt.title("Boundary mask")
    plt.subplot(236); plt.imshow(overlay); plt.axis('off'); plt.title("Overlay (red = boundary)")

    plt.tight_layout()
    plt.show()

    # --- 7) save key artifacts ---
    cv.imwrite(os.path.join(OUT_DIR, "01_gray_eq.png"), gray_eq)
    cv.imwrite(os.path.join(OUT_DIR, "02_gray_smooth.png"), gray_smooth)
    cv.imwrite(os.path.join(OUT_DIR, "03_binary_clean.png"), ice_clean)
    cv.imwrite(os.path.join(OUT_DIR, "04_distance_norm.png"), (dist_norm*255).astype(np.uint8))
    cv.imwrite(os.path.join(OUT_DIR, "05_sure_fg.png"), sure_fg)
    cv.imwrite(os.path.join(OUT_DIR, "06_sure_bg.png"), sure_bg)
    cv.imwrite(os.path.join(OUT_DIR, "07_unknown.png"), unknown)
    cv.imwrite(os.path.join(OUT_DIR, "08_boundary_mask.png"), boundary_mask)
    cv.imwrite(os.path.join(OUT_DIR, "09_overlay.png"), cv.cvtColor(overlay, cv.COLOR_RGB2BGR))

    print(f"[info] components (sure_fg): {n_labels-1} | kept contours: {kept}")
    print(f"[info] CSV written: {os.path.join(OUT_DIR,'floe_measurements_pixels.csv')}")
    print(f"[info] images saved to: {OUT_DIR}")

if __name__ == "__main__":
    watershed()
