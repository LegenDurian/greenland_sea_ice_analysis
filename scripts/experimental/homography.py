"""Experimental: Automatic thermal-to-visible image registration via distance-transform template matching."""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
try:
    from .binary_mask import FloeSeparator
except ImportError:
    from binary_mask import FloeSeparator


def _visible_to_ice_mask(vis_bgr):
    """
    Convert visible RGB/BGR image into a binary ice/sea mask.
    Output: vis_mask (uint8, 0 or 255, ice=255, sea=0)
    """
    vis_gray = cv.cvtColor(vis_bgr, cv.COLOR_BGR2GRAY)

    # Enhance contrast (helps separate bright ice from darker water)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    vis_clahe = clahe.apply(vis_gray)

    # Blur to suppress cracks/shadows inside floes
    blur = cv.GaussianBlur(vis_clahe, (11, 11), 0)

    # Otsu threshold
    _, vis_mask = cv.threshold(
        blur, 0, 255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # Ensure ice is white (255). If mostly black, invert.
    if vis_mask.mean() < 127:
        vis_mask = 255 - vis_mask

    # Morphological cleanup
    k = np.ones((5, 5), np.uint8)
    vis_mask = cv.morphologyEx(vis_mask, cv.MORPH_OPEN, k)
    vis_mask = cv.morphologyEx(vis_mask, cv.MORPH_CLOSE, k)

    return vis_mask


def _clean_thermal_mask(mask_th):
    """
    Ensure thermal mask is clean binary ice/sea mask.
    Input: mask_th (any dtype)
    Output: mask_th_bin (uint8, 0 or 255, ice=255, sea=0)
    """
    if mask_th is None:
        raise ValueError("mask_th is None")

    if mask_th.dtype != np.uint8:
        mask_th = np.clip(mask_th, 0, 255).astype(np.uint8)

    # If not already binary, re-threshold
    _, mask_bin = cv.threshold(
        mask_th, 0, 255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # Clean small noise
    k = np.ones((3, 3), np.uint8)
    mask_bin = cv.morphologyEx(mask_bin, cv.MORPH_OPEN, k)
    mask_bin = cv.morphologyEx(mask_bin, cv.MORPH_CLOSE, k)

    return mask_bin


def register_thermal_to_visible(
    vis_bgr,
    mask_th,
    thermal_gray=None,
    scale_range=(0.6, 1.4),
    scale_steps=25
):
    """
    vis_bgr:      visible image (BGR, np.ndarray)
    mask_th:      thermal mask image (single-channel, np.ndarray, ice=white)
    thermal_gray: optional thermal grayscale image (same FOV as mask_th)
    scale_range:  (min_scale, max_scale) search range
    scale_steps:  number of scales to sample between min and max

    Returns:
        H: 3x3 homography matrix (thermal mask coords -> visible coords)
        corners_vis: 4x1x2 array of mapped thermal corners in visible image
    """

    if vis_bgr is None:
        raise ValueError("vis_bgr is None (visible image not provided).")
    if mask_th is None:
        raise ValueError("mask_th is None (thermal mask not provided).")

    # --- 1. Get binary masks for both modalities ---
    vis_mask = _visible_to_ice_mask(vis_bgr)
    mask_th_bin = _clean_thermal_mask(mask_th)

    # Binary 0/1 (foreground = ice)
    vis_ice = (vis_mask > 0).astype(np.uint8)
    th_ice = (mask_th_bin > 0).astype(np.uint8)

    # --- 2. Distance transforms (shape fields) ---
    vis_dt = cv.distanceTransform(vis_ice, cv.DIST_L2, 5)
    th_dt = cv.distanceTransform(th_ice, cv.DIST_L2, 5)

    # Normalize to [0,1] for stable template matching
    vis_dt_norm = cv.normalize(vis_dt, None, 0, 1.0, cv.NORM_MINMAX)
    th_dt_norm = cv.normalize(th_dt, None, 0, 1.0, cv.NORM_MINMAX)

    h_th0, w_th0 = th_dt_norm.shape[:2]

    # --- 3. Multi-scale template matching on distance transforms ---
    best_score = -1.0
    best_rect = None      # (x, y, w, h)
    best_scale = None

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    for s in scales:
        w = int(w_th0 * s)
        h = int(h_th0 * s)
        if w < 20 or h < 20:
            continue
        if w >= vis_dt_norm.shape[1] or h >= vis_dt_norm.shape[0]:
            continue

        tmpl = cv.resize(th_dt_norm, (w, h), interpolation=cv.INTER_LINEAR)

        res = cv.matchTemplate(vis_dt_norm, tmpl, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_rect = (max_loc[0], max_loc[1], w, h)
            best_scale = s

    if best_rect is None:
        raise RuntimeError("Template matching failed (no valid scales).")

    x, y, w, h = best_rect
    print(f"[DT template] best score={best_score:.3f}, scale={best_scale:.3f}, "
          f"x={x}, y={y}, w={w}, h={h}")

    # --- 4. Build homography from thermal-mask coords -> visible coords ---
    corners_th = np.float32([
        [0,      0],
        [w_th0,  0],
        [w_th0,  h_th0],
        [0,      h_th0]
    ]).reshape(-1, 1, 2)

    corners_vis = np.float32([
        [x,     y],
        [x + w, y],
        [x + w, y + h],
        [x,     y + h]
    ]).reshape(-1, 1, 2)

    H = cv.getPerspectiveTransform(corners_th, corners_vis)
    print("Estimated homography H (from distance-transform template match):")
    print(H)

    # --- 5. Visualization: visible + quad, and overlay ---
    vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)
    vis_with_quad = vis_rgb.copy()
    corners_int = np.int32(corners_vis)
    cv.polylines(
        vis_with_quad,
        [corners_int],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3
    )

    # Warp either thermal grayscale or mask for overlay
    if thermal_gray is not None:
        if thermal_gray.dtype != np.uint8:
            thermal_gray = np.clip(thermal_gray, 0, 255).astype(np.uint8)
        thermal_rgb = cv.cvtColor(thermal_gray, cv.COLOR_GRAY2RGB)
        warped_thermal = cv.warpPerspective(
            thermal_rgb,
            H,
            (vis_rgb.shape[1], vis_rgb.shape[0])
        )
    else:
        mask_rgb = cv.cvtColor(mask_th_bin, cv.COLOR_GRAY2RGB)
        warped_thermal = cv.warpPerspective(
            mask_rgb,
            H,
            (vis_rgb.shape[1], vis_rgb.shape[0])
        )

    alpha = 0.5
    overlay = cv.addWeighted(vis_rgb, 1 - alpha, warped_thermal, alpha, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(vis_with_quad)
    axes[0].set_title("Visible with Thermal FOV (quad)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Thermal warped onto Visible")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return H, corners_vis


if __name__ == "__main__":
    # Example usage: python -m sea_ice_analysis.scripts.homography
    from pathlib import Path as _Path
    visible_path = str(_Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice.jpg")
    thermal_path = str(_Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice_thermal.jpg")

    vis_bgr = cv.imread(visible_path, cv.IMREAD_COLOR)
    if vis_bgr is None:
        raise FileNotFoundError(f"Could not read visible image at: {visible_path}")

    processor = FloeSeparator(thermal_path)
    img_gray, img_rgb, mask = processor.preprocess(
        clahe_clip=1.5,
        clahe_grid=(8, 8),
        thresh_val=70,
    )

    # Optionally use img_gray as the thermal intensity image
    thermal_gray = None
    # thermal_gray = img_gray

    H, corners_vis = register_thermal_to_visible(
        vis_bgr,
        mask,
        thermal_gray=thermal_gray,
        scale_range=(0.6, 1.4),
        scale_steps=25,
    )
