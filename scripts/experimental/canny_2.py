"""
Canny (with boundary clean-up) — single function, simple intermediates.

What it does:
1) Smooth -> Canny
2) Otsu threshold to get an "ice mask" (ice ~ bright, water ~ dark)
3) Morphological gradient of the mask = thin boundary band
4) Keep only edges that lie on the mask boundary (removes inside-ice edges)
5) Close small gaps to reduce discontinuities and double lines
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def canny_boundary():
    img_path = str(Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice.jpg")
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(os.path.abspath(img_path))

    img_rgb  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray     = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # --- 1) Smooth to reduce texture (helps remove inner edges) ---
    # Bilateral preserves edges while smoothing within floes
    smooth = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # (Try Gaussian instead if you prefer)
    # smooth = cv.GaussianBlur(gray, (5,5), 1.2)

    # --- 2) Canny (base edges) ---
    LOW_T, HIGH_T = 60, 180    # tune: raise both to reduce clutter
    edges = cv.Canny(smooth, LOW_T, HIGH_T)

    # --- 3) Ice vs water mask (Otsu) ---
    # Assumption: ice brighter than water
    _, ice_mask = cv.threshold(smooth, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # --- 4) Thin "boundary band" of mask (outer contour of ice blobs) ---
    # This band hugs the transition ice<->water; interior texture gets suppressed
    k = np.ones((3,3), np.uint8)
    boundary_band = cv.morphologyEx(ice_mask, cv.MORPH_GRADIENT, k)  # a 1–2px ring

    # --- 5) Keep only edges that fall on that boundary band ---
    edges_on_boundary = cv.bitwise_and(edges, boundary_band)

    # --- 6) Fix small discontinuities / reduce doubles (closing then light thinning) ---
    # Close tiny gaps and fuse split/double edges
    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    closed  = cv.morphologyEx(edges_on_boundary, cv.MORPH_CLOSE, close_k, iterations=1)

    # Optional gentle erode to remove lingering doubles (comment out if it breaks too much)
    cleaned = cv.erode(closed, np.ones((2,2), np.uint8), iterations=1)

    # --- 7) Overlay on original for sanity check ---
    overlay = img_rgb.copy()
    overlay[cleaned > 0] = [255, 0, 0]   # red edges

    # --- Show intermediates (simple grid like your example) ---
    plt.figure(figsize=(12, 9))

    plt.subplot(241); plt.title('RGB'); plt.axis('off'); plt.imshow(img_rgb)
    plt.subplot(242); plt.title('Gray'); plt.axis('off'); plt.imshow(gray, cmap='gray')
    plt.subplot(243); plt.title('Smoothed'); plt.axis('off'); plt.imshow(smooth, cmap='gray')
    plt.subplot(244); plt.title(f'Canny ({LOW_T},{HIGH_T})'); plt.axis('off'); plt.imshow(edges, cmap='gray')

    plt.subplot(245); plt.title('Ice mask (Otsu)'); plt.axis('off'); plt.imshow(ice_mask, cmap='gray')
    plt.subplot(246); plt.title('Mask boundary band'); plt.axis('off'); plt.imshow(boundary_band, cmap='gray')
    plt.subplot(247); plt.title('Edges ∩ boundary'); plt.axis('off'); plt.imshow(edges_on_boundary, cmap='gray')
    plt.subplot(248); plt.title('Closed & cleaned'); plt.axis('off'); plt.imshow(cleaned, cmap='gray')

    plt.figure(figsize=(6,5))
    plt.title('Overlay'); plt.axis('off'); plt.imshow(overlay)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    canny_boundary()
