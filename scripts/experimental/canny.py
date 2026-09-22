"""Experimental: Basic Canny edge detection on thermal images. Not used in production."""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def canny():
    img_path = str(Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice_thermal.jpg")
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {os.path.abspath(img_path)}")

    # Prep
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb  = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # --- (A) Denoise (choose ONE) ---
    img_smooth = cv.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    # img_smooth = cv.GaussianBlur(img_gray, (5,5), 1.2)

    # --- (B) Gradients (Sobel) ---
    gx = cv.Sobel(img_smooth, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img_smooth, cv.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx*gx + gy*gy)

    def norm8(a):
        a = a.astype(np.float32)
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-6:
            return np.zeros_like(a, dtype=np.uint8)
        return (255*(a - mn)/(mx - mn)).astype(np.uint8)

    # --- (C) Canny ---
    LOW_T, HIGH_T = 75, 225
    edges = cv.Canny(img_smooth, LOW_T, HIGH_T)

    # --- (D) Show intermediates ---
    plt.figure(figsize=(12, 7))
    plt.subplot(231); plt.imshow(img_rgb);            plt.title("RGB");        plt.axis('off')
    plt.subplot(232); plt.imshow(img_gray, cmap='gray'); plt.title("Gray");      plt.axis('off')
    plt.subplot(233); plt.imshow(img_smooth, cmap='gray'); plt.title("Smooth");    plt.axis('off')
    plt.subplot(234); plt.imshow(norm8(np.abs(gx)), cmap='gray'); plt.title("Sobel X"); plt.axis('off')
    plt.subplot(235); plt.imshow(norm8(np.abs(gy)), cmap='gray'); plt.title("Sobel Y"); plt.axis('off')
    plt.subplot(236); plt.imshow(edges, cmap='gray'); plt.title("Canny");     plt.axis('off')
    plt.tight_layout()

    # --- NEW: overlay edges in red on original image in a new figure (111) ---
    overlay = img_rgb.copy()
    overlay[edges > 0] = [255, 0, 0]  # mark edge pixels red

    plt.figure(111)  # new tab/window
    plt.imshow(overlay)
    plt.title("Edges overlaid (red) on original")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    canny()
