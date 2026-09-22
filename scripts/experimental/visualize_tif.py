#!/usr/bin/env python3
"""
visualize_thermal.py

Visualize all radiometric thermal .tif files inside the folder:

    sea_ice/thermal_tif/

Usage:
    python3 -m sea_ice.visualize_thermal
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def load_thermal_tif(path: Path) -> np.ndarray:
    """Load a thermal .tif image into a NumPy array."""
    img = Image.open(path)
    arr = np.array(img)

    print(f"[INFO] Loaded {path.name}")
    print(f"       shape: {arr.shape}")
    print(f"       dtype: {arr.dtype}")
    print(f"       min:   {arr.min():.2f}")
    print(f"       max:   {arr.max():.2f}")

    return arr


def visualize_thermal(arr: np.ndarray, title: str):
    """Visualize a thermal image (float32 °C values)."""

    # Clip percentile outliers for nicer visualization
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, cmap="inferno", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Temperature (°C)")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_temperature_histogram(arr, bins=100, title="Temperature Histogram"):
    """Plot a histogram of temperature values from a thermal array."""
    
    # Flatten array → 1D
    temps = arr.flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(temps, bins=bins, color='gray', edgecolor='black')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency (pixel count)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



def main():
    # Folder containing thermal TIFFs — this directory is inside your package
    thermal_dir = Path(__file__).parent.parent.parent / "data" / "thermal_tif"

    if not thermal_dir.exists():
        raise FileNotFoundError(f"Folder not found: {thermal_dir}")

    tif_files = sorted(thermal_dir.glob("*.tif"))

    if not tif_files:
        print("[WARNING] No .tif files found in", thermal_dir)
        return

    print(f"[INFO] Found {len(tif_files)} TIFF files in {thermal_dir}\n")

    # for tif_path in tif_files:
    #     arr = load_thermal_tif(tif_path)
    #     arr = np.rot90(arr, 2)
    #     visualize_thermal(arr, title=tif_path.name)

    arr = load_thermal_tif(tif_files[0])
    arr = np.rot90(arr, 2)
    plot_temperature_histogram(arr, bins=100, title=f"Temperature Histogram: {tif_files[0].name}")
    visualize_thermal(arr, title=tif_files[0].name)


if __name__ == "__main__":
    main()
