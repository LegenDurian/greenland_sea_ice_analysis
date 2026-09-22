#!/usr/bin/env python3
"""
Sea-ice boundary detection (object‑oriented refactor)

Classes:
  - ImageAsset: loads and exposes image representations and metadata
  - OutputManager: manages timestamped output directory, image saves, run_config.txt
  - EdgeSegmentation: Canny & Watershed pipelines + helper ops

Behavior mirrors the original script: fixed input 'sea_ice.jpg',
outputs saved under <output-dir>/results_<YYYYMMDD_HHMMSS>/

Usage examples:
  python sea_ice_oo.py --segmentation canny --threshold-mode edge-local --chroma-weight 0.6
  python sea_ice_oo.py --segmentation watershed --marker-frac 0.35
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse
from dataclasses import dataclass
from datetime import datetime
import platform, sys

# ---------------------- Data classes ----------------------

@dataclass
class AppArgs:
    output_prefix: str = "out"
    output_dir: str = "outputs"
    segmentation: str = "canny"              # {"canny","watershed"}
    threshold_mode: str = "global"           # {"global","edge-local"}
    edge_percentile: float = 85.0
    sigma: float = 0.33
    chroma_weight: float = 0.5
    blur_ksize: int = 5
    close_ksize: int = 5
    dilate_iter: int = 1
    min_area: int = 200
    mask_open: int = 3
    mask_close: int = 5
    marker_frac: float = 0.4

    @staticmethod
    def from_namespace(ns: argparse.Namespace) -> "AppArgs":
        return AppArgs(**vars(ns))


# ---------------------- Image class ----------------------

class ImageAsset:
    """Loads an image once and exposes common representations and metadata."""
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Image '{path.name}' not found.")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read '{path.name}'.")
        self.path = path
        self.bgr = bgr
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape[:2]

    @property
    def size(self) -> tuple[int, int]:
        return (self.w, self.h)


# ---------------------- Output class ----------------------

class OutputManager:
    """Handles output folder creation, saving artifacts, and writing a run config."""
    def __init__(self, base_dir: Path, prefix: str):
        self.base_dir = base_dir
        self.prefix = prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"results_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, stem: str, image: np.ndarray) -> Path:
        out_path = self.run_dir / f"{self.prefix}_{stem}.png"
        cv2.imwrite(str(out_path), image)
        return out_path

    def write_config(self, meta: dict):
        lines = [
            "# Sea-ice run configuration",
            f"timestamp           : {meta.get('timestamp', self.timestamp)}",
            f"input_file          : {meta.get('input_file')}",
            f"image_size          : {meta.get('image_w')}x{meta.get('image_h')}",
            f"output_prefix       : {meta.get('output_prefix', self.prefix)}",
            f"output_dir          : {self.run_dir}",
            "",
            "# Mode & thresholds",
            f"segmentation        : {meta.get('segmentation')}",
            f"threshold_mode      : {meta.get('threshold_mode')}",
            f"sigma               : {meta.get('sigma')}",
            f"edge_percentile     : {meta.get('edge_percentile')}",
            f"chroma_weight       : {meta.get('chroma_weight')}",
            f"canny_low           : {meta.get('canny_low')}",
            f"canny_high          : {meta.get('canny_high')}",
            "",
            "# Pre/Post-processing",
            f"blur_ksize          : {meta.get('blur_ksize')}",
            f"close_ksize         : {meta.get('close_ksize')}",
            f"dilate_iter         : {meta.get('dilate_iter')}",
            f"min_area            : {meta.get('min_area')}",
            f"mask_open           : {meta.get('mask_open')}",
            f"mask_close          : {meta.get('mask_close')}",
            f"marker_frac         : {meta.get('marker_frac')}",
            "",
            "# Results",
            f"contours_kept       : {meta.get('contours_kept')}",
            "",
            "# Environment",
            f"python_version      : {meta.get('python_version')}",
            f"platform            : {meta.get('platform')}",
            f"opencv_version      : {meta.get('opencv_version')}",
            f"numpy_version       : {meta.get('numpy_version')}",
            "",
        ]
        (self.run_dir / "run_config.txt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------- Segmentation & helpers ----------------------

class EdgeSegmentation:
    """Implements color-aware Canny and watershed pipelines and utilities."""

    # ----- Threshold strategies -----
    @staticmethod
    def thresholds_global(gray_like: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
        v = float(np.median(gray_like))
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        if upper <= lower:
            upper = min(255, lower + 10)
        return lower, upper

    @staticmethod
    def thresholds_edge_local(gray_like: np.ndarray, sigma: float = 0.33, edge_percentile: float = 85.0) -> tuple[int, int]:
        gx = cv2.Sobel(gray_like, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_like, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        cutoff = np.percentile(mag_norm, edge_percentile)
        mask = mag_norm >= cutoff
        if not np.any(mask):
            return EdgeSegmentation.thresholds_global(gray_like, sigma=sigma)
        ref = float(np.median(mag_norm[mask]))
        lower = int(max(0, (1.0 - sigma) * ref))
        upper = int(min(255, (1.0 + sigma) * ref))
        if upper <= lower:
            upper = min(255, lower + 10)
        return lower, upper

    # ----- Helpers -----
    @staticmethod
    def preprocess_L(L: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        Lp = clahe.apply(L)
        if blur_ksize and (blur_ksize % 2 == 1):
            Lp = cv2.GaussianBlur(Lp, (blur_ksize, blur_ksize), 0)
        return Lp

    @staticmethod
    def lab_edge_channel(img_bgr: np.ndarray, chroma_weight: float = 0.5, blur_ksize: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Build a single 8-bit 'edge image' from Lab with optional chroma weighting."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        Lp = EdgeSegmentation.preprocess_L(L, blur_ksize=blur_ksize)
        a_f = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        b_f = cv2.normalize(b.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

        def gradmag(x):
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)

        gL = gradmag(Lp)
        ga = gradmag(a_f)
        gb = gradmag(b_f)
        g = gL + chroma_weight * (ga + gb)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        g_u8 = g_norm.astype(np.uint8)
        return Lp, g_u8

    @staticmethod
    def connect_edges(edges: np.ndarray, close_ksize: int = 5, dilate_iter: int = 1) -> np.ndarray:
        work = edges.copy()
        if close_ksize and close_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)
        if dilate_iter and dilate_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            work = cv2.dilate(work, k, iterations=dilate_iter)
        return work

    @staticmethod
    def find_and_draw_contours(bw: np.ndarray, img_bgr: np.ndarray, min_area: int = 200):
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]
        overlay = img_bgr.copy()
        cv2.drawContours(overlay, kept, -1, (0, 255, 255), 2, cv2.LINE_AA)
        return overlay, kept

    # ----- Watershed pipeline -----
    @staticmethod
    def segment_watershed(img_bgr: np.ndarray, blur_ksize=5, min_area=200,
                           mask_open=3, mask_close=5, marker_frac=0.4):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        Lp = EdgeSegmentation.preprocess_L(L, blur_ksize=blur_ksize)
        _, ice_bin = cv2.threshold(Lp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_open, mask_open))
            ice_bin = cv2.morphologyEx(ice_bin, cv2.MORPH_OPEN, k, iterations=1)
        if mask_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_close, mask_close))
            ice_bin = cv2.morphologyEx(ice_bin, cv2.MORPH_CLOSE, k, iterations=1)

        dist = cv2.distanceTransform(ice_bin, cv2.DIST_L2, 5)
        if dist.max() > 0:
            markers_bin = (dist > (marker_frac * dist.max())).astype(np.uint8) * 255
        else:
            markers_bin = np.zeros_like(ice_bin)

        num_labels, markers = cv2.connectedComponents(markers_bin)
        gx = cv2.Sobel(Lp, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Lp, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_u8 = cv2.GaussianBlur(grad_u8, (3, 3), 0)

        markers_ws = markers.copy().astype(np.int32)
        markers_ws[ice_bin == 0] = 0
        grad_rgb = cv2.merge([grad_u8, grad_u8, grad_u8])
        cv2.watershed(grad_rgb, markers_ws)

        labels = markers_ws.copy()
        labels[labels < 0] = 0
        unique = np.unique(labels)
        overlay = img_bgr.copy()
        kept_contours = []
        for lbl in unique:
            if lbl == 0:
                continue
            region = (labels == lbl).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= float(min_area):
                    kept_contours.append(c)
        cv2.drawContours(overlay, kept_contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        return ice_bin, markers_bin, grad_u8, overlay, kept_contours


# ---------------------- CLI / Orchestration ----------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sea-ice boundary detection (fixed input: sea_ice.jpg).")
    ap.add_argument("--output-prefix", "-o", default="out", help="Prefix for saved outputs.")
    ap.add_argument("--output-dir", "-d", default=str(Path(__file__).parent.parent.parent / "outputs"), help="Base directory to save outputs.")
    ap.add_argument("--segmentation", choices=["canny", "watershed"], default="canny",
                    help="Choose Canny (edge map) or Watershed (region split) pipeline.")
    # Canny options
    ap.add_argument("--threshold-mode", choices=["global", "edge-local"], default="global",
                    help="Canny threshold strategy.")
    ap.add_argument("--edge-percentile", type=float, default=85.0,
                    help="In edge-local mode, percentile of gradients to define likely edges.")
    ap.add_argument("--sigma", type=float, default=0.33, help="Threshold spread (typ. 0.2–0.5).")
    ap.add_argument("--chroma-weight", type=float, default=0.5,
                    help="Weight for Lab chroma gradients (a+b) in Canny mode.")
    # Shared denoise/morph
    ap.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur kernel (odd).")
    ap.add_argument("--close-ksize", type=int, default=5, help="Morphological closing size (0=off).")
    ap.add_argument("--dilate-iter", type=int, default=1, help="Dilate iterations before contours.")
    ap.add_argument("--min-area", type=int, default=200, help="Min contour area to keep.")
    # Watershed options
    ap.add_argument("--mask-open", type=int, default=3, help="Mask opening kernel size.")
    ap.add_argument("--mask-close", type=int, default=5, help="Mask closing kernel size.")
    ap.add_argument("--marker-frac", type=float, default=0.4,
                    help="Distance-transform fraction for marker creation (0.3–0.5 typical).")
    return ap


def main():
    ap = build_argparser()
    args = AppArgs.from_namespace(ap.parse_args())

    # Load input (fixed filename to match original behavior)
    image_path = Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice.jpg"
    img = ImageAsset(image_path)

    # Prepare outputs
    out = OutputManager(base_dir=Path(args.output_dir), prefix=args.output_prefix)

    if args.segmentation == "canny":
        Lp, edge_img = EdgeSegmentation.lab_edge_channel(img.bgr, chroma_weight=args.chroma_weight, blur_ksize=args.blur_ksize)
        if args.threshold_mode == "edge-local":
            low, high = EdgeSegmentation.thresholds_edge_local(edge_img, sigma=args.sigma, edge_percentile=args.edge_percentile)
        else:
            low, high = EdgeSegmentation.thresholds_global(edge_img, sigma=args.sigma)

        edges = cv2.Canny(edge_img, threshold1=low, threshold2=high, L2gradient=True)
        edges_connected = EdgeSegmentation.connect_edges(edges, close_ksize=args.close_ksize, dilate_iter=args.dilate_iter)
        overlay, contours = EdgeSegmentation.find_and_draw_contours(edges_connected, img.bgr, min_area=args.min_area)

        # Save
        out.save_image("edge_channel", edge_img)
        out.save_image("edges", edges)
        out.save_image("edges_connected", edges_connected)
        out.save_image("overlay_contours", overlay)

        # Config
        meta = {
            "timestamp": out.timestamp,
            "input_file": img.path.name,
            "image_w": img.w, "image_h": img.h,
            "output_prefix": args.output_prefix,
            "segmentation": "canny",
            "threshold_mode": args.threshold_mode,
            "sigma": args.sigma,
            "edge_percentile": args.edge_percentile if args.threshold_mode == "edge-local" else None,
            "chroma_weight": args.chroma_weight,
            "canny_low": low, "canny_high": high,
            "blur_ksize": args.blur_ksize, "close_ksize": args.close_ksize,
            "dilate_iter": args.dilate_iter, "min_area": args.min_area,
            "mask_open": None, "mask_close": None, "marker_frac": None,
            "contours_kept": len(contours),
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }
        out.write_config(meta)

    else:  # watershed
        ice_mask, marker_mask, grad_u8, overlay, contours = EdgeSegmentation.segment_watershed(
            img.bgr, blur_ksize=args.blur_ksize, min_area=args.min_area,
            mask_open=args.mask_open, mask_close=args.mask_close, marker_frac=args.marker_frac
        )
        out.save_image("mask_ice", ice_mask)
        out.save_image("mask_markers", marker_mask)
        out.save_image("grad_elevation", grad_u8)
        out.save_image("overlay_contours", overlay)

        meta = {
            "timestamp": out.timestamp,
            "input_file": img.path.name,
            "image_w": img.w, "image_h": img.h,
            "output_prefix": args.output_prefix,
            "segmentation": "watershed",
            "threshold_mode": None, "sigma": None, "edge_percentile": None,
            "chroma_weight": None, "canny_low": None, "canny_high": None,
            "blur_ksize": args.blur_ksize, "close_ksize": args.close_ksize,
            "dilate_iter": args.dilate_iter, "min_area": args.min_area,
            "mask_open": args.mask_open, "mask_close": args.mask_close, "marker_frac": args.marker_frac,
            "contours_kept": len(contours),
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        }
        out.write_config(meta)

    print(f"[OK] Saved results to: {out.run_dir}")


if __name__ == "__main__":
    main()
