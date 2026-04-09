import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image

class FloeSeparator:
    """This class is used to process a singular thermal sea ice image into segmented ice floes. 
    Results are stored as self.labels_filtered
    """
    def __init__(self, img_path):
        # load image
        self.img_path = img_path
        self.img = cv.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not read image at: {os.path.abspath(img_path)}")

        self.img_gray = None
        self.img_rgb = None
        self.mask = None
        self.labels_filtered = None
        self.areas_px = None
        self.centroids = None

    @staticmethod
    def save_rgb_as_bgr(path, img_rgb):
        """Accepts float/int arrays, converts to uint8 BGR, then writes."""
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
        cv.imwrite(path, img_bgr)
        return

    def preprocess(self, img_path=None, clahe_clip=1.5, clahe_grid=(8, 8), thresh_val=70):
        """Load image, CLAHE-equalize gray, binary-inv threshold. Returns (img_gray, img_rgb, mask)."""
        # if a new path is provided, update self.img
        if img_path is not None:
            self.img_path = img_path
            self.img = cv.imread(img_path)

        img = self.img
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {os.path.abspath(self.img_path)}")

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_rgb  = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # CLAHE
        clahe = cv.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        img_gray_eq = clahe.apply(img_gray)
        cv.imwrite('clahe_output.jpg', img_gray_eq)

        # Binary INV: 255=ice, 0=sea
        _, mask = cv.threshold(img_gray_eq, thresh_val, 255, cv.THRESH_BINARY_INV)
        mask = mask.astype(np.uint8)

        # store on self so they are accessible later
        self.img_gray = img_gray
        self.img_rgb = img_rgb
        self.mask = mask

        return img_gray, img_rgb, mask

    def split_erode_dilate(self, mask, min_area=30):
        k3 = np.ones((3,3), np.uint8)
        eroded = cv.erode(mask, k3, iterations=1)
        num_labels, labels = cv.connectedComponents(eroded)
        # Dilate each label back separately, then stack (keeps identities)
        labels_out = np.zeros_like(labels, dtype=np.int32)
        nid = 1
        for lbl in range(1, num_labels):
            r = (labels == lbl).astype(np.uint8) * 255
            d = cv.dilate(r, k3, iterations=1)
            # Constrain by original mask to avoid leaking into sea
            d = cv.bitwise_and(d, mask)
            a = int((d > 0).sum())
            if a >= min_area:
                labels_out[d > 0] = nid
                nid += 1
        # areas/centroids
        areas_px, centroids = [], []
        for i in range(1, nid):
            rr = (labels_out == i)
            areas_px.append(int(rr.sum()))
            ys, xs = np.nonzero(rr)
            centroids.append((float(xs.mean()), float(ys.mean())))

        # store on self
        self.labels_filtered = labels_out
        self.areas_px = areas_px
        self.centroids = centroids

        return labels_out, areas_px, centroids

    def split_erode_dilate_small(
        self,
        mask,
        min_area=30,
        erode_min_size=500,
        meters_per_pixel=0.1218
    ):
        k3 = np.ones((3, 3), np.uint8)

        # Work from original mask components first
        num0, labels0, stats0, cents0 = cv.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv.CV_32S
        )

        labels_out = np.zeros_like(labels0, dtype=np.int32)
        areas_px, areas_m, centroids = [], [], []
        nid = 1

        px_to_m2 = meters_per_pixel ** 2

        for i in range(1, num0):
            area0 = int(stats0[i, cv.CC_STAT_AREA])
            if area0 < min_area:
                continue

            comp = (labels0 == i).astype(np.uint8) * 255

            if area0 < erode_min_size:
                rr = (comp > 0)
                a_px = int(rr.sum())
                a_m = a_px * px_to_m2

                labels_out[rr] = nid
                areas_px.append(a_px)
                areas_m.append(a_m)

                ys, xs = np.nonzero(rr)
                centroids.append((float(xs.mean()), float(ys.mean())))
                nid += 1
                continue

            # Erode/dilate only for large components
            eroded = cv.erode(comp, k3, iterations=1)
            num_labels, labels = cv.connectedComponents(eroded)

            # Dilate each label back separately
            for lbl in range(1, num_labels):
                r = (labels == lbl).astype(np.uint8) * 255
                d = cv.dilate(r, k3, iterations=1)
                d = cv.bitwise_and(d, comp)

                a_px = int((d > 0).sum())
                if a_px >= min_area:
                    a_m = a_px * px_to_m2

                    labels_out[d > 0] = nid
                    areas_px.append(a_px)
                    areas_m.append(a_m)

                    ys, xs = np.nonzero(d)
                    centroids.append((float(xs.mean()), float(ys.mean())))
                    nid += 1

        # store on self
        self.labels_filtered = labels_out
        self.areas_px = areas_px          # optional, but often useful
        self.areas_m = areas_m            # area in m^2
        self.centroids = centroids

        return labels_out, areas_m, centroids


    def split_simple(self, mask, min_area=30):
        """
        Simple connected-components segmentation directly on the binary mask.
        No erosion, no watershed.
        - mask: 255=ice, 0=sea
        - min_area: filter out tiny specks
        Returns (labels_filtered, areas_px, centroids)
        """
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv.CV_32S
        )
        labels_filtered = np.zeros_like(labels, dtype=np.int32)
        areas_px = []
        centroids_new = []
        next_id = 1
        for i in range(1, num_labels):  # skip background (label 0)
            area_i = int(stats[i, cv.CC_STAT_AREA])
            if area_i >= min_area:
                labels_filtered[labels == i] = next_id
                areas_px.append(area_i)
                centroids_new.append(tuple(centroids[i]))  # (cx, cy)
                next_id += 1

        # store on self
        self.labels_filtered = labels_filtered
        self.areas_px = areas_px
        self.centroids = centroids_new

        return labels_filtered, areas_px, centroids_new

    def get_colored_segments(self, labels_filtered, count, color_seed=0):
        """
        Given filtered labels and count (number of distinct labels), produce a colorized segmentation image.
        Returns image of segmented regions in random colors.
        """
        if count > 0:
            rng = np.random.default_rng(color_seed)
            lut = np.vstack([[0, 0, 0], rng.integers(0, 255, size=(count, 3), dtype=np.uint8)])
            seg_rgb = lut[labels_filtered].astype(np.uint8)
        else:
            seg_rgb = np.zeros((*labels_filtered.shape, 3), dtype=np.uint8)
        return seg_rgb

    def get_numbered_segments(self, img_gray, labels_filtered, centroids, count):
        """
        Given grayscale image, filtered labels, centroids, and count,
        produce an overlay image with red boundaries and numbers at centroids.
        """
        overlay_rgb = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)
        for lbl in range(1, count + 1):
            comp_mask = (labels_filtered == lbl).astype(np.uint8) * 255
            if comp_mask.sum() == 0:
                continue
            contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 1, lineType=cv.LINE_AA)  # red borders

        # Numbers at centroids
        overlay_bgr = cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR)
        H, W = overlay_bgr.shape[:2]
        font_scale = max(0.5, min(2.0, H / 1000.0))
        for idx, (cx, cy) in enumerate(centroids, start=1):
            x, y = int(round(cx)), int(round(cy))
            cv.putText(overlay_bgr, str(idx), (x, y),
                    cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv.LINE_AA)  # outline
            cv.putText(overlay_bgr, str(idx), (x, y),
                    cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)  # fill
        overlay_numbered = cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)
        return overlay_numbered

    def visualize(self, img_gray, img_rgb, mask, labels_filtered, centroids, areas_px,
                        color_seed=0, hist_path='size_distribution_hist.png'):
        """
        Make a compact 2x2 figure:
        1) Original grayscale
        2) Binary mask
        3) Segmentation (random color per label)
        4) Grayscale overlaid with red boundaries + numbers
        """
        count = int(labels_filtered.max())
        print("Estimated number of ice chunks:", count)

        seg_rgb = self.get_colored_segments(labels_filtered, count, color_seed)

        overlay_numbered = self.get_numbered_segments(img_gray, labels_filtered, centroids, count)

        # figure
        plt.figure(figsize=(12, 9))
        plt.subplot(221)
        plt.title("Original (gray)")
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')

        plt.subplot(222)
        plt.title("Binary mask (ice=white, sea=black)")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(223)
        plt.title(f"Segmentation (N={count})")
        plt.imshow(seg_rgb)
        plt.axis('off')

        plt.subplot(224)
        plt.title("Overlay: red edges + IDs")
        plt.imshow(overlay_numbered)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('figure_2x2.png', dpi=150)
        plt.show()

        # size distribution
        if len(areas_px) > 0:
            areas_px = np.asarray(areas_px, dtype=np.int32)
            print(f"Area stats (m^2) — N={len(areas_px)}")
            print(f"  min: {areas_px.min():,}")
            print(f"  max: {areas_px.max():,}")
            print(f"  mean: {areas_px.mean():,.2f}")
            print(f"  median: {np.median(areas_px):,.2f}")

            plt.figure(figsize=(8, 5))
            plt.hist(areas_px, bins='auto', edgecolor='k')
            plt.xlabel('Area (m²)')
            plt.ylabel('Count')
            plt.title('Floe Size Distribution (m²)')
            plt.tight_layout()
            plt.show()
        else:
            print("No components above min_area; skipping size distribution plot.")


def _pick_points(img_rgb, n_points=8, title="Click points (close window when done)"):
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"{title}\nPick {n_points} points in order; close the window when done.")
    pts = plt.ginput(n_points, timeout=0)
    plt.close()
    if len(pts) < 4:
        raise ValueError("Need at least 4 points for a homography.")
    return np.array(pts, dtype=np.float32)

def _warp_labels_nearest(labels_int, H, out_w, out_h):
    warped = cv.warpPerspective(labels_int.astype(np.int32), H, (out_w, out_h),
                                flags=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return warped

def _draw_contours_and_ids_on_visible(visible_rgb, labels_vis, centroids_vis):
    overlay_rgb = visible_rgb.copy()
    count = int(labels_vis.max())
    for lbl in range(1, count + 1):
        comp_mask = (labels_vis == lbl).astype(np.uint8) * 255
        if comp_mask.sum() == 0:
            continue
        contours, _ = cv.findContours(comp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay_rgb, contours, -1, (255, 0, 0), 1, lineType=cv.LINE_AA)
    overlay_bgr = cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR)
    Hh, Ww = overlay_bgr.shape[:2]
    font_scale = max(0.5, min(2.0, Hh / 1000.0))
    for idx, (x, y) in enumerate(centroids_vis, start=1):
        xi, yi = int(round(x)), int(round(y))
        cv.putText(overlay_bgr, str(idx), (xi, yi),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(overlay_bgr, str(idx), (xi, yi),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv.LINE_AA)
    return cv.cvtColor(overlay_bgr, cv.COLOR_BGR2RGB)

def register_and_overlay_visible(thermal_gray, thermal_labels, thermal_centroids,
                                 visible_rgb, n_points=8, ransac_thresh=3.0):
    th_rgb_for_click = cv.cvtColor(thermal_gray, cv.COLOR_GRAY2RGB)
    pts_th = _pick_points(th_rgb_for_click, n_points=n_points, title="THERMAL: pick points")
    pts_vis = _pick_points(visible_rgb, n_points=n_points, title="VISIBLE: pick SAME points in SAME order")

    H, inliers = cv.findHomography(pts_th, pts_vis, method=cv.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed. Try different / more points.")

    vis_h, vis_w = visible_rgb.shape[:2]
    warped_labels = _warp_labels_nearest(thermal_labels, H, vis_w, vis_h)

    if len(thermal_centroids) > 0:
        pts = np.array(thermal_centroids, dtype=np.float32).reshape(-1, 1, 2)
        pts_warp = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    else:
        pts_warp = np.zeros((0, 2), dtype=np.float32)

    overlay_numbered = _draw_contours_and_ids_on_visible(visible_rgb, warped_labels, pts_warp)
    return overlay_numbered

def match_floes(labels1, centroids1, areas1,
                labels2, centroids2, areas2,
                max_centroid_dist=50,  # pixels
                min_iou=0.05):
    """
    Match floes between two frames using centroid distance + IoU.

    labels1, labels2: 2D int arrays, labels in [0..N]
    centroids1, centroids2: list of (cx, cy) for each label index 1..N
    areas1, areas2: list of areas (same ordering as centroids)
    Returns:
        matches: list of (idx1, idx2, dx, dy, dist, iou)
                 where idx1, idx2 are 0-based indices into centroids1/centroids2
        unmatched1: list of idx1 with no match
        unmatched2: list of idx2 with no match
    """
    H, W = labels1.shape
    N1 = len(centroids1)
    N2 = len(centroids2)

    unmatched2 = set(range(N2))
    matches = []
    unmatched1 = []

    for i in range(N1):
        cx1, cy1 = centroids1[i]
        lbl1 = i + 1  # label value in labels1

        best_j = None
        best_score = -np.inf
        best_iou = 0.0
        best_dist = np.inf

        for j in list(unmatched2):
            cx2, cy2 = centroids2[j]
            lbl2 = j + 1  # label value in labels2

            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = np.hypot(dx, dy)
            if dist > max_centroid_dist:
                continue  # too far to be plausible

            # IoU between the two floe masks
            mask1 = (labels1 == lbl1)
            mask2 = (labels2 == lbl2)
            inter = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou = inter / union if union > 0 else 0.0
            if iou < min_iou:
                continue

            # Score: prefer high IoU, small distance
            score = 5.0 * iou - 0.01 * dist

            if score > best_score:
                best_score = score
                best_j = j
                best_iou = iou
                best_dist = dist

        if best_j is not None:
            cx2, cy2 = centroids2[best_j]
            dx = cx2 - cx1
            dy = cy2 - cy1
            matches.append((i, best_j, dx, dy, best_dist, best_iou))
            unmatched2.remove(best_j)
        else:
            unmatched1.append(i)

    unmatched2 = list(unmatched2)
    return matches, unmatched1, unmatched2


def visualize_motion(img_gray, centroids1, matches,
                     save_path='floe_motion_vectors.png'):
    """
    Visualize floe motion as arrows from centroids in frame 1.

    img_gray: grayscale image from frame 1
    centroids1: list of (cx, cy) for frame 1
    matches: list of (idx1, idx2, dx, dy, dist, iou)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img_gray, cmap='gray')
    plt.title("Floe motion vectors (t0 → t1)")
    plt.axis('off')

    xs, ys, us, vs = [], [], [], []

    for idx1, idx2, dx, dy, dist, iou in matches:
        cx, cy = centroids1[idx1]
        xs.append(cx)
        ys.append(cy)
        us.append(dx)
        vs.append(dy)

    if len(xs) > 0:
        plt.quiver(xs, ys, us, vs,
                   angles='xy', scale_units='xy', scale=1,
                   color='red', width=0.002)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def compute_average_temps(arr, labels):
    """
    Compute average temperature per region (label).
    
    Parameters:
        arr:    2D float32 thermal image (temperature per pixel)
        labels: 2D int32 label map (0 = background, 1..N = floes)
    
    Returns:
        avg_temps: dict mapping label_id → average_temperature
        temp_list: list of averages in label order [label1_avg, label2_avg, ...]
    """
    if arr.shape != labels.shape:
        raise ValueError("arr and labels must have the same shape!")

    max_lbl = labels.max()
    avg_temps = {}
    temp_list = []

    for lbl in range(1, max_lbl + 1):       # skip label 0 (background)
        mask = (labels == lbl)
        if np.any(mask):
            mean_temp = float(arr[mask].mean())
            avg_temps[lbl] = mean_temp
            temp_list.append(mean_temp)
        else:
            avg_temps[lbl] = np.nan
            temp_list.append(np.nan)

    return avg_temps, temp_list

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

def plot_floe_temperature_hist(temp_list, bins=20):
    temps = np.array(temp_list)
    temps = temps[~np.isnan(temps)]   # remove NaNs if any

    plt.figure(figsize=(7,5))
    plt.hist(temps, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Floe Count")
    plt.title("Histogram of Average Floe Temperatures")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def export_floe_temperature_csv(arr, labels, areas_px, centroids, csv_path):
    """
    Export per-floe temperature statistics to a CSV file.

    Parameters:
        arr:        2D float array, thermal image (temperature per pixel)
        labels:     2D int array, same shape as arr, 0 = background, 1..N = floe labels
        areas_px:   list of areas in pixels; index (label-1) corresponds to label
        centroids:  list of (cx, cy); index (label-1) corresponds to label
        csv_path:   output CSV file path (str or Path)
    """
    if arr.shape != labels.shape:
        raise ValueError("arr and labels must have the same shape!")

    max_lbl = int(labels.max())

    rows = []
    for lbl in range(1, max_lbl + 1):
        mask = (labels == lbl)
        if not np.any(mask):
            # no pixels with this label; skip
            continue

        temps = arr[mask]
        avg_temp = float(temps.mean())
        min_temp = float(temps.min())
        max_temp = float(temps.max())
        std_temp = float(temps.std())

        # area and centroid from provided lists if available
        if areas_px is not None and len(areas_px) >= lbl:
            area_px_lbl = int(areas_px[lbl - 1])
        else:
            area_px_lbl = int(mask.sum())

        if centroids is not None and len(centroids) >= lbl:
            cx, cy = centroids[lbl - 1]
        else:
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())

        rows.append({
            "floe_id": lbl,
            "area_px": area_px_lbl,
            "centroid_x": cx,
            "centroid_y": cy,
            "avg_temp": avg_temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "std_temp": std_temp,
        })

    fieldnames = [
        "floe_id",
        "area_px",
        "centroid_x",
        "centroid_y",
        "avg_temp",
        "min_temp",
        "max_temp",
        "std_temp",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} floe records to {csv_path}")

def load_and_segment_sequence(folder, clahe_clip=1.5, clahe_grid=(8, 8), thresh_val=75,
                              min_area=30, erode_min_size=500):
    """
    Load all images in `folder`, sort them by filename, and run segmentation
    on each to get labels, areas, and centroids.

    Returns:
        img_grays:   list of grayscale images per frame
        img_rgbs:    list of RGB images per frame
        labels_list: list of 2D label arrays per frame
        areas_list:  list of areas_px lists per frame
        cents_list:  list of centroids lists per frame
        paths:       list of image paths in order
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # You can adjust extensions if needed (.JPG, .jpg, .png, etc.)
    img_paths = sorted(list(folder.glob("*.JPG")))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {folder}")

    img_grays   = []
    img_rgbs    = []
    labels_list = []
    areas_list  = []
    cents_list  = []

    for p in img_paths:
        processor = FloeSeparator(str(p))
        img_gray, img_rgb, mask = processor.preprocess(
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            thresh_val=thresh_val
        )
        # Use the same method you used in __main__
        labels_filtered, areas_px, centroids = processor.split_erode_dilate_small(
            mask,
            min_area=min_area,
            erode_min_size=erode_min_size
        )

        img_grays.append(img_gray)
        img_rgbs.append(img_rgb)
        labels_list.append(labels_filtered)
        areas_list.append(areas_px)
        cents_list.append(centroids)

    return img_grays, img_rgbs, labels_list, areas_list, cents_list, img_paths

def track_floes_over_sequence(labels_list, areas_list, cents_list,
                              max_centroid_dist=40, min_iou=0.05):
    """
    Use match_floes between consecutive frames to build floe tracks over time.

    Returns:
        track_centroids: dict track_id -> list of (frame_index, cx, cy)
        num_frames:      total number of frames
    """
    num_frames = len(labels_list)
    if num_frames < 2:
        raise ValueError("Need at least 2 frames for motion tracking.")

    # track_assignments[f][i] = track_id of floe i in frame f (i is index into cents_list[f])
    track_assignments = [dict() for _ in range(num_frames)]
    track_centroids   = {}

    # Initialize tracks from frame 0: each floe becomes a new track
    track_id_counter = 0
    for i, (cx, cy) in enumerate(cents_list[0]):
        tid = track_id_counter
        track_id_counter += 1
        track_assignments[0][i] = tid
        track_centroids[tid] = [(0, cx, cy)]

    # Propagate through subsequent frames
    for f in range(num_frames - 1):
        labels1    = labels_list[f]
        centroids1 = cents_list[f]
        areas1     = areas_list[f]

        labels2    = labels_list[f + 1]
        centroids2 = cents_list[f + 1]
        areas2     = areas_list[f + 1]

        matches, unmatched1, unmatched2 = match_floes(
            labels1, centroids1, areas1,
            labels2, centroids2, areas2,
            max_centroid_dist=max_centroid_dist,
            min_iou=min_iou
        )

        # Propagate tracks from frame f to f+1
        for idx1, idx2, dx, dy, dist, iou in matches:
            if idx1 not in track_assignments[f]:
                # This floe didn't have a track (e.g., appeared mid-sequence), skip
                continue
            tid = track_assignments[f][idx1]
            track_assignments[f + 1][idx2] = tid
            cx2, cy2 = centroids2[idx2]
            track_centroids[tid].append((f + 1, cx2, cy2))

        # Any unmatched floe in frame f+1 is a new track
        for idx2 in unmatched2:
            cx2, cy2 = centroids2[idx2]
            tid = track_id_counter
            track_id_counter += 1
            track_assignments[f + 1][idx2] = tid
            track_centroids[tid] = [(f + 1, cx2, cy2)]

    return track_centroids, num_frames

def compute_avg_velocity_for_full_tracks(track_centroids, num_frames, dt_seconds=2.0):
    """
    For each track that spans ALL frames, compute average velocity vector and direction.

    Returns:
        results: list of dict with keys:
            track_id, avg_vx_px_s, avg_vy_px_s, speed_px_s, direction_deg, points
    """
    results = []

    for tid, points in track_centroids.items():
        # points: list of (frame_index, cx, cy)
        frames_present = {fr for fr, _, _ in points}
        # Require that this floe appears in every frame exactly once
        if len(frames_present) != num_frames:
            continue

        # Sort by frame index
        points_sorted = sorted(points, key=lambda x: x[0])

        dxs, dys = [], []
        for (f0, x0, y0), (f1, x1, y1) in zip(points_sorted[:-1], points_sorted[1:]):
            # assuming f1 = f0 + 1
            dxs.append(x1 - x0)
            dys.append(y1 - y0)

        if not dxs:
            continue

        mean_dx = float(np.mean(dxs))
        mean_dy = float(np.mean(dys))

        vx = mean_dx / dt_seconds
        vy = mean_dy / dt_seconds

        # Image coordinates: x right, y down.
        # If you want conventional math angle (y up), flip sign of vy in atan2.
        speed = float(np.hypot(vx, vy))
        direction_rad = float(np.arctan2(-vy, vx))  # -vy to treat up as positive
        direction_deg = float(np.degrees(direction_rad))

        results.append({
            "track_id": tid,
            "avg_vx_px_s": vx,
            "avg_vy_px_s": vy,
            "speed_px_s": speed,
            "direction_deg": direction_deg,
            "points": points_sorted,
        })

    return results

def plot_floe_trajectories(base_img_gray,
                           track_centroids,
                           track_ids=None,
                           title="Floe trajectories",
                           save_path=None):
    """
    Plot trajectories for one or more floe tracks on top of a grayscale image.

    Parameters:
        base_img_gray: 2D array, grayscale image to use as background (e.g. frame 0)
        track_centroids: dict track_id -> list of (frame_index, cx, cy)
        track_ids: list of track_ids to plot. If None, plot all tracks in track_centroids.
        title: plot title
        save_path: if not None, save figure to this path
    """
    if track_ids is None:
        track_ids = sorted(track_centroids.keys())

    plt.figure(figsize=(8, 6))
    plt.imshow(base_img_gray, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # Color cycle
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(track_ids))))

    for k, tid in enumerate(track_ids):
        if tid not in track_centroids:
            continue

        points = track_centroids[tid]
        # sort by frame index
        points_sorted = sorted(points, key=lambda x: x[0])
        frames = [p[0] for p in points_sorted]
        xs     = [p[1] for p in points_sorted]
        ys     = [p[2] for p in points_sorted]

        col = colors[k % len(colors)]

        # Plot trajectory line
        plt.plot(xs, ys, '-', marker='o', markersize=3, linewidth=1.5,
                 color=col, label=f"track {tid} (frames {frames[0]}–{frames[-1]})")

        # Mark start and end more clearly
        plt.scatter(xs[0], ys[0], s=40, color=col, edgecolors='black', linewidths=1.0)
        plt.scatter(xs[-1], ys[-1], s=40, color=col, marker='X', edgecolors='black', linewidths=1.0)

    if len(track_ids) <= 15:
        plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved trajectory plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    _timelapse = Path(__file__).parent.parent / "data" / "thermal_timelapse"
    img1_path = str(_timelapse / "DJI_20250218031618_0007_T.JPG")
    img2_path = str(_timelapse / "DJI_20250218031620_0008_T.JPG")

    processor1= FloeSeparator(img1_path)
    processor2 = FloeSeparator(img2_path)

    # run preprocess to populate self.img_gray, self.img_rgb, self.mask
    img_gray, img_rgb, mask = processor1.preprocess(clahe_clip=1.5, clahe_grid=(8,8), thresh_val=75)
    img_gray2, img_rgb2, mask2 = processor2.preprocess(clahe_clip=1.5, clahe_grid=(8,8), thresh_val=75)

    # select one of the splitting methods:
    # labels_filtered, areas_px, centroids = processor1.split_simple(mask, min_area=30) # no erosion, just use binary mask
    # labels_filtered, areas_px, centroids = processor1.split_erode_dilate(mask, min_area=10) # 1px erosion/dilation
    labels_filtered, areas_px, centroids = processor1.split_erode_dilate_small(mask, min_area=30) # 1px erosion/dilation
    labels_filtered2, areas_px2, centroids2 = processor2.split_erode_dilate_small(mask2, min_area=30)

    # Folder containing thermal TIFFs
    thermal_dir = Path(__file__).parent.parent / "data" / "thermal_tif"

    if not thermal_dir.exists():
        raise FileNotFoundError(f"Folder not found: {thermal_dir}")

    tif_files = sorted(thermal_dir.glob("*.tif"))

    print(f"[INFO] Found {len(tif_files)} TIFF files in {thermal_dir}\n")

    # for tif_path in tif_files:
    #     arr = load_thermal_tif(tif_path)
    #     arr = np.rot90(arr, 2)
    #     visualize_thermal(arr, title=tif_path.name)

    arr = load_thermal_tif(tif_files[0])
    arr = np.rot90(arr, 2)
    plot_temperature_histogram(arr, bins=100, title=f"Temperature Histogram: {tif_files[0].name}")
    visualize_thermal(arr, title=tif_files[0].name)

    avg_temps, temp_list = compute_average_temps(arr, labels_filtered)
    plot_floe_temperature_hist(temp_list)

    csv_out = "floe_temperature_stats.csv"
    export_floe_temperature_csv(arr, labels_filtered, areas_px, centroids, csv_out)

    # below code visualizes and saves outputs
    processor1.visualize(img_gray, img_rgb, mask, labels_filtered, centroids, areas_px)

    # match floes between frame 1 and 2
    matches, unmatched1, unmatched2 = match_floes(
        labels_filtered,  centroids,  areas_px,
        labels_filtered2, centroids2, areas_px2,
        max_centroid_dist=40,   # tweak based on expected motion
        min_iou=0.05            # tweak based on overlap
    )

    print(f"Total floes t0: {len(centroids)}, t1: {len(centroids2)}")
    print(f"Matched floes: {len(matches)}")
    print(f"Unmatched in t0 (no match in t1): {unmatched1}")
    print(f"Unmatched in t1 (new / disappeared): {unmatched2}")

    for i, j, dx, dy, dist, iou in matches:
        print(f"floe {i+1} → floe {j+1}: "
              f"dx={dx:.2f}, dy={dy:.2f}, |d|={dist:.2f}, IoU={iou:.3f}")

    # visualize motion on the first frame
    visualize_motion(img_gray, centroids, matches,
                     save_path='floe_motion_vectors.png')

    '''erm'''
    # visible_path = 'sea_ice_analysis/sea_ice.jpg'
    # vis_bgr = cv.imread(visible_path)
    # if vis_bgr is None:
    #     raise FileNotFoundError(f"Could not read visible image at: {os.path.abspath(visible_path)}")
    # vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)

    # overlay_numbered_vis = register_and_overlay_visible(
    #     thermal_gray=img_gray,
    #     thermal_labels=labels_filtered,
    #     thermal_centroids=centroids,
    #     visible_rgb=vis_rgb,
    #     n_points=4,
    #     ransac_thresh=3.0
    # )

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.title("Visible image")
    # plt.imshow(vis_rgb)
    # plt.axis('off')
    # plt.subplot(122)
    # plt.title("Thermal edges overlaid on visible")
    # plt.imshow(overlay_numbered_vis)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('visible_overlay_preview.png', dpi=150)
    # plt.show()

    timelapse_folder = str(Path(__file__).parent.parent / "data" / "thermal_timelapse")

    # 1) Load and segment all frames
    img_grays, img_rgbs, labels_list, areas_list, cents_list, img_paths = load_and_segment_sequence(
        timelapse_folder,
        clahe_clip=1.5,
        clahe_grid=(8, 8),
        thresh_val=75,
        min_area=30,
        erode_min_size=500
    )

    print(f"[INFO] Loaded and segmented {len(img_paths)} frames.")

    # 2) Track floes over sequence
    track_centroids, num_frames = track_floes_over_sequence(
        labels_list,
        areas_list,
        cents_list,
        max_centroid_dist=40,
        min_iou=0.05
    )

    # 3) Compute average velocity for floes present in ALL frames
    dt_seconds = 2.0  # images taken every 2 s
    full_track_results = compute_avg_velocity_for_full_tracks(
        track_centroids,
        num_frames,
        dt_seconds=dt_seconds
    )

    full_track_ids = [res["track_id"] for res in full_track_results]
    subset_ids = full_track_ids[40:45]

    plot_floe_trajectories(
        base_img_gray=img_grays[0],
        track_centroids=track_centroids,
        track_ids=subset_ids,
        title="Trajectories of selected floes",
        save_path="floe_trajectories_selected.png"
    )

    print(f"[INFO] Found {len(full_track_results)} floes tracked across all {num_frames} frames.\n")

    for res in full_track_results:
        tid = res["track_id"]
        vx  = res["avg_vx_px_s"]
        vy  = res["avg_vy_px_s"]
        spd = res["speed_px_s"]
        ang = res["direction_deg"]
        print(f"Track {tid}: "
              f"avg vx = {vx:.3f} px/s, "
              f"avg vy = {vy:.3f} px/s, "
              f"speed = {spd:.3f} px/s, "
              f"direction = {ang:.1f}°")


def characterize_floes(labels_filtered: np.ndarray) -> list:
    """Compute shape metrics for each labeled floe via ellipse fitting."""
    results = []
    n = int(labels_filtered.max())
    for lbl in range(1, n + 1):
        comp = (labels_filtered == lbl).astype(np.uint8) * 255
        area_px = int((comp > 0).sum())
        if area_px == 0:
            continue
        contours, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv.contourArea)
        perimeter = cv.arcLength(cnt, True)
        circularity = float(4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0.0
        circularity = round(min(circularity, 1.0), 4)
        if len(cnt) >= 5:
            _, (w, h), _ = cv.fitEllipse(cnt)
            semi_major = round(float(max(w, h)) / 2, 2)
            semi_minor = round(float(min(w, h)) / 2, 2)
        else:
            r = float(np.sqrt(area_px / np.pi))
            semi_major = round(r, 2)
            semi_minor = round(r, 2)
        aspect_ratio = round(semi_minor / semi_major, 4) if semi_major > 0 else 0.0
        results.append({
            "label":         lbl,
            "area_px":       area_px,
            "perimeter_px":  round(float(perimeter), 2),
            "semi_major_px": semi_major,
            "semi_minor_px": semi_minor,
            "aspect_ratio":  aspect_ratio,
            "circularity":   circularity,
        })
    return results


def compute_fractal_dimension(areas, perimeters, min_floes=10, n_bootstrap=1000, rng_seed=42) -> dict:
    """Estimate boundary fractal dimension D via log-log perimeter-area regression."""
    areas      = np.asarray(areas,      dtype=np.float64)
    perimeters = np.asarray(perimeters, dtype=np.float64)
    ok = (areas > 0) & (perimeters > 0) & np.isfinite(areas) & np.isfinite(perimeters)
    A  = areas[ok]
    P  = perimeters[ok]
    result = {
        "valid": False, "n_floes": int(ok.sum()),
        "D": None, "slope": None, "intercept": None,
        "r_squared": None, "D_ci_low": None, "D_ci_high": None,
    }
    if result["n_floes"] < min_floes:
        return result
    logA = np.log(A)
    logP = np.log(P)
    slope, intercept = np.polyfit(logA, logP, 1)
    logP_pred = slope * logA + intercept
    ss_res = np.sum((logP - logP_pred) ** 2)
    ss_tot = np.sum((logP - logP.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rng = np.random.default_rng(rng_seed)
    n   = len(A)
    boot_slopes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_slopes[i] = np.polyfit(logA[idx], logP[idx], 1)[0]
    ci_low, ci_high = np.percentile(boot_slopes * 2, [2.5, 97.5])
    result.update({
        "valid":     True,
        "D":         round(float(slope * 2), 4),
        "slope":     round(float(slope),     6),
        "intercept": round(float(intercept), 6),
        "r_squared": round(r2,               4),
        "D_ci_low":  round(float(ci_low),    4),
        "D_ci_high": round(float(ci_high),   4),
    })
    return result


def measure_raw_contours(mask_u8: np.ndarray, min_area_px: int = 30):
    """Extract area and perimeter for each connected component in a raw binary mask."""
    contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    areas, perimeters = [], []
    for cnt in contours:
        a = float(cv.contourArea(cnt))
        if a < min_area_px:
            continue
        p = float(cv.arcLength(cnt, True))
        if p > 0:
            areas.append(a)
            perimeters.append(p)
    return areas, perimeters