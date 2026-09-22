"""Experimental: Interactive GUI for manual thermal-to-visible homography calibration (drag corners)."""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- USER CONFIG: set your image paths here ---
VISIBLE_PATH = str(Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice.jpg")
THERMAL_PATH = str(Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice_thermal.jpg")

ALPHA = 0.5  # transparency of thermal overlay (0=only visible, 1=only thermal)


class HomographyGUI:
    def __init__(self, vis_bgr, th_bgr, alpha=0.5):
        self.alpha = alpha

        # Store images (convert to RGB for matplotlib)
        self.vis_rgb = cv.cvtColor(vis_bgr, cv.COLOR_BGR2RGB)
        self.th_rgb = cv.cvtColor(th_bgr, cv.COLOR_BGR2RGB)

        self.h_vis, self.w_vis = self.vis_rgb.shape[:2]
        self.h_th, self.w_th = self.th_rgb.shape[:2]

        # Source corners in thermal image coordinates
        self.src_pts = np.float32([
            [0,           0],
            [self.w_th-1, 0],
            [self.w_th-1, self.h_th-1],
            [0,           self.h_th-1],
        ])

        # Initial guess for destination corners: small rect near bottom-left
        scale = min(self.w_vis / (3 * self.w_th), self.h_vis / (3 * self.h_th))
        w0 = int(self.w_th * scale)
        h0 = int(self.h_th * scale)
        x0 = 50
        y0 = self.h_vis - h0 - 50

        self.dst_pts = np.float32([
            [x0,       y0],
            [x0 + w0,  y0],
            [x0 + w0,  y0 + h0],
            [x0,       y0 + h0],
        ])

        # Homography placeholder
        self.H = None

        # Matplotlib figure / axes
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_title("Drag red corners, press Enter when done")
        self.ax.set_axis_off()

        # Show initial overlay
        overlay = self.compute_overlay()
        self.im_artist = self.ax.imshow(overlay)

        # Plot corners & connecting quad
        xs = self.dst_pts[:, 0]
        ys = self.dst_pts[:, 1]
        (self.corners_plot,) = self.ax.plot(
            xs, ys, "ro", markersize=8, markeredgecolor="black"
        )
        (self.quad_plot,) = self.ax.plot(
            list(xs) + [xs[0]], list(ys) + [ys[0]], "r-", linewidth=2
        )

        # For dragging state
        self.active_idx = None  # which corner is being dragged (0–3)

        # Connect event handlers
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_move = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_move
        )
        self.cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

    # --- Core functions ---

    def compute_overlay(self):
        """Compute overlay image given current dst_pts and src_pts."""
        self.H = cv.getPerspectiveTransform(self.src_pts, self.dst_pts)

        warped_th = cv.warpPerspective(
            self.th_rgb,
            self.H,
            (self.w_vis, self.h_vis),
        )

        overlay = cv.addWeighted(
            self.vis_rgb, 1.0 - self.alpha, warped_th, self.alpha, 0
        )
        return overlay

    def update_display(self):
        """Recompute overlay and redraw everything."""
        overlay = self.compute_overlay()

        self.im_artist.set_data(overlay)

        xs = self.dst_pts[:, 0]
        ys = self.dst_pts[:, 1]
        self.corners_plot.set_data(xs, ys)
        self.quad_plot.set_data(
            list(xs) + [xs[0]], list(ys) + [ys[0]]
        )

        self.fig.canvas.draw_idle()

    # --- Event handlers ---

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        pts = self.dst_pts

        # Pick nearest corner if within some radius
        dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        idx = np.argmin(dists)
        if dists[idx] < 20:  # pixel threshold
            self.active_idx = idx

    def on_release(self, event):
        self.active_idx = None

    def on_move(self, event):
        if self.active_idx is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        # Clamp to image bounds
        x = max(0, min(self.w_vis - 1, x))
        y = max(0, min(self.h_vis - 1, y))

        self.dst_pts[self.active_idx] = [x, y]
        self.update_display()

    def on_key(self, event):
        if event.key == "enter":
            print("\nFinal destination corners (visible image coords):")
            print(self.dst_pts)
            print("\nHomography H (thermal -> visible):")
            print(self.H)
            print("\nClose the window to end.")
        elif event.key == "r":
            # Reset to initial rectangle
            print("Resetting corners.")
            self.__init__(cv.cvtColor(self.vis_rgb, cv.COLOR_RGB2BGR),
                          cv.cvtColor(self.th_rgb, cv.COLOR_RGB2BGR),
                          alpha=self.alpha)
            plt.close(self.fig)  # close current and reopen in new init

    def run(self):
        plt.show()
        # When the window is closed, self.H and self.dst_pts form your result
        print("\n=== Session ended ===")
        print("Final H:")
        print(self.H)
        print("Final dst_pts:")
        print(self.dst_pts)


def main():
    vis_bgr = cv.imread(VISIBLE_PATH, cv.IMREAD_COLOR)
    if vis_bgr is None:
        raise FileNotFoundError(f"Could not read visible image at: {VISIBLE_PATH}")

    th_bgr = cv.imread(THERMAL_PATH, cv.IMREAD_COLOR)
    if th_bgr is None:
        raise FileNotFoundError(f"Could not read thermal image at: {THERMAL_PATH}")

    gui = HomographyGUI(vis_bgr, th_bgr, alpha=ALPHA)
    gui.run()


if __name__ == "__main__":
    main()
