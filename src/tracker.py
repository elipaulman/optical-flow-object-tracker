import cv2
import numpy as np

from .pyramidal_lk import PyramidalLucasKanade
from .vision import preprocess_frame, shi_tomasi_corners


def ransac_translation(prev_pts, curr_pts, threshold=3.0, max_iters=100):
    """Robustly estimate translation between two point sets."""
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return np.zeros(2, dtype=np.float32), np.zeros((len(prev_pts),), dtype=bool)

    best_inliers = np.zeros((len(prev_pts),), dtype=bool)
    best_translation = np.zeros(2, dtype=np.float32)

    for _ in range(max_iters):
        idx = np.random.randint(0, len(prev_pts))
        translation = curr_pts[idx] - prev_pts[idx]
        residuals = np.linalg.norm((prev_pts + translation) - curr_pts, axis=1)
        inliers = residuals < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_translation = translation

    if best_inliers.sum() == 0:
        # Fallback to median translation if RANSAC fails
        translations = curr_pts - prev_pts
        best_translation = np.median(translations, axis=0)
        residuals = np.linalg.norm((prev_pts + best_translation) - curr_pts, axis=1)
        best_inliers = residuals < threshold

    return best_translation, best_inliers


class FeatureTracker:
    """Manage feature lifecycle and motion estimation."""

    def __init__(
        self,
        max_corners=300,
        quality_level=0.01,
        min_distance=5,
        block_size=5,
        lk_window=9,
        lk_levels=3,
        lk_iters=10,
        lk_epsilon=0.01,
        ransac_thresh=3.0,
        ransac_iters=100,
        min_features=80,
        detect_interval=10,
        intensity_thresh=None,
        intensity_blur=0,
        auto_intensity_percentile=None,
        exclude_rect=None,
        bright_spot=False,
        bright_spot_radius=60,
        bright_spot_blur=3,
        bright_spot_percentile=99.5,
        bright_min_area=1,
        bright_max_area=120,
        bright_max_aspect=3.0,
        bright_dist_weight=1.0,
        bright_max_jump=60.0,
        bright_min_fill=0.2,
        bright_max_dim=120,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.ransac_thresh = ransac_thresh
        self.ransac_iters = ransac_iters
        self.min_features = min_features
        self.detect_interval = detect_interval
        self.intensity_thresh = intensity_thresh
        self.intensity_blur = intensity_blur
        self.auto_intensity_percentile = auto_intensity_percentile
        self.exclude_rect = exclude_rect
        self.bright_spot = bright_spot
        self.bright_spot_radius = bright_spot_radius
        self.bright_spot_blur = bright_spot_blur
        self.bright_spot_percentile = bright_spot_percentile
        self.bright_min_area = bright_min_area
        self.bright_max_area = bright_max_area
        self.bright_max_aspect = bright_max_aspect
        self.bright_dist_weight = bright_dist_weight
        self.bright_max_jump = bright_max_jump
        self.bright_min_fill = bright_min_fill
        self.bright_max_dim = bright_max_dim

        self.lk = PyramidalLucasKanade(
            window_size=lk_window,
            max_iters=lk_iters,
            epsilon=lk_epsilon,
            levels=lk_levels,
        )

        self.points = np.empty((0, 2), dtype=np.float32)
        self.frame_idx = 0
        self.roi = None
        self.initial_roi_size = None
        self.last_center = None

    def initialize(self, gray, roi):
        """Initialize tracker with first frame and region of interest."""
        self.roi = roi
        if roi is not None:
            self.initial_roi_size = (roi[2], roi[3])
        self.points = self._detect(gray)
        if self.bright_spot:
            spot, _ = self._find_bright_spot(gray)
            if spot is not None:
                self.last_center = spot
                if self.initial_roi_size is not None:
                    w0, h0 = self.initial_roi_size
                    self.roi = (spot[0] - w0 * 0.5, spot[1] - h0 * 0.5, w0, h0)
        self.frame_idx = 0
        return self.points

    def _roi_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if self.roi is None:
            mask[:, :] = 255
        else:
            x, y, w, h = self.roi
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask[y : y + h, x : x + w] = 255

        if self.exclude_rect is not None:
            ex, ey, ew, eh = self.exclude_rect
            ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
            mask[ey : ey + eh, ex : ex + ew] = 0
        return mask

    def _find_bright_spot(self, gray):
        """Find the brightest pixel (optionally local search near last center) within the ROI/mask."""
        mask = self._roi_mask(gray.shape)
        if self.last_center is not None and self.bright_spot_radius is not None and self.bright_spot_radius > 0:
            local_mask = np.zeros_like(mask)
            cx, cy = self.last_center
            cx, cy = float(cx), float(cy)
            cv2.circle(local_mask, (int(cx), int(cy)), int(self.bright_spot_radius), 255, -1)
            mask = cv2.bitwise_and(mask, local_mask)

        if self.bright_spot_blur and self.bright_spot_blur > 1:
            k = self.bright_spot_blur if self.bright_spot_blur % 2 == 1 else self.bright_spot_blur + 1
            g = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            g = gray

        masked = cv2.bitwise_and(g, g, mask=mask)
        pixels = masked[mask > 0]
        if pixels.size == 0:
            return None, 0.0

        # Threshold high-percentile bright regions to avoid long field lines
        thresh_val = np.percentile(pixels, self.bright_spot_percentile)
        binary = np.zeros_like(masked, dtype=np.uint8)
        binary[masked >= thresh_val] = 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        best_idx = -1
        best_score = -1e9
        ref = self.last_center

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.bright_min_area or area > self.bright_max_area:
                continue
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w > self.bright_max_dim or h > self.bright_max_dim:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > self.bright_max_aspect:
                continue
            fill = area / float(max(1, w * h))
            if fill < self.bright_min_fill:
                continue
            cx, cy = centroids[i]
            # Score: brighter and closer to last center if available
            mask_i = (labels == i)
            max_val = masked[mask_i].max()
            score = max_val
            if ref is not None:
                dist = np.linalg.norm(np.array([cx, cy]) - ref)
                score -= dist * self.bright_dist_weight  # prefer closer blobs
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            cx, cy = centroids[best_idx]
            return np.array([cx, cy], dtype=np.float32), best_score

        # Fallback to single max pixel
        _, max_val, _, max_loc = cv2.minMaxLoc(masked)
        if max_val <= 0:
            return None, 0.0
        return np.array(max_loc, dtype=np.float32), max_val

    def _detect(self, gray):
        mask = self._roi_mask(gray.shape)
        if self.intensity_thresh is not None or self.auto_intensity_percentile is not None:
            g = gray
            if self.intensity_blur and self.intensity_blur > 1:
                k = self.intensity_blur if self.intensity_blur % 2 == 1 else self.intensity_blur + 1
                g = cv2.GaussianBlur(gray, (k, k), 0)
            thresh_val = self.intensity_thresh
            if thresh_val is None and self.auto_intensity_percentile is not None:
                roi_pixels = g[mask > 0]
                if roi_pixels.size > 0:
                    thresh_val = np.percentile(roi_pixels, self.auto_intensity_percentile)
            if thresh_val is None:
                bright = mask
            else:
                bright = (g >= thresh_val).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, bright)
        corners = shi_tomasi_corners(
            gray,
            max_corners=self.max_corners,
            quality_level=self.quality_level,
            min_distance=self.min_distance,
            block_size=self.block_size,
            mask=mask,
        )
        return corners

    def _merge_points(self, existing, new_points):
        """Merge new points that are not too close to existing ones."""
        if len(existing) == 0:
            return new_points
        merged = existing.tolist()
        for pt in new_points:
            if np.linalg.norm(existing - pt, axis=1).min() >= self.min_distance:
                merged.append(pt.tolist())
        return np.array(merged, dtype=np.float32)

    def update(self, prev_gray, curr_gray):
        """Track features, reject outliers, and refresh if needed."""
        self.frame_idx += 1
        if len(self.points) == 0:
            self.points = self._detect(prev_gray)

        next_points, status, errors = self.lk.track(prev_gray, curr_gray, self.points)

        good_prev = self.points[status == 1]
        good_curr = next_points[status == 1]

        translation = np.zeros(2, dtype=np.float32)
        inlier_mask = np.zeros((len(good_prev),), dtype=bool)

        if len(good_prev) > 0:
            translation, inlier_mask = ransac_translation(
                good_prev, good_curr, threshold=self.ransac_thresh, max_iters=self.ransac_iters
            )

        inlier_prev = good_prev[inlier_mask]
        inlier_curr = good_curr[inlier_mask]

        if self.roi is not None and inlier_mask.size > 0:
            self.roi = (
                self.roi[0] + translation[0],
                self.roi[1] + translation[1],
                self.roi[2],
                self.roi[3],
            )

        need_redetect = len(inlier_curr) < self.min_features or self.frame_idx % self.detect_interval == 0
        if need_redetect:
            new_points = self._detect(curr_gray)
            self.points = self._merge_points(inlier_curr, new_points)
        else:
            self.points = inlier_curr

        center = None
        if len(inlier_curr) > 0:
            center = np.mean(inlier_curr, axis=0)

        # Optional bright-spot override (useful for tiny, bright targets)
        if self.bright_spot:
            spot, _ = self._find_bright_spot(curr_gray)
            if spot is not None:
                if self.last_center is not None and self.bright_max_jump > 0:
                    if np.linalg.norm(spot - self.last_center) > self.bright_max_jump:
                        spot = None  # reject implausible jump
                if spot is not None:
                    center = spot
                    if self.last_center is not None:
                        translation = spot - self.last_center
                    if self.initial_roi_size is not None:
                        w0, h0 = self.initial_roi_size
                        self.roi = (spot[0] - w0 * 0.5, spot[1] - h0 * 0.5, w0, h0)

        if center is not None:
            self.last_center = center

        return {
            "points": self.points,
            "prev_points": inlier_prev,
            "curr_points": inlier_curr,
            "translation": translation,
            "center": center,
        }
