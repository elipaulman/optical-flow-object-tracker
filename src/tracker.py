import cv2
import numpy as np

from .pyramidal_lk import PyramidalLucasKanade
from .vision import preprocess_frame, shi_tomasi_corners

'''
Orchestration layer: manages feature detection, tracking, and re-detection.
Uses RANSAC to reject outliers and estimate robust object motion.
Optionally, tracks bright spots for small/highlighted objects.
'''

# this is our additional feature mentioned in the project proposal to enhance tracking
def ransac_translation(prev_pts, curr_pts, threshold=3.0, max_iters=100):
    """Robustly estimate translation between two point sets.

    RANSAC (RANdom SAmple Consensus) filters out bad tracking points (outliers)
    by repeatedly testing different motion hypotheses and picking the one most
    points agree with. This makes tracking robust to mismatched features.
    """
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return np.zeros(2, dtype=np.float32), np.zeros((len(prev_pts),), dtype=bool)

    best_inliers = np.zeros((len(prev_pts),), dtype=bool)
    best_translation = np.zeros(2, dtype=np.float32)

    # try random point pairs to find consensus motion
    for _ in range(max_iters):
        idx = np.random.randint(0, len(prev_pts))
        translation = curr_pts[idx] - prev_pts[idx]  # hypothesis: all points move like this
        # check which points fit this motion model
        residuals = np.linalg.norm((prev_pts + translation) - curr_pts, axis=1)
        inliers = residuals < threshold  # points that moved consistently
        if inliers.sum() > best_inliers.sum():  # found better consensus
            best_inliers = inliers
            best_translation = translation

    if best_inliers.sum() == 0:
        # fallback: use median translation if RANSAC fails
        translations = curr_pts - prev_pts
        best_translation = np.median(translations, axis=0)
        residuals = np.linalg.norm((prev_pts + best_translation) - curr_pts, axis=1)
        best_inliers = residuals < threshold

    return best_translation, best_inliers


class FeatureTracker:
    """Manage feature lifecycle and motion estimation.

    Main tracking class that:
    1. Detects good features (corners) to track
    2. Tracks them frame-to-frame with Lucas-Kanade
    3. Filters outliers with RANSAC
    4. Re-detects features when needed
    5. Estimates object motion from feature movement
    """

    def __init__(
        self,
        max_corners=300,  # max num of corner features to detect
        quality_level=0.01,  # min quality of corners (lower = more corners but weaker)
        min_distance=5,  # min spacing between detected corners (pixels)
        block_size=5,  # neighborhood size for corner detection
        lk_window=9,  # window size for Lucas-Kanade tracking
        lk_levels=3,  # num pyramid levels for LK (more = handle larger motion)
        lk_iters=10,  # max iterations per pyramid level
        lk_epsilon=0.01,  # convergence threshold for LK
        ransac_thresh=3.0,  # max pixel error to consider a point an inlier
        ransac_iters=100,  # num RANSAC iterations
        min_features=80,  # min features needed before re-detecting
        detect_interval=10,  # re-detect features every N frames
        intensity_thresh=None,  # only detect on pixels above this brightness (0-255)
        intensity_blur=0,  # blur kernel size before intensity thresholding
        auto_intensity_percentile=None,  # auto-compute intensity thresh from percentile
        exclude_rect=None,  # (x,y,w,h) rect to exclude from detection (e.g., watermark)
        bright_spot=False,  # enable brightest-spot tracking mode
        bright_spot_radius=60,  # search radius around last center for bright spot
        bright_spot_blur=3,  # blur before bright-spot search
        bright_spot_percentile=99.5,  # percentile thresh for bright regions
        bright_min_area=1,  # min blob area (pixels) for bright spot
        bright_max_area=120,  # max blob area to filter out large bright regions
        bright_max_aspect=3.0,  # max aspect ratio to reject elongated blobs
        bright_dist_weight=1.0,  # how much to penalize distance in bright spot scoring
        bright_max_jump=60.0,  # reject bright spots that jumped more than this
        bright_min_fill=0.2,  # min fill ratio (area/bbox_area) for blobs
        bright_max_dim=120,  # max width or height for bright blobs
    ):
        # corner detection params
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        # RANSAC params
        self.ransac_thresh = ransac_thresh
        self.ransac_iters = ransac_iters
        # feature management
        self.min_features = min_features
        self.detect_interval = detect_interval
        # intensity filtering
        self.intensity_thresh = intensity_thresh
        self.intensity_blur = intensity_blur
        self.auto_intensity_percentile = auto_intensity_percentile
        # masking
        self.exclude_rect = exclude_rect
        # bright-spot tracking params
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

        # create Lucas-Kanade tracker instance
        self.lk = PyramidalLucasKanade(
            window_size=lk_window,
            max_iters=lk_iters,
            epsilon=lk_epsilon,
            levels=lk_levels,
        )

        # state variables
        self.points = np.empty((0, 2), dtype=np.float32)  # current tracked points
        self.frame_idx = 0  # current frame number
        self.roi = None  # current region of interest (x,y,w,h)
        self.initial_roi_size = None  # original ROI size
        self.last_center = None  # last known object center

    def initialize(self, gray, roi):
        """Initialize tracker with first frame and region of interest."""
        self.roi = roi  # region to search for features
        if roi is not None:
            self.initial_roi_size = (roi[2], roi[3])  # remember original size
        # detect initial features to track
        self.points = self._detect(gray)
        # optional: use brightest spot as initial center (good for bright balls)
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
        """Create binary mask for where to detect/search for features."""
        mask = np.zeros(shape, dtype=np.uint8)
        if self.roi is None:
            mask[:, :] = 255  # search whole image
        else:
            # only search inside ROI
            x, y, w, h = self.roi
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask[y : y + h, x : x + w] = 255

        # exclude specific regions (e.g., watermarks)
        if self.exclude_rect is not None:
            ex, ey, ew, eh = self.exclude_rect
            ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
            mask[ey : ey + eh, ex : ex + ew] = 0
        return mask

    def _find_bright_spot(self, gray):
        """Find the brightest pixel (optionally local search near last center) within the ROI/mask.

        This is useful for tracking small bright objects (like tennis balls) by
        finding the brightest blob that meets size/shape constraints.
        """
        mask = self._roi_mask(gray.shape)
        # narrow search to area near last known position
        if self.last_center is not None and self.bright_spot_radius is not None and self.bright_spot_radius > 0:
            local_mask = np.zeros_like(mask)
            cx, cy = self.last_center
            cx, cy = float(cx), float(cy)
            cv2.circle(local_mask, (int(cx), int(cy)), int(self.bright_spot_radius), 255, -1)
            mask = cv2.bitwise_and(mask, local_mask)

        # optionally blur to reduce noise
        if self.bright_spot_blur and self.bright_spot_blur > 1:
            k = self.bright_spot_blur if self.bright_spot_blur % 2 == 1 else self.bright_spot_blur + 1
            g = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            g = gray

        masked = cv2.bitwise_and(g, g, mask=mask)
        pixels = masked[mask > 0]
        if pixels.size == 0:
            return None, 0.0

        # threshold high-percentile bright regions (filters out large bright areas like field lines)
        thresh_val = np.percentile(pixels, self.bright_spot_percentile)
        binary = np.zeros_like(masked, dtype=np.uint8)
        binary[masked >= thresh_val] = 255

        # find connected components (bright blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        best_idx = -1
        best_score = -1e9
        ref = self.last_center

        # evaluate each blob and pick the best one
        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            # reject blobs that are too small or too large
            if area < self.bright_min_area or area > self.bright_max_area:
                continue
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w > self.bright_max_dim or h > self.bright_max_dim:
                continue
            # reject elongated blobs (field lines)
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > self.bright_max_aspect:
                continue
            # reject sparse blobs
            fill = area / float(max(1, w * h))
            if fill < self.bright_min_fill:
                continue
            cx, cy = centroids[i]
            # score: prefer brighter blobs closer to last known position
            mask_i = (labels == i)
            max_val = masked[mask_i].max()
            score = max_val
            if ref is not None:
                dist = np.linalg.norm(np.array([cx, cy]) - ref)
                score -= dist * self.bright_dist_weight  # penalize distance
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            cx, cy = centroids[best_idx]
            return np.array([cx, cy], dtype=np.float32), best_score

        # fallback: just use brightest pixel if no good blobs
        _, max_val, _, max_loc = cv2.minMaxLoc(masked)
        if max_val <= 0:
            return None, 0.0
        return np.array(max_loc, dtype=np.float32), max_val

    def _detect(self, gray):
        """Detect new corner features to track."""
        mask = self._roi_mask(gray.shape)
        # optional: only detect on bright pixels (useful for bright objects on dark backgrounds)
        if self.intensity_thresh is not None or self.auto_intensity_percentile is not None:
            g = gray
            if self.intensity_blur and self.intensity_blur > 1:
                k = self.intensity_blur if self.intensity_blur % 2 == 1 else self.intensity_blur + 1
                g = cv2.GaussianBlur(gray, (k, k), 0)
            thresh_val = self.intensity_thresh
            # auto-compute threshold from percentile
            if thresh_val is None and self.auto_intensity_percentile is not None:
                roi_pixels = g[mask > 0]
                if roi_pixels.size > 0:
                    thresh_val = np.percentile(roi_pixels, self.auto_intensity_percentile)
            if thresh_val is None:
                bright = mask
            else:
                bright = (g >= thresh_val).astype(np.uint8) * 255  # binary mask of bright pixels
            mask = cv2.bitwise_and(mask, bright)  # combine with ROI mask
        # detect Shi-Tomasi corners within the masked region
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
        """Merge new points that are not too close to existing ones.

        When re-detecting features, we add new ones without duplicating
        points that are already being tracked.
        """
        if len(existing) == 0:
            return new_points
        merged = existing.tolist()
        for pt in new_points:
            # only add if far enough from all existing points
            if np.linalg.norm(existing - pt, axis=1).min() >= self.min_distance:
                merged.append(pt.tolist())
        return np.array(merged, dtype=np.float32)

    def update(self, prev_gray, curr_gray):
        """Track features, reject outliers, and refresh if needed.

        Main tracking pipeline per frame:
        1. Track existing features with LK
        2. Filter outliers with RANSAC
        3. Estimate object motion from inliers
        4. Re-detect features if needed
        5. Compute object center
        """
        self.frame_idx += 1
        # detect initial features if we have none
        if len(self.points) == 0:
            self.points = self._detect(prev_gray)

        # track all points from prev to curr frame
        next_points, status, errors = self.lk.track(prev_gray, curr_gray, self.points)

        # keep only successfully tracked points
        good_prev = self.points[status == 1]
        good_curr = next_points[status == 1]

        translation = np.zeros(2, dtype=np.float32)
        inlier_mask = np.zeros((len(good_prev),), dtype=bool)

        # use RANSAC to filter out outliers (mismatched tracks)
        if len(good_prev) > 0:
            translation, inlier_mask = ransac_translation(
                good_prev, good_curr, threshold=self.ransac_thresh, max_iters=self.ransac_iters
            )

        # final set of reliable tracked points
        inlier_prev = good_prev[inlier_mask]
        inlier_curr = good_curr[inlier_mask]

        # move the ROI along with the object
        if self.roi is not None and inlier_mask.size > 0:
            self.roi = (
                self.roi[0] + translation[0],
                self.roi[1] + translation[1],
                self.roi[2],
                self.roi[3],
            )

        # re-detect features if we lost too many or it's time to refresh
        need_redetect = len(inlier_curr) < self.min_features or self.frame_idx % self.detect_interval == 0
        if need_redetect:
            new_points = self._detect(curr_gray)
            self.points = self._merge_points(inlier_curr, new_points)
        else:
            self.points = inlier_curr

        # compute object center as mean of tracked points
        center = None
        if len(inlier_curr) > 0:
            center = np.mean(inlier_curr, axis=0)

        # optional bright-spot override (useful for tiny bright targets like tennis balls)
        # this can be more reliable than feature-based center for small objects
        if self.bright_spot:
            spot, _ = self._find_bright_spot(curr_gray)
            if spot is not None:
                # sanity check: reject if object jumped too far (likely wrong detection)
                if self.last_center is not None and self.bright_max_jump > 0:
                    if np.linalg.norm(spot - self.last_center) > self.bright_max_jump:
                        spot = None  # reject implausible jump
                if spot is not None:
                    center = spot  # override feature-based center
                    if self.last_center is not None:
                        translation = spot - self.last_center
                    # re-center ROI on bright spot
                    if self.initial_roi_size is not None:
                        w0, h0 = self.initial_roi_size
                        self.roi = (spot[0] - w0 * 0.5, spot[1] - h0 * 0.5, w0, h0)

        # remember center for next frame
        if center is not None:
            self.last_center = center

        return {
            "points": self.points,  # all current tracking points
            "prev_points": inlier_prev,  # reliable points in prev frame
            "curr_points": inlier_curr,  # reliable points in curr frame
            "translation": translation,  # estimated object motion
            "center": center,  # estimated object center
        }
