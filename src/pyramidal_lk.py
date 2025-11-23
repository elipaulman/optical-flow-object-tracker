import cv2
import numpy as np

from .vision import build_pyramid_with_gradients


class PyramidalLucasKanade:
    """Custom implementation of pyramidal Lucas-Kanade optical flow.

    Lucas-Kanade tracks points by assuming pixels around a point move together
    (constant motion in a small window). Pyramidal LK uses multiple image scales
    to handle large motions - starts coarse and refines at finer scales.
    """

    def __init__(self, window_size=9, max_iters=10, epsilon=0.01, levels=3):
        # make sure window size is odd (needed for symmetric patches)
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        # max iterations for convergence at each pyramid level
        self.max_iters = max_iters
        # threshold for when to stop iterating (convergence criterion)
        self.epsilon = epsilon
        # num of pyramid levels (coarse to fine tracking)
        self.levels = levels

    def track(self, prev_img, curr_img, points):
        """
        Track points from prev_img to curr_img.

        Args:
            prev_img: float32 grayscale image.
            curr_img: float32 grayscale image.
            points: Nx2 array of point coordinates (x, y) in prev_img.
        Returns:
            next_points: Nx2 array of tracked coordinates.
            status: Nx1 array indicating success per point.
            errors: Nx1 array of photometric residuals.
        """
        # edge case: no points to track
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

        # build image pyramids (coarse to fine) + compute gradients for prev image
        prev_pyr, prev_grads = build_pyramid_with_gradients(prev_img, self.levels)
        curr_pyr, _ = build_pyramid_with_gradients(curr_img, self.levels)

        # init output arrays for where points moved to
        next_points = np.zeros_like(points, dtype=np.float32)
        status = np.zeros((len(points),), dtype=np.uint8)  # 1 = success, 0 = fail
        errors = np.zeros((len(points),), dtype=np.float32)  # tracking error per point

        # half window size for bounds checking
        half_w = self.window_size // 2

        # track each point independently
        for idx, pt in enumerate(points):
            flow = np.zeros(2, dtype=np.float32)  # accumulated motion vector (x, y)
            ok = True  # flag if tracking succeeded
            last_residual = 0.0  # final photometric error

            # go from coarsest to finest level (reversed pyramid)
            for lvl in reversed(range(self.levels)):
                scale = float(2 ** lvl)  # scale factor for this level
                inv_scale = 1.0 / scale
                pt_lvl = pt * inv_scale  # scale point coords to current level

                # grab the pyramid level and gradients
                prev_level = prev_pyr[lvl]
                curr_level = curr_pyr[lvl]
                grad_x_level, grad_y_level = prev_grads[lvl]
                h, w = prev_level.shape

                # iterative refinement at this level
                for _ in range(self.max_iters):
                    pos = pt_lvl + flow  # current estimate of point position
                    x, y = pos
                    # check if we're too close to image borders
                    if x < half_w or x >= (w - half_w) or y < half_w or y >= (h - half_w):
                        ok = False
                        break

                    # extract patches around the point from prev & curr frames
                    patch_prev = cv2.getRectSubPix(prev_level, (self.window_size, self.window_size), (x, y))
                    patch_curr = cv2.getRectSubPix(curr_level, (self.window_size, self.window_size), (x, y))
                    gx = cv2.getRectSubPix(grad_x_level, (self.window_size, self.window_size), (x, y))
                    gy = cv2.getRectSubPix(grad_y_level, (self.window_size, self.window_size), (x, y))

                    # flatten patches into 1D arrays for math
                    gx_flat = gx.reshape(-1)
                    gy_flat = gy.reshape(-1)
                    it = (patch_curr - patch_prev).reshape(-1)  # temporal diff

                    # build the structure tensor (2x2 matrix A^T A)
                    # LK assumes: patch_curr ≈ patch_prev + gx*dx + gy*dy
                    # we're solving for [dx, dy] that minimizes the brightness error
                    a11 = np.dot(gx_flat, gx_flat)
                    a12 = np.dot(gx_flat, gy_flat)
                    a22 = np.dot(gy_flat, gy_flat)

                    # check if matrix is invertible
                    det = a11 * a22 - a12 * a12
                    if det < 1e-6:  # singular matrix = can't solve (flat region)
                        ok = False
                        break

                    # compute right-hand side vector (A^T b)
                    b1 = -np.dot(gx_flat, it)
                    b2 = -np.dot(gy_flat, it)

                    # solve for optical flow update using inverse 2x2 matrix formula
                    # this is the least-squares solution to the LK equation
                    inv_det = 1.0 / det
                    dx = inv_det * (a22 * b1 - a12 * b2)
                    dy = inv_det * (-a12 * b1 + a11 * b2)
                    delta = np.array([dx, dy], dtype=np.float32)
                    flow += delta  # accumulate the update
                    last_residual = np.mean(np.abs(it))  # track the error

                    # check if converged (delta is tiny)
                    if np.dot(delta, delta) < self.epsilon * self.epsilon:
                        break

                # if tracking failed at this level, bail out
                if not ok:
                    break

                # scale flow up for next finer level (unless already at finest)
                if lvl != 0:
                    flow *= 2.0

            # save result if tracking succeeded
            if ok:
                next_points[idx] = pt + flow
                status[idx] = 1
                errors[idx] = last_residual

        return next_points, status, errors
