import cv2
import numpy as np

from .vision import build_pyramid_with_gradients


class PyramidalLucasKanade:
    """Custom implementation of pyramidal Lucas-Kanade optical flow."""

    def __init__(self, window_size=9, max_iters=10, epsilon=0.01, levels=3):
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.max_iters = max_iters
        self.epsilon = epsilon
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
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

        prev_pyr, prev_grads = build_pyramid_with_gradients(prev_img, self.levels)
        curr_pyr, _ = build_pyramid_with_gradients(curr_img, self.levels)

        next_points = np.zeros_like(points, dtype=np.float32)
        status = np.zeros((len(points),), dtype=np.uint8)
        errors = np.zeros((len(points),), dtype=np.float32)

        half_w = self.window_size // 2

        for idx, pt in enumerate(points):
            flow = np.zeros(2, dtype=np.float32)
            ok = True
            last_residual = 0.0

            for lvl in reversed(range(self.levels)):
                scale = float(2 ** lvl)
                inv_scale = 1.0 / scale
                pt_lvl = pt * inv_scale

                prev_level = prev_pyr[lvl]
                curr_level = curr_pyr[lvl]
                grad_x_level, grad_y_level = prev_grads[lvl]
                h, w = prev_level.shape

                for _ in range(self.max_iters):
                    pos = pt_lvl + flow
                    x, y = pos
                    if x < half_w or x >= (w - half_w) or y < half_w or y >= (h - half_w):
                        ok = False
                        break

                    patch_prev = cv2.getRectSubPix(prev_level, (self.window_size, self.window_size), (x, y))
                    patch_curr = cv2.getRectSubPix(curr_level, (self.window_size, self.window_size), (x, y))
                    gx = cv2.getRectSubPix(grad_x_level, (self.window_size, self.window_size), (x, y))
                    gy = cv2.getRectSubPix(grad_y_level, (self.window_size, self.window_size), (x, y))

                    gx_flat = gx.reshape(-1)
                    gy_flat = gy.reshape(-1)
                    it = (patch_curr - patch_prev).reshape(-1)

                    a11 = np.dot(gx_flat, gx_flat)
                    a12 = np.dot(gx_flat, gy_flat)
                    a22 = np.dot(gy_flat, gy_flat)

                    det = a11 * a22 - a12 * a12
                    if det < 1e-6:
                        ok = False
                        break

                    b1 = -np.dot(gx_flat, it)
                    b2 = -np.dot(gy_flat, it)

                    inv_det = 1.0 / det
                    dx = inv_det * (a22 * b1 - a12 * b2)
                    dy = inv_det * (-a12 * b1 + a11 * b2)
                    delta = np.array([dx, dy], dtype=np.float32)
                    flow += delta
                    last_residual = np.mean(np.abs(it))

                    if np.dot(delta, delta) < self.epsilon * self.epsilon:
                        break

                if not ok:
                    break

                if lvl != 0:
                    flow *= 2.0

            if ok:
                next_points[idx] = pt + flow
                status[idx] = 1
                errors[idx] = last_residual

        return next_points, status, errors
