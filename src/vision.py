import cv2
import numpy as np


def preprocess_frame(frame, blur_ksize=5):
    """Convert BGR frame to blurred grayscale float32."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray.astype(np.float32)


def gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid with the requested number of levels."""
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def compute_gradients(image):
    """Compute spatial image gradients for a single grayscale level."""
    ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_REFLECT101)
    iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_REFLECT101)
    return ix, iy


def build_pyramid_with_gradients(image, levels):
    """Return Gaussian pyramid plus per-level gradients."""
    pyr = gaussian_pyramid(image, levels)
    grads = [compute_gradients(level) for level in pyr]
    return pyr, grads


def _non_maximum_suppression(response, threshold, min_distance):
    """Return coordinates of local maxima above threshold with spacing enforced."""
    dilated = cv2.dilate(response, None)
    maxima_mask = (response == dilated) & (response >= threshold)
    coords = np.column_stack(np.nonzero(maxima_mask))
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Sort by response strength descending
    values = response[maxima_mask]
    order = np.argsort(values)[::-1]
    coords = coords[order]

    selected = []
    min_dist_sq = min_distance * min_distance
    for y, x in coords:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist_sq for sy, sx in selected):
            selected.append((y, x))
            if len(selected) >= len(coords):
                break
    if not selected:
        return np.empty((0, 2), dtype=np.float32)
    selected = np.array(selected, dtype=np.float32)
    return np.flip(selected, axis=1)  # return as (x, y)


def shi_tomasi_corners(
    gray,
    max_corners=200,
    quality_level=0.01,
    min_distance=5,
    block_size=5,
    mask=None,
):
    """Compute Shi-Tomasi response and pick strong corners."""
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    sxx = cv2.boxFilter(ix2, ddepth=-1, ksize=(block_size, block_size), normalize=False, borderType=cv2.BORDER_REFLECT101)
    syy = cv2.boxFilter(iy2, ddepth=-1, ksize=(block_size, block_size), normalize=False, borderType=cv2.BORDER_REFLECT101)
    sxy = cv2.boxFilter(ixy, ddepth=-1, ksize=(block_size, block_size), normalize=False, borderType=cv2.BORDER_REFLECT101)

    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    inside = np.maximum(trace * trace - 4.0 * det, 0.0)
    lambda_min = 0.5 * (trace - np.sqrt(inside))

    response = lambda_min
    if mask is not None:
        response = response * (mask.astype(np.float32) > 0)

    max_response = response.max()
    if max_response <= 0:
        return np.empty((0, 2), dtype=np.float32)

    threshold = quality_level * max_response
    candidates = _non_maximum_suppression(response, threshold, min_distance)
    if candidates.shape[0] > max_corners:
        candidates = candidates[:max_corners]
    return candidates.astype(np.float32)
