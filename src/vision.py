import cv2
import numpy as np


def preprocess_frame(frame, blur_ksize=5):
    """Convert BGR frame to blurred grayscale float32."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = gaussian_blur(gray.astype(np.float32), ksize=blur_ksize)
    return gray.astype(np.float32)

def _convolve_separable(image, kx, ky):
    """Separable 1D conv: horizontal kx then vertical ky. image float32."""
    h, w = image.shape
    kx = np.asarray(kx, dtype=np.float32)
    ky = np.asarray(ky, dtype=np.float32)
    rx = len(kx) // 2
    ry = len(ky) // 2

    # horizontal pass (pad left/right only)
    padded_h = np.pad(image, ((0, 0), (rx, rx)), mode="reflect")
    tmp = np.zeros_like(image, dtype=np.float32)
    for i, kv in enumerate(kx):
        tmp += kv * padded_h[:, i : i + w]

    # vertical pass (pad top/bottom only)
    padded_v = np.pad(tmp, ((ry, ry), (0, 0)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for j, kv in enumerate(ky):
        out += kv * padded_v[j : j + h, :]
    return out


def sobel_gradients(image):
    """Compute Sobel gradients (dx, dy) without cv2.Sobel."""
    # Separable Sobel kernels
    smooth = [1.0, 2.0, 1.0]
    diff = [1.0, 0.0, -1.0]
    gx = _convolve_separable(image, diff, smooth)
    gy = _convolve_separable(image, smooth, diff)
    return gx, gy


def gaussian_blur_5x5(image):
    """5x5 Gaussian blur via separable filter (1,4,6,4,1)/16."""
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    return _convolve_separable(image, kernel, kernel)


def _gaussian_kernel1d(ksize, sigma=None):
    """Generate 1D Gaussian kernel."""
    if ksize % 2 == 0:
        ksize += 1
    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # OpenCV-like default
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(image, ksize=5, sigma=None):
    """Gaussian blur via custom separable convolution."""
    if ksize <= 1:
        return image
    kx = _gaussian_kernel1d(ksize, sigma)
    return _convolve_separable(image.astype(np.float32), kx, kx)


def pyr_down_custom(image):
    """Rudimentary pyrDown: blur then decimate by 2."""
    blurred = gaussian_blur_5x5(image)
    return blurred[::2, ::2]


def gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid with the requested number of levels."""
    pyramid = [image]
    for _ in range(1, levels):
        image = pyr_down_custom(image)
        pyramid.append(image)
    return pyramid


def compute_gradients(image):
    """Compute spatial image gradients for a single grayscale level."""
    ix, iy = sobel_gradients(image)
    return ix, iy


def build_pyramid_with_gradients(image, levels):
    """Return Gaussian pyramid plus per-level gradients."""
    pyr = gaussian_pyramid(image, levels)
    grads = [compute_gradients(level) for level in pyr]
    return pyr, grads


def _box_filter(image, ksize):
    """Simple box filter using separable uniform kernels."""
    if isinstance(ksize, tuple):
        kx, ky = ksize
    else:
        kx = ky = ksize
    kx = int(kx)
    ky = int(ky)
    kernel_x = np.ones(kx, dtype=np.float32) / float(kx)
    kernel_y = np.ones(ky, dtype=np.float32) / float(ky)
    return _convolve_separable(image, kernel_x, kernel_y)


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

    ix, iy = sobel_gradients(gray)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    sxx = _box_filter(ix2, (block_size, block_size))
    syy = _box_filter(iy2, (block_size, block_size))
    sxy = _box_filter(ixy, (block_size, block_size))

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
