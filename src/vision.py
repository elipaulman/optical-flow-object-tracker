import cv2
import numpy as np

'''
Provides image processing functions for building Gaussian pyramids,
computing image gradients, and detecting Shi-Tomasi corners.
Fully implemented using numpy, no built-in functions used.
'''

def preprocess_frame(frame, blur_ksize=5):
    """Convert BGR frame to blurred grayscale float32."""
    # convert color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply blur to reduce noise (if specified)
    if blur_ksize and blur_ksize > 1:
        gray = gaussian_blur(gray.astype(np.float32), ksize=blur_ksize)
    return gray.astype(np.float32)

def _convolve_separable(image, kx, ky):
    """Separable 1D conv: horizontal kx then vertical ky. image float32."""
    h, w = image.shape
    kx = np.asarray(kx, dtype=np.float32)
    ky = np.asarray(ky, dtype=np.float32)
    rx = len(kx) // 2  # radius for horizontal kernel
    ry = len(ky) // 2  # radius for vertical kernel

    # horizontal pass (pad left/right only)
    padded_h = np.pad(image, ((0, 0), (rx, rx)), mode="reflect")
    tmp = np.zeros_like(image, dtype=np.float32)
    for i, kv in enumerate(kx):
        tmp += kv * padded_h[:, i : i + w]  # weighted sum

    # vertical pass (pad top/bottom only)
    padded_v = np.pad(tmp, ((ry, ry), (0, 0)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for j, kv in enumerate(ky):
        out += kv * padded_v[j : j + h, :]  # weighted sum
    return out


def sobel_gradients(image):
    """Compute Sobel gradients (dx, dy) without cv2.Sobel."""
    # Separable Sobel kernels (smooth in one dir, diff in other)
    smooth = [1.0, 2.0, 1.0]  # binomial smoothing
    diff = [1.0, 0.0, -1.0]  # central difference
    gx = _convolve_separable(image, diff, smooth)  # x gradient
    gy = _convolve_separable(image, smooth, diff)  # y gradient
    return gx, gy


def gaussian_blur_5x5(image):
    """5x5 Gaussian blur via separable filter (1,4,6,4,1)/16."""
    # binomial approx to gaussian
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    return _convolve_separable(image, kernel, kernel)


def _gaussian_kernel1d(ksize, sigma=None):
    """Generate 1D Gaussian kernel."""
    if ksize % 2 == 0:
        ksize += 1  # force odd kernel size
    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # OpenCV-like default sigma
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float32)  # symmetric range
    kernel = np.exp(-0.5 * (x / sigma) ** 2)  # gaussian formula
    kernel /= kernel.sum()  # normalize so weights sum to 1
    return kernel


def gaussian_blur(image, ksize=5, sigma=None):
    """Gaussian blur via custom separable convolution."""
    if ksize <= 1:
        return image  # no blur needed
    kx = _gaussian_kernel1d(ksize, sigma)
    # apply same kernel horizontally then vertically
    return _convolve_separable(image.astype(np.float32), kx, kx)


def pyr_down_custom(image):
    """Rudimentary pyrDown: blur then decimate by 2."""
    blurred = gaussian_blur_5x5(image)  # smooth before downsampling
    return blurred[::2, ::2]  # take every other pixel (downsample by 2)


def gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid with the requested number of levels."""
    pyramid = [image]  # level 0 = original
    for _ in range(1, levels):
        image = pyr_down_custom(image)  # each level is half the size
        pyramid.append(image)
    return pyramid


def compute_gradients(image):
    """Compute spatial image gradients for a single grayscale level."""
    ix, iy = sobel_gradients(image)  # get x and y derivatives
    return ix, iy


def build_pyramid_with_gradients(image, levels):
    """Return Gaussian pyramid plus per-level gradients."""
    pyr = gaussian_pyramid(image, levels)  # multi-scale imgs
    grads = [compute_gradients(level) for level in pyr]  # compute grads at each scale
    return pyr, grads


def _box_filter(image, ksize):
    """Simple box filter using separable uniform kernels."""
    # handle square or rectangular kernel sizes
    if isinstance(ksize, tuple):
        kx, ky = ksize
    else:
        kx = ky = ksize
    kx = int(kx)
    ky = int(ky)
    # uniform kernels (all weights equal)
    kernel_x = np.ones(kx, dtype=np.float32) / float(kx)
    kernel_y = np.ones(ky, dtype=np.float32) / float(ky)
    return _convolve_separable(image, kernel_x, kernel_y)


def _non_maximum_suppression(response, threshold, min_distance):
    """Return coordinates of local maxima above threshold with spacing enforced."""
    # find local maxima using morphological dilation
    dilated = cv2.dilate(response, None)
    maxima_mask = (response == dilated) & (response >= threshold)  # peaks above thresh
    coords = np.column_stack(np.nonzero(maxima_mask))  # get (y, x) coords
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # sort by response strength (strongest first)
    values = response[maxima_mask]
    order = np.argsort(values)[::-1]
    coords = coords[order]

    # greedily select points that are far enough apart
    selected = []
    min_dist_sq = min_distance * min_distance
    for y, x in coords:
        # check distance to all already selected points
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
    """Compute Shi-Tomasi response and pick strong corners.

    Shi-Tomasi finds points that are easy to track by looking for regions
    where the image changes a lot in multiple directions (corners).
    These are good features for optical flow tracking.
    """
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    # compute image gradients (how much brightness changes in x & y)
    ix, iy = sobel_gradients(gray)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # build structure tensor M = [sxx, sxy; sxy, syy] for each pixel
    # this matrix captures how image intensity varies locally
    sxx = _box_filter(ix2, (block_size, block_size))
    syy = _box_filter(iy2, (block_size, block_size))
    sxy = _box_filter(ixy, (block_size, block_size))

    # compute smallest eigenvalue (lambda_min) of the structure tensor
    # high lambda_min = corner (changes in all directions)
    # low lambda_min = edge or flat region (not good for tracking)
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    inside = np.maximum(trace * trace - 4.0 * det, 0.0)
    lambda_min = 0.5 * (trace - np.sqrt(inside))

    response = lambda_min
    if mask is not None:
        response = response * (mask.astype(np.float32) > 0)  # only search in ROI

    max_response = response.max()
    if max_response <= 0:
        return np.empty((0, 2), dtype=np.float32)

    # keep only corners above quality threshold
    threshold = quality_level * max_response
    candidates = _non_maximum_suppression(response, threshold, min_distance)
    if candidates.shape[0] > max_corners:
        candidates = candidates[:max_corners]
    return candidates.astype(np.float32)
