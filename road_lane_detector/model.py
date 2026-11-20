"""Road lane detection on synthetic images (numpy only, no OpenCV).

Pipeline: Sobel gradient magnitude -> edge threshold -> Hough-style
line voting (accumulator over rho/theta, near-vertical angles only),
separately for the left and right halves of the image -> the strongest
line in each half is returned as endpoints at the bottom of the image
and at the horizon (vanishing-point height). A synthetic road image
generator is included for demos and tests.
"""

import numpy as np

# The horizon / vanishing-point height, as a fraction of image height.
HORIZON_FRAC = 0.40


def _convolve3x3(image, kernel):
    """3x3 convolution with replicate padding (numpy shifts only)."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            w = kernel[dy, dx]
            if w != 0:
                out += w * padded[dy:dy + image.shape[0],
                                 dx:dx + image.shape[1]]
    return out


def sobel_edges(image):
    """Gradient magnitude of a grayscale ``image`` via Sobel filters."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError(
            f"image must be at least 3x3, got shape {image.shape}")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = _convolve3x3(image, kx)
    gy = _convolve3x3(image, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


def hough_lines(edge_mask, theta_degrees=None, min_votes=20):
    """Hough accumulator peak for near-vertical lines in ``edge_mask``.

    Returns ``(rho, theta_rad, votes)`` of the strongest line, or
    ``None`` when no accumulator cell reaches ``min_votes``.
    """
    edge_mask = np.asarray(edge_mask)
    if edge_mask.ndim != 2:
        raise ValueError(
            f"edge_mask must be 2D, got shape {edge_mask.shape}")
    if int(min_votes) < 1:
        raise ValueError(f"min_votes must be >= 1, got {min_votes}")
    if theta_degrees is None:
        theta_degrees = np.arange(-40, 41, 1)
    thetas = np.deg2rad(np.asarray(theta_degrees, dtype=np.float64))
    height, width = edge_mask.shape
    diag = int(np.ceil(np.hypot(height, width)))
    rhos = np.arange(-diag, diag + 1)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64)
    ys, xs = np.nonzero(edge_mask)
    if len(xs) == 0:
        return None
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        rho_idx = np.round(rho_vals).astype(int) + diag
        accumulator[rho_idx, np.arange(len(thetas))] += 1
    peak = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    votes = int(accumulator[peak])
    if votes < min_votes:
        return None
    return float(rhos[peak[0]]), float(thetas[peak[1]]), votes


def line_endpoints(rho, theta, image_shape, y_top=None):
    """Endpoints of the line ``(rho, theta)`` at the bottom and horizon.

    Returns ``((x_bottom, y_bottom), (x_top, y_top))`` in image
    coordinates, or ``None`` if the line is (near-)horizontal.
    """
    height, width = image_shape
    if abs(np.cos(theta)) < 1e-6:
        return None
    y_bottom = height - 1
    if y_top is None:
        y_top = int(HORIZON_FRAC * height)
    x_bottom = (rho - y_bottom * np.sin(theta)) / np.cos(theta)
    x_top = (rho - y_top * np.sin(theta)) / np.cos(theta)
    if not (-width <= x_bottom <= 2 * width
            and -width <= x_top <= 2 * width):
        return None
    return ((float(x_bottom), float(y_bottom)),
            (float(x_top), float(y_top)))


def make_road_image(width=320, height=240, noise=0.0, seed=0,
                    line_width=4):
    """Synthetic road: two lane lines converging to a vanishing point.

    The road is dark grey on a lighter background, with two bright lane
    lines from the bottom corners to the vanishing point at
    ``(width/2, 0.40*height)`` and a thin dashed centre line. Returns a
    float64 grayscale image in [0, 255]. Also returns the true lane
    endpoints ``{"left": ..., "right": ...}`` in the same format as
    :func:`detect_lanes`.
    """
    if width < 40 or height < 40:
        raise ValueError(
            f"road image must be at least 40x40, got {(height, width)}")
    if line_width < 1:
        raise ValueError(f"line_width must be >= 1, got {line_width}")
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 90.0)
    # Road surface: trapezoid from the bottom edge to the horizon.
    horizon_y = int(HORIZON_FRAC * height)
    for y in range(horizon_y, height):
        t = (y - horizon_y) / max(1, height - 1 - horizon_y)
        half = width * (0.08 + 0.42 * t)
        left = int(width / 2 - half)
        right = int(width / 2 + half)
        image[y, left:right] = 45.0
    vanishing = (width / 2.0, float(horizon_y))
    left_bottom = (width * 0.12, float(height - 1))
    right_bottom = (width * 0.88, float(height - 1))

    def _draw_line(p0, p1, value, thickness):
        steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) * 2 + 1
        for s in range(steps + 1):
            t = s / steps
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            half = thickness // 2
            r0 = max(0, int(round(y)) - half)
            r1 = min(height, int(round(y)) + half + 1)
            c0 = max(0, int(round(x)) - half)
            c1 = min(width, int(round(x)) + half + 1)
            image[r0:r1, c0:c1] = value

    _draw_line(left_bottom, vanishing, 220.0, line_width)
    _draw_line(right_bottom, vanishing, 220.0, line_width)
    # Dashed centre line (thin, so it never out-votes a lane line).
    centre_bottom = (width / 2.0, float(height - 1))
    for k in range(6):
        t0, t1 = k / 6.0, (k + 0.5) / 6.0
        p0 = (centre_bottom[0] + (vanishing[0] - centre_bottom[0]) * t0,
              centre_bottom[1] + (vanishing[1] - centre_bottom[1]) * t0)
        p1 = (centre_bottom[0] + (vanishing[0] - centre_bottom[0]) * t1,
              centre_bottom[1] + (vanishing[1] - centre_bottom[1]) * t1)
        _draw_line(p0, p1, 180.0, 2)
    if noise > 0:
        image = np.clip(image + rng.normal(0.0, noise, image.shape),
                        0.0, 255.0)
    truth = {
        "left": (left_bottom, vanishing),
        "right": (right_bottom, vanishing),
    }
    return image, truth


def detect_lanes(image, edge_fraction=0.25, min_votes=20):
    """Detect the left and right lane lines of a road ``image``.

    Returns ``{"left": endpoints or None, "right": endpoints or None,
    "votes": {"left": int, "right": int}}`` where endpoints are
    ``((x_bottom, y_bottom), (x_top, y_top))``.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if image.shape[0] < 40 or image.shape[1] < 40:
        raise ValueError(
            f"image must be at least 40x40, got shape {image.shape}")
    if not 0.0 < edge_fraction < 1.0:
        raise ValueError(
            f"edge_fraction must be in (0, 1), got {edge_fraction}")
    magnitude = sobel_edges(image)
    threshold = magnitude.max() * edge_fraction
    edges = magnitude > threshold
    height, width = image.shape
    result = {"left": None, "right": None, "votes": {"left": 0, "right": 0}}
    for side, half in (("left", edges[:, : width // 2]),
                       ("right", edges[:, width // 2:])):
        found = hough_lines(half, min_votes=min_votes)
        if found is None:
            continue
        rho, theta, votes = found
        if side == "right":
            rho = rho + (width // 2) * np.cos(theta)
        result[side] = line_endpoints(rho, theta, image.shape)
        result["votes"][side] = votes
    return result
