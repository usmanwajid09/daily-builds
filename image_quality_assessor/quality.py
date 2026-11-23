"""No-reference image quality assessment (numpy only).

Measures four things about an image and combines them into a single
0-100 quality score with a verdict:

* **blur** -- variance of the Laplacian response (a 3x3 Laplacian kernel
  convolved with numpy, no OpenCV). Sharp images have strong second
  derivatives at edges, so a high variance means sharp; a low variance
  means blurred. The blur sub-score is computed on a lightly box-blurred
  copy so sensor noise cannot masquerade as sharpness.
* **noise** -- an estimate of the Gaussian noise level (sigma, in grey
  levels). It uses the high-pass residual (image minus a 3x3 box blur)
  inside the centre of 8x8 blocks, taking the median across blocks, so
  real edges do not inflate the estimate the way a plain residual
  standard deviation would.
* **contrast** -- standard deviation of the pixel values.
* **brightness** -- mean pixel value; mid-range brightness scores best.

The combined score weights blur 35%, noise 30%, contrast 20% and
brightness 15%. A sharp, clean, well-contrasted, mid-brightness image
scores near 100; heavy blur, heavy noise, very low contrast, or
near-black / near-white brightness each drag the score down.

Inputs are numpy arrays, 2D (grayscale) or 3D with 1, 3 or 4 channels
(RGB/RGBA; alpha is ignored). Pixel values may be in 0-255 or in 0-1
(values <= 1.0 are treated as the 0-1 range and scaled to 0-255).
"""

import numpy as np

__all__ = [
    "assess",
    "assess_batch",
    "box_blur",
    "brightness_score",
    "contrast_score",
    "estimate_noise",
    "laplacian",
    "laplacian_variance",
    "make_checkerboard",
    "make_gradient",
    "add_gaussian_noise",
    "make_low_contrast",
    "noise_score",
    "blur_score",
    "verdict_for",
]

_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
)

# Laplacian variance of the lightly blurred image at which the blur
# sub-score reaches 50 (sharp synthetic images are in the thousands,
# heavily blurred ones are in the tens).
_BLUR_HALF_POINT = 400.0
# Noise sigma (grey levels) at which the noise sub-score falls to ~37.
_NOISE_SCALE = 20.0
# Contrast (pixel std) at which the contrast sub-score reaches 50.
_CONTRAST_HALF_POINT = 25.0

_WEIGHTS = {"blur": 0.35, "noise": 0.30, "contrast": 0.20, "brightness": 0.15}


def _validate(image):
    """Validate an image and return it as a float64 array in 0-255."""
    if isinstance(image, (list, tuple)):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.dtype == bool or image.dtype == object:
        raise ValueError(
            f"image must have a numeric dtype, got {image.dtype}"
        )
    if np.iscomplexobj(image):
        raise ValueError("image must be real-valued, not complex")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (grayscale) or 3D (H, W, channels), "
            f"got {image.ndim}D with shape {image.shape}"
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"3D image must have 1, 3 or 4 channels, "
            f"got {image.shape[2]} (shape {image.shape})"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError(
            f"image must be at least 3x3 pixels, got shape {image.shape}"
        )
    arr = image.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values (no NaN/inf)")
    lo, hi = float(arr.min()), float(arr.max())
    if lo < 0.0 or hi > 255.0:
        raise ValueError(
            f"pixel values must be in 0-255 (or 0-1), got range [{lo}, {hi}]"
        )
    if hi <= 1.0:
        # 0-1 range: scale up. (A genuine 0-255 image whose brightest
        # pixel is 1 is indistinguishable from this; document the rule.)
        arr = arr * 255.0
    return arr


def _to_grayscale(arr):
    """Convert a validated 0-255 array to a 2D float64 grayscale array."""
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    # RGB or RGBA: plain channel mean (alpha, if present, is ignored).
    return arr[:, :, :3].mean(axis=2)


def _pad_edge(a):
    return np.pad(a, 1, mode="edge")


def box_blur(gray, iterations=1):
    """Mean (box) blur of a 2D grayscale array, applied `iterations` times.

    Each pass is a 3x3 mean filter implemented with shifted sums.
    Accepts any numeric 2D array; values are not range-checked here
    (use :func:`assess` for the validated path).
    """
    a = np.asarray(gray, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"box_blur expects a 2D array, got shape {a.shape}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    for _ in range(iterations):
        p = _pad_edge(a)
        a = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return a


def laplacian(gray):
    """Laplacian response of a 2D grayscale array (3x3 kernel).

    Edge-replicated padding keeps the output the same shape as the input.
    """
    a = np.asarray(gray, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"laplacian expects a 2D array, got shape {a.shape}")
    p = _pad_edge(a)
    return (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        - 4.0 * p[1:-1, 1:-1]
    )


def laplacian_variance(image):
    """Variance of the Laplacian response (classic blur measure).

    High values mean sharp (strong edges); low values mean blurred or
    textureless. Accepts grayscale or RGB, in 0-255 or 0-1.
    """
    gray = _to_grayscale(_validate(image))
    return float(np.var(laplacian(gray)))


def estimate_noise(image):
    """Estimate the Gaussian noise level (sigma, in grey levels).

    Uses the high-pass residual (image minus a 3x3 box blur) inside the
    centre 4x4 of each 8x8 block, taking the median block standard
    deviation. Block centres avoid edges, so real image structure does
    not inflate the estimate; the median across blocks makes it robust
    to the few blocks that still contain an edge. The residual of white
    noise with sigma s has std ~0.94*s, hence the /0.94 correction.
    Returns 0.0 for a noiseless image.
    """
    gray = _to_grayscale(_validate(image))
    residual = gray - box_blur(gray, iterations=1)
    h, w = gray.shape
    block = 8
    if h < block or w < block:
        return float(max(np.std(residual) / 0.94, 0.0))
    hb, wb = (h // block) * block, (w // block) * block
    trimmed = residual[:hb, :wb].reshape(hb // block, block, wb // block, block)
    centres = trimmed[:, 2:6, :, 2:6]  # 4x4 centre of each block
    block_stds = np.std(centres, axis=(1, 3))
    return float(max(np.median(block_stds) / 0.94, 0.0))


def blur_score(image):
    """0-100 sharpness sub-score from the (noise-robust) Laplacian variance."""
    gray = _to_grayscale(_validate(image))
    lv = float(np.var(laplacian(box_blur(gray, iterations=1))))
    return float(100.0 * lv / (lv + _BLUR_HALF_POINT))


def noise_score(image):
    """0-100 sub-score: 100 for noiseless, falling as noise grows."""
    return float(100.0 * np.exp(-estimate_noise(image) / _NOISE_SCALE))


def contrast_score(image):
    """0-100 sub-score from the pixel standard deviation."""
    gray = _to_grayscale(_validate(image))
    contrast = float(np.std(gray))
    return float(100.0 * contrast / (contrast + _CONTRAST_HALF_POINT))


def brightness_score(image):
    """0-100 sub-score: 100 for mid-range brightness, 0 near black/white."""
    gray = _to_grayscale(_validate(image))
    dist = abs(float(np.mean(gray)) - 127.5)
    if dist <= 30.0:
        return 100.0
    return float(max(0.0, 100.0 * (1.0 - (dist - 30.0) / 97.5)))


def verdict_for(score):
    """Verdict for a 0-100 score: excellent / good / fair / poor."""
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 40:
        return "fair"
    return "poor"


def assess(image):
    """Assess one image and return a dict of metrics and the verdict.

    Keys: ``score`` (0-100 float), ``verdict``, ``blur`` (Laplacian
    variance of the raw image), ``noise`` (estimated sigma), ``contrast``
    (pixel std), ``brightness`` (mean pixel value), plus the 0-100
    sub-scores ``blur_score``, ``noise_score``, ``contrast_score`` and
    ``brightness_score``.
    """
    gray = _to_grayscale(_validate(image))
    blur = float(np.var(laplacian(gray)))
    noise = estimate_noise(image)
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    sub = {
        "blur_score": blur_score(image),
        "noise_score": float(100.0 * np.exp(-noise / _NOISE_SCALE)),
        "contrast_score": float(
            100.0 * contrast / (contrast + _CONTRAST_HALF_POINT)
        ),
        "brightness_score": brightness_score(image),
    }
    score = float(
        _WEIGHTS["blur"] * sub["blur_score"]
        + _WEIGHTS["noise"] * sub["noise_score"]
        + _WEIGHTS["contrast"] * sub["contrast_score"]
        + _WEIGHTS["brightness"] * sub["brightness_score"]
    )
    score = float(min(100.0, max(0.0, score)))
    return {
        "score": score,
        "verdict": verdict_for(score),
        "blur": blur,
        "noise": noise,
        "contrast": contrast,
        "brightness": brightness,
        **sub,
    }


def assess_batch(images):
    """Assess an iterable of images; returns a list of :func:`assess` dicts."""
    return [assess(img) for img in images]


# ---------------------------------------------------------------------------
# Synthetic image helpers (used by the demo and the tests)
# ---------------------------------------------------------------------------

def make_checkerboard(size=64, square=8, seed=None):
    """Sharp test image: checkerboard blended with a horizontal gradient."""
    del seed  # deterministic; kept for a uniform helper signature
    y, x = np.mgrid[0:size, 0:size]
    checks = ((x // square) + (y // square)) % 2
    gradient = np.linspace(0, 255, size)[None, :] * np.ones((size, 1))
    return checks * 255.0 * 0.7 + gradient * 0.3


def make_gradient(size=64):
    """Smooth horizontal gradient from black (left) to white (right)."""
    return np.tile(np.linspace(0, 255, size), (size, 1))


def add_gaussian_noise(image, sigma=25.0, seed=0):
    """Return a copy of `image` with additive Gaussian noise (clipped)."""
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image, dtype=np.float64) + rng.normal(
        0.0, sigma, np.asarray(image).shape
    )
    return np.clip(noisy, 0, 255)


def make_low_contrast(image, factor=0.12, level=110.0):
    """Compress `image` towards `level` so its contrast drops."""
    return level + (np.asarray(image, dtype=np.float64) - 127.5) * factor
