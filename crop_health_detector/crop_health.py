"""Crop-health detection from synthetic multispectral patches.

Detects crop health from synthetic red + near-infrared (NIR) reflectance
bands using the standard vegetation indices NDVI and EVI (EVI2 when no
blue band is available), classifies each pixel/patch as healthy, stressed
or bare soil, and builds a field health map.

Only numpy + the standard library are used; all data is generated in code.
"""

import numpy as np

__all__ = [
    "LABELS",
    "LABEL_NAMES",
    "make_field",
    "ndvi",
    "evi",
    "classify_patch",
    "health_map",
    "zone_mean_ndvi",
    "classification_accuracy",
]

# Zone label codes (also used as the ground-truth label map).
HEALTHY = 0
STRESSED = 1
BARE = 2

LABELS = {"healthy": HEALTHY, "stressed": STRESSED, "bare": BARE}
LABEL_NAMES = ("healthy", "stressed", "bare")

# Default classification thresholds (on mean NDVI / EVI of a patch or pixel).
HEALTHY_NDVI = 0.55
STRESSED_NDVI = 0.20
HEALTHY_EVI = 0.40
STRESSED_EVI = 0.12


def _as_float_array(values, name):
    """Convert input to a float64 ndarray, rejecting empty / non-finite data."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array, got {type(values).__name__}") from exc
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return arr


def _check_reflectance(arr, name):
    """Reflectance bands must live in [0, 1]."""
    if arr.min() < 0.0:
        raise ValueError(f"{name} contains negative reflectance values (minimum {arr.min():.4f}); reflectance must be >= 0")
    if arr.max() > 1.0:
        raise ValueError(f"{name} contains reflectance values above 1 (maximum {arr.max():.4f}); reflectance must be <= 1")


def _check_same_shape(a, b, name_a, name_b):
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} have mismatched shapes: {a.shape} vs {b.shape}"
        )


def make_field(height, width, seed=0):
    """Generate a synthetic multispectral field.

    The field is split into a 2x2 grid of planted zones:

    ============  ======================================================
    Zone          Reflectance signature
    ============  ======================================================
    healthy       NIR 0.70-0.90, red 0.05-0.15, NDVI ~0.65-0.89
    stressed     NIR 0.45-0.55, red 0.20-0.30, NDVI ~0.20-0.47
    bare soil     NIR ~ red (0.25-0.35), NDVI ~0.00-0.15
    ============  ======================================================

    Layout (top-left, top-right / bottom-left, bottom-right):
    healthy, stressed / bare, healthy -- so two healthy zones bracket
    the stressed and bare zones. Small Gaussian noise is added and the
    bands are clipped back into their zone ranges so the planted NDVI
    bands hold.

    Parameters
    ----------
    height, width : int
        Field size in pixels (must be positive integers).
    seed : int
        Random seed; the same seed always gives the same field.

    Returns
    -------
    red, nir : ndarray, shape (height, width), float64 in [0, 1]
    labels : ndarray, shape (height, width), int
        Ground-truth zone map: 0 = healthy, 1 = stressed, 2 = bare.
    """
    if isinstance(height, bool) or not isinstance(height, (int, np.integer)):
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    rng = np.random.default_rng(seed)
    red = np.empty((height, width), dtype=float)
    nir = np.empty((height, width), dtype=float)
    labels = np.empty((height, width), dtype=int)

    h_mid, w_mid = height // 2, width // 2
    # (row slice, col slice, label, nir range, red range)
    zones = [
        (slice(0, h_mid), slice(0, w_mid), HEALTHY, (0.70, 0.90), (0.05, 0.15)),
        (slice(0, h_mid), slice(w_mid, width), STRESSED, (0.45, 0.55), (0.20, 0.30)),
        (slice(h_mid, height), slice(0, w_mid), BARE, (0.25, 0.35), (0.22, 0.32)),
        (slice(h_mid, height), slice(w_mid, width), HEALTHY, (0.70, 0.90), (0.05, 0.15)),
    ]
    for rows, cols, code, nir_range, red_range in zones:
        shape = (rows.stop - rows.start, cols.stop - cols.start)
        if code == BARE:
            # Bare soil: NIR tracks red closely so NDVI stays near zero.
            base = rng.uniform(0.25, 0.35, size=shape)
            nir_block = np.clip(base + rng.normal(0.0, 0.01, size=shape), 0.20, 0.40)
            red_block = np.clip(base + rng.normal(0.0, 0.01, size=shape), 0.20, 0.40)
        else:
            nir_block = rng.uniform(*nir_range, size=shape)
            red_block = rng.uniform(*red_range, size=shape)
            nir_block = np.clip(nir_block + rng.normal(0.0, 0.01, size=shape), *nir_range)
            red_block = np.clip(red_block + rng.normal(0.0, 0.01, size=shape), *red_range)
        nir[rows, cols] = nir_block
        red[rows, cols] = red_block
        labels[rows, cols] = code

    return red, nir, labels


def ndvi(red, nir):
    """Normalised Difference Vegetation Index: (NIR - red) / (NIR + red).

    Returns 0 wherever the denominator is 0 (safe division). Output is
    in [-1, 1]; dense healthy vegetation scores ~0.7-0.9, bare soil ~0.
    """
    red = _as_float_array(red, "red")
    nir = _as_float_array(nir, "nir")
    _check_same_shape(red, nir, "red", "nir")
    _check_reflectance(red, "red")
    _check_reflectance(nir, "nir")
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0.0, 0.0, (nir - red) / denom)
    return result


def evi(nir, red, blue=None):
    """Enhanced Vegetation Index.

    With a blue band, the standard 3-band EVI is used::

        2.5 * (NIR - red) / (NIR + 6*red - 7.5*blue + 1)

    Without one, EVI2 (the 2-band form) is used::

        2.5 * (NIR - red) / (NIR + 2.4*red + 1)

    Returns 0 wherever the denominator is 0 (safe division).
    """
    nir = _as_float_array(nir, "nir")
    red = _as_float_array(red, "red")
    _check_same_shape(nir, red, "nir", "red")
    _check_reflectance(nir, "nir")
    _check_reflectance(red, "red")
    if blue is None:
        denom = nir + 2.4 * red + 1.0
    else:
        blue = _as_float_array(blue, "blue")
        _check_same_shape(nir, blue, "nir", "blue")
        _check_reflectance(blue, "blue")
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0.0, 0.0, 2.5 * (nir - red) / denom)
    return result


def classify_patch(ndvi_value, evi_value, healthy_ndvi=HEALTHY_NDVI,
                   stressed_ndvi=STRESSED_NDVI, healthy_evi=HEALTHY_EVI,
                   stressed_evi=STRESSED_EVI):
    """Classify one patch (or pixel) as 'healthy', 'stressed' or 'bare'.

    ``ndvi_value`` and ``evi_value`` may be scalars or arrays; arrays are
    summarised by their mean. A patch is healthy when both its mean NDVI
    and mean EVI clear the healthy thresholds, stressed when both clear
    the stressed thresholds, and bare otherwise.
    """
    if not (0.0 <= stressed_ndvi < healthy_ndvi <= 1.0):
        raise ValueError(
            f"thresholds must satisfy 0 <= stressed_ndvi < healthy_ndvi <= 1, "
            f"got stressed_ndvi={stressed_ndvi}, healthy_ndvi={healthy_ndvi}"
        )
    if not (0.0 <= stressed_evi < healthy_evi):
        raise ValueError(
            f"thresholds must satisfy 0 <= stressed_evi < healthy_evi, "
            f"got stressed_evi={stressed_evi}, healthy_evi={healthy_evi}"
        )
    ndvi_arr = _as_float_array(ndvi_value, "ndvi")
    evi_arr = _as_float_array(evi_value, "evi")
    if ndvi_arr.min() < -1.0 or ndvi_arr.max() > 1.0:
        raise ValueError(
            f"ndvi values must lie in [-1, 1], got range [{ndvi_arr.min():.3f}, {ndvi_arr.max():.3f}]"
        )
    mean_ndvi = float(ndvi_arr.mean())
    mean_evi = float(evi_arr.mean())
    if mean_ndvi >= healthy_ndvi and mean_evi >= healthy_evi:
        return "healthy"
    if mean_ndvi >= stressed_ndvi and mean_evi >= stressed_evi:
        return "stressed"
    return "bare"


def health_map(ndvi_values, evi_values=None, healthy_ndvi=HEALTHY_NDVI,
               stressed_ndvi=STRESSED_NDVI, healthy_evi=HEALTHY_EVI,
               stressed_evi=STRESSED_EVI):
    """Classify every pixel of an NDVI map and summarise the field.

    Parameters
    ----------
    ndvi_values : array, values in [-1, 1]
    evi_values : array or None
        If given, a pixel must also clear the EVI threshold for its class.

    Returns
    -------
    label_map : ndarray of int, same shape as ``ndvi_values``
        0 = healthy, 1 = stressed, 2 = bare.
    summary : dict
        Per-class pixel ``counts`` and ``percentages`` (0-100), plus
        ``total_pixels``.
    """
    if not (0.0 <= stressed_ndvi < healthy_ndvi <= 1.0):
        raise ValueError(
            f"thresholds must satisfy 0 <= stressed_ndvi < healthy_ndvi <= 1, "
            f"got stressed_ndvi={stressed_ndvi}, healthy_ndvi={healthy_ndvi}"
        )
    ndvi_arr = _as_float_array(ndvi_values, "ndvi")
    if ndvi_arr.min() < -1.0 or ndvi_arr.max() > 1.0:
        raise ValueError(
            f"ndvi values must lie in [-1, 1], got range [{ndvi_arr.min():.3f}, {ndvi_arr.max():.3f}]"
        )
    if evi_values is None:
        evi_arr = np.full_like(ndvi_arr, np.inf)
    else:
        evi_arr = _as_float_array(evi_values, "evi")
        _check_same_shape(ndvi_arr, evi_arr, "ndvi", "evi")

    label_map = np.full(ndvi_arr.shape, BARE, dtype=int)
    stressed_mask = (ndvi_arr >= stressed_ndvi) & (evi_arr >= stressed_evi)
    label_map[stressed_mask] = STRESSED
    healthy_mask = (ndvi_arr >= healthy_ndvi) & (evi_arr >= healthy_evi)
    label_map[healthy_mask] = HEALTHY

    total = int(ndvi_arr.size)
    counts = {name: int(np.count_nonzero(label_map == code))
              for name, code in LABELS.items()}
    percentages = {name: 100.0 * count / total for name, count in counts.items()}
    summary = {"counts": counts, "percentages": percentages, "total_pixels": total}
    return label_map, summary


def zone_mean_ndvi(ndvi_values, labels):
    """Mean NDVI per ground-truth zone.

    Returns a dict mapping zone name ('healthy' / 'stressed' / 'bare')
    to its mean NDVI (only zones present in ``labels`` are included).
    """
    ndvi_arr = _as_float_array(ndvi_values, "ndvi")
    labels_arr = _as_float_array(labels, "labels")
    _check_same_shape(ndvi_arr, labels_arr, "ndvi", "labels")
    if not np.all(np.isin(labels_arr, list(LABELS.values()))):
        raise ValueError("labels must contain only 0 (healthy), 1 (stressed) or 2 (bare)")
    means = {}
    for name, code in LABELS.items():
        mask = labels_arr == code
        if np.any(mask):
            means[name] = float(ndvi_arr[mask].mean())
    return means


def classification_accuracy(predicted, ground_truth):
    """Fraction of pixels where the predicted label matches ground truth.

    Returns a float in [0, 1]. Both inputs must be integer label arrays
    of the same shape containing only 0, 1 or 2.
    """
    pred = _as_float_array(predicted, "predicted")
    truth = _as_float_array(ground_truth, "ground_truth")
    _check_same_shape(pred, truth, "predicted", "ground_truth")
    for arr, name in ((pred, "predicted"), (truth, "ground_truth")):
        if not np.all(np.isin(arr, list(LABELS.values()))):
            raise ValueError(f"{name} must contain only 0 (healthy), 1 (stressed) or 2 (bare)")
    return float(np.mean(pred == truth))
