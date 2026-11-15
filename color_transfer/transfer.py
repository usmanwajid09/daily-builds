"""Reinhard colour transfer from scratch (numpy only).

Re-colours a source image so its colour statistics match a target
image: for each channel, subtract the source mean, divide by the
source standard deviation, then scale by the target standard
deviation and add the target mean.  By default this is done in CIELAB
space (implemented here with the standard sRGB/D65 matrices), where
the channels are roughly decorrelated; plain RGB transfer is also
available.  Only numpy is used.
"""

import numpy as np

__all__ = ["color_transfer", "transfer_rgb", "rgb_to_lab", "lab_to_rgb"]

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
# D65 reference white (Xn, Yn, Zn).
_WHITE = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _validate_image(image, name):
    """Return the image as float64 in [0, 255], raising on bad input."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{name} must be an RGB image with shape (H, W, 3), got "
            f"shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(
            f"{name} pixel values must be in [0, 255], got "
            f"[{arr.min()}, {arr.max()}]")
    return arr


def _srgb_to_linear(channel):
    """sRGB gamma expansion, channel in [0, 1]."""
    return np.where(channel <= 0.04045,
                    channel / 12.92,
                    ((channel + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(channel):
    """sRGB gamma compression, channel in [0, 1]."""
    return np.where(channel <= 0.0031308,
                    12.92 * channel,
                    1.055 * np.power(np.clip(channel, 0.0, None),
                                     1.0 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    """Convert an RGB image (values in [0, 255]) to CIELAB."""
    arr = _validate_image(rgb, "rgb")
    linear = _srgb_to_linear(arr / 255.0)
    xyz = linear @ _RGB2XYZ.T
    xyz = xyz / _WHITE
    f = np.where(xyz > _DELTA ** 3,
                 np.cbrt(np.clip(xyz, 0.0, None)),
                 xyz / (3 * _DELTA ** 2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab):
    """Convert a CIELAB image back to RGB (values in [0, 255])."""
    arr = np.asarray(lab, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"lab must have shape (H, W, 3), got shape {arr.shape}")
    fy = (arr[..., 0] + 16.0) / 116.0
    fx = fy + arr[..., 1] / 500.0
    fz = fy - arr[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA,
                   f ** 3,
                   3 * _DELTA ** 2 * (f - 4.0 / 29.0))
    xyz = xyz * _WHITE
    linear = xyz @ _XYZ2RGB.T
    rgb = _linear_to_srgb(np.clip(linear, 0.0, None)) * 255.0
    return rgb


def color_transfer(source, target, colorspace="lab"):
    """Transfer the colour statistics of ``target`` onto ``source``.

    Parameters
    ----------
    source, target : RGB images with values in [0, 255]
        Shapes may differ; both must have 3 channels.
    colorspace : "lab" or "rgb"
        Channel space in which means/standard deviations are matched.

    Returns
    -------
    float64 RGB image, same shape as ``source``, values clipped to
    [0, 255].  Each channel's mean/std approximately equal the
    target's.
    """
    src = _validate_image(source, "source")
    tgt = _validate_image(target, "target")
    if colorspace not in ("lab", "rgb"):
        raise ValueError(
            f"colorspace must be 'lab' or 'rgb', got {colorspace!r}")
    if colorspace == "lab":
        src_c, tgt_c = rgb_to_lab(src), rgb_to_lab(tgt)
    else:
        src_c, tgt_c = src, tgt
    out_c = np.empty_like(src_c)
    for channel in range(3):
        src_mean = float(src_c[..., channel].mean())
        src_std = float(src_c[..., channel].std())
        tgt_mean = float(tgt_c[..., channel].mean())
        tgt_std = float(tgt_c[..., channel].std())
        if src_std < 1e-8:
            raise ValueError(
                f"source channel {channel} has (near-)zero variance; "
                "cannot match statistics")
        out_c[..., channel] = ((src_c[..., channel] - src_mean) / src_std
                               * tgt_std + tgt_mean)
    if colorspace == "lab":
        out = lab_to_rgb(out_c)
    else:
        out = out_c
    return np.clip(out, 0.0, 255.0)


def transfer_rgb(source, target):
    """Colour transfer done directly in RGB space (see color_transfer)."""
    return color_transfer(source, target, colorspace="rgb")
