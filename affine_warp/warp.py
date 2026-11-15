"""Affine image warping from scratch (numpy only).

Applies translation, rotation, scaling and shear to grayscale or RGB
images using inverse mapping with bilinear interpolation: for each
output pixel, the inverse affine matrix is used to find where it came
from in the input, and the four surrounding input pixels are blended.
Pixels that map outside the input are filled with ``fill_value``.
"""

import numpy as np

__all__ = ["build_affine_matrix", "rotation_matrix", "warp_affine",
           "warp_translate", "warp_rotate", "warp_scale"]


def build_affine_matrix(translation=(0.0, 0.0), rotation_deg=0.0,
                        scale=(1.0, 1.0), shear=(0.0, 0.0)):
    """Build a 3x3 affine matrix as T @ R @ S @ Shear.

    ``translation`` is (tx, ty) in pixels, ``rotation_deg`` is
    counter-clockwise degrees about the origin, ``scale`` is (sx, sy)
    and ``shear`` is (shx, shy).  Scale factors must be non-zero.
    """
    tx, ty = float(translation[0]), float(translation[1])
    sx, sy = float(scale[0]), float(scale[1])
    if sx == 0 or sy == 0:
        raise ValueError("scale factors must be non-zero")
    shx, shy = float(shear[0]), float(shear[1])
    theta = np.deg2rad(float(rotation_deg))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    shear_m = np.array([[1.0, shx, 0.0],
                        [shy, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
    scale_m = np.diag([sx, sy, 1.0])
    rot_m = np.array([[cos_t, -sin_t, 0.0],
                      [sin_t, cos_t, 0.0],
                      [0.0, 0.0, 1.0]])
    trans_m = np.array([[1.0, 0.0, tx],
                        [0.0, 1.0, ty],
                        [0.0, 0.0, 1.0]])
    return trans_m @ rot_m @ scale_m @ shear_m


def rotation_matrix(angle_deg, center):
    """Return a 3x3 matrix rotating by ``angle_deg`` about ``center``.

    ``center`` is (cx, cy) in (column, row) coordinates.
    """
    cx, cy = float(center[0]), float(center[1])
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    return np.array([
        [cos_t, -sin_t, cx - cx * cos_t + cy * sin_t],
        [sin_t, cos_t, cy - cx * sin_t - cy * cos_t],
        [0.0, 0.0, 1.0],
    ])


def _bilinear_sample(image, src_x, src_y, fill_value):
    """Sample ``image`` at float coordinates with bilinear weights."""
    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx = src_x - x0
    wy = src_y - y0
    valid = ((src_x >= 0) & (src_x <= width - 1) &
             (src_y >= 0) & (src_y <= height - 1))
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    if image.ndim == 2:
        top = image[y0c, x0c] * (1 - wx) + image[y0c, x1c] * wx
        bottom = image[y1c, x0c] * (1 - wx) + image[y1c, x1c] * wx
        out = top * (1 - wy) + bottom * wy
        return np.where(valid, out, fill_value)
    top = (image[y0c, x0c, :] * (1 - wx)[..., None] +
           image[y0c, x1c, :] * wx[..., None])
    bottom = (image[y1c, x0c, :] * (1 - wx)[..., None] +
              image[y1c, x1c, :] * wx[..., None])
    out = top * (1 - wy)[..., None] + bottom * wy[..., None]
    return np.where(valid[..., None], out, fill_value)


def warp_affine(image, matrix, output_shape=None, fill_value=0.0):
    """Warp an image with a 2x3 or 3x3 affine matrix.

    The matrix maps input coordinates to output coordinates; the warp
    is computed by inverse mapping, so ``output_shape`` defaults to
    the input shape.  ``fill_value`` fills output pixels that map
    outside the input image.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D grayscale or 3D (H, W, C), got shape "
            f"{arr.shape}")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"image must have 1, 3 or 4 channels, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (2, 3):
        mat = np.vstack([mat, [0.0, 0.0, 1.0]])
    if mat.shape != (3, 3):
        raise ValueError(
            f"matrix must be 2x3 or 3x3, got shape {mat.shape}")
    if abs(np.linalg.det(mat)) < 1e-12:
        raise ValueError("affine matrix must be invertible")
    height, width = arr.shape[:2]
    if output_shape is None:
        out_h, out_w = height, width
    else:
        if len(output_shape) != 2:
            raise ValueError(
                f"output_shape must be (height, width), got {output_shape}")
        out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_shape dimensions must be positive")
    inv = np.linalg.inv(mat)
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    src_x = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    src_y = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return _bilinear_sample(arr, src_x, src_y, fill_value)


def warp_translate(image, dx, dy, output_shape=None, fill_value=0.0):
    """Shift an image right by ``dx`` and down by ``dy`` pixels."""
    matrix = build_affine_matrix(translation=(dx, dy))
    return warp_affine(image, matrix, output_shape, fill_value)


def warp_rotate(image, angle_deg, center=None, output_shape=None,
                fill_value=0.0):
    """Rotate an image by ``angle_deg`` (counter-clockwise).

    ``center`` is (cx, cy); it defaults to the image centre.
    """
    height, width = np.asarray(image).shape[:2]
    if center is None:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return warp_affine(image, rotation_matrix(angle_deg, center),
                       output_shape, fill_value)


def warp_scale(image, scale, center=None, output_shape=None,
               fill_value=0.0):
    """Scale an image by ``scale`` (float, or (sx, sy)) about ``center``.

    ``center`` is (cx, cy); it defaults to the image centre.
    """
    if isinstance(scale, (tuple, list, np.ndarray)):
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)
    if sx == 0 or sy == 0:
        raise ValueError("scale factors must be non-zero")
    height, width = np.asarray(image).shape[:2]
    if center is None:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
    cx, cy = float(center[0]), float(center[1])
    translate_to = np.array([[1.0, 0.0, cx],
                             [0.0, 1.0, cy],
                             [0.0, 0.0, 1.0]])
    translate_back = np.array([[1.0, 0.0, -cx],
                               [0.0, 1.0, -cy],
                               [0.0, 0.0, 1.0]])
    matrix = translate_to @ np.diag([sx, sy, 1.0]) @ translate_back
    return warp_affine(image, matrix, output_shape, fill_value)
