"""Simplified SIFT-like feature detection and description, from scratch.

Implements the core ideas of Lowe's SIFT using only NumPy:

1.  Gaussian scale-space (separable 1-D Gaussian blur) and Difference of
    Gaussians (DoG) across several octaves.
2.  Keypoint detection as local extrema of the DoG over a 3x3x3
    neighbourhood (space x scale), with a contrast threshold and Lowe's
    edge-response test on the 2x2 Hessian.
3.  Orientation assignment from a 36-bin gradient-orientation histogram
    around each keypoint.
4.  A 128-dimensional descriptor: 4x4 spatial cells x 8 orientation bins,
    Gaussian-weighted, L2-normalised with clipping at 0.2 and
    re-normalisation (the standard SIFT recipe).
5.  Descriptor matching with Lowe's ratio test.

Only NumPy is used — the Gaussian blur, gradients and interpolation are
all implemented here.
"""

import math

import numpy as np

__all__ = [
    "SIFTFeatures",
    "match_descriptors",
    "gaussian_kernel_1d",
    "gaussian_blur",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma):
    """Return a normalised 1-D Gaussian kernel for ``sigma``.

    The kernel spans ``ceil(3 * sigma)`` on either side of the centre.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(image, sigma):
    """Separable Gaussian blur of a 2-D array (edge padding)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    # Convolve along rows, then along columns.
    rows = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=padded
    )
    padded_rows = np.pad(rows, ((radius, radius), (0, 0)), mode="edge")
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=padded_rows
    )
    return blurred


def _validate_image(image):
    """Validate a grayscale image and return it as float64."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D grayscale, got {image.ndim}-D")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if min(image.shape) < 32:
        raise ValueError(
            f"image must be at least 32x32, got shape {image.shape}"
        )
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"image must have a numeric dtype, got {image.dtype}")
    image = image.astype(np.float64)
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values (no NaN/inf)")
    return image


def _bilinear_sample(array, ys, xs):
    """Bilinear interpolation of ``array`` at float coordinates ``ys, xs``."""
    height, width = array.shape
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    wy = ys - y0
    wx = xs - x0
    top = array[y0c, x0c] * (1 - wx) + array[y0c, x1c] * wx
    bottom = array[y1c, x0c] * (1 - wx) + array[y1c, x1c] * wx
    return top * (1 - wy) + bottom * wy


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_descriptors(desc1, desc2, ratio=0.75):
    """Match two descriptor sets with Lowe's ratio test.

    Parameters
    ----------
    desc1, desc2 : (N, 128) and (M, 128) float arrays.
    ratio : float in (0, 1]; a match is kept only when the best distance
        is less than ``ratio`` times the second-best distance.

    Returns
    -------
    list of (i, j, distance) tuples, sorted by increasing distance.
    """
    if not 0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")
    desc1 = np.asarray(desc1, dtype=np.float64)
    desc2 = np.asarray(desc2, dtype=np.float64)
    if desc1.ndim != 2 or desc1.shape[1] != 128:
        raise ValueError(
            f"desc1 must have shape (N, 128), got {desc1.shape}"
        )
    if desc2.ndim != 2 or desc2.shape[1] != 128:
        raise ValueError(
            f"desc2 must have shape (M, 128), got {desc2.shape}"
        )
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        raise ValueError("descriptor sets must be non-empty")
    if not (np.all(np.isfinite(desc1)) and np.all(np.isfinite(desc2))):
        raise ValueError("descriptors must contain only finite values")

    # Squared Euclidean distances, (N, M).
    d2 = (
        (desc1 ** 2).sum(axis=1, keepdims=True)
        + (desc2 ** 2).sum(axis=1)[None, :]
        - 2.0 * desc1 @ desc2.T
    )
    np.maximum(d2, 0.0, out=d2)
    matches = []
    for i in range(desc1.shape[0]):
        order = np.argsort(d2[i])
        best = math.sqrt(d2[i, order[0]])
        if desc2.shape[0] == 1:
            matches.append((i, int(order[0]), float(best)))
            continue
        second = math.sqrt(d2[i, order[1]])
        if second > 0 and best < ratio * second:
            matches.append((i, int(order[0]), float(best)))
    matches.sort(key=lambda m: m[2])
    return matches


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SIFTFeatures:
    """Simplified SIFT-like detector/descriptor (NumPy only).

    Parameters
    ----------
    n_octaves : int, number of pyramid octaves (>= 1).
    n_scales : int, number of scales per octave (>= 1).
    sigma0 : float, base blur sigma (> 0).
    contrast_threshold : float, minimum |DoG| for a keypoint (> 0).
    edge_threshold : float, max allowed Hessian trace^2/det ratio (> 0).
    n_bins : int, orientation bins in each descriptor cell (>= 1).
    patch_size : int, descriptor patch side in samples (multiple of 4).
    """

    def __init__(self, n_octaves=3, n_scales=3, sigma0=1.6,
                 contrast_threshold=0.04, edge_threshold=10.0,
                 n_bins=8, patch_size=16):
        if not isinstance(n_octaves, int) or isinstance(n_octaves, bool) \
                or n_octaves < 1:
            raise ValueError(f"n_octaves must be a positive int, got {n_octaves!r}")
        if not isinstance(n_scales, int) or isinstance(n_scales, bool) \
                or n_scales < 1:
            raise ValueError(f"n_scales must be a positive int, got {n_scales!r}")
        if sigma0 <= 0:
            raise ValueError(f"sigma0 must be positive, got {sigma0!r}")
        if contrast_threshold <= 0:
            raise ValueError(
                f"contrast_threshold must be positive, got {contrast_threshold!r}"
            )
        if edge_threshold <= 0:
            raise ValueError(
                f"edge_threshold must be positive, got {edge_threshold!r}"
            )
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError(f"n_bins must be a positive int, got {n_bins!r}")
        if not isinstance(patch_size, int) or patch_size < 4 \
                or patch_size % 4 != 0:
            raise ValueError(
                f"patch_size must be a multiple of 4 >= 4, got {patch_size!r}"
            )
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma0 = float(sigma0)
        self.contrast_threshold = float(contrast_threshold)
        self.edge_threshold = float(edge_threshold)
        self.n_bins = n_bins
        self.patch_size = patch_size
        self._k = 2.0 ** (1.0 / n_scales)

    # -- scale-space ------------------------------------------------------

    def _build_pyramid(self, image):
        """Return (gaussian_levels, dog_levels) per octave.

        gaussian_levels[o] is a list of ``n_scales + 3`` blurred images,
        dog_levels[o] a list of ``n_scales + 2`` DoG images.
        """
        gaussian_levels = []
        dog_levels = []
        base = image
        for _ in range(self.n_octaves):
            sigmas = [self.sigma0 * (self._k ** s)
                      for s in range(self.n_scales + 3)]
            levels = [gaussian_blur(base, s) for s in sigmas]
            dogs = [levels[i + 1] - levels[i]
                    for i in range(len(levels) - 1)]
            gaussian_levels.append(levels)
            dog_levels.append(dogs)
            # Next octave: downsample by 2 from the level at sigma0.
            base = levels[self.n_scales][::2, ::2]
            if min(base.shape) < 32:
                break
        return gaussian_levels, dog_levels

    # -- keypoint detection ------------------------------------------------

    def _find_extrema(self, dogs, octave):
        """Local DoG extrema in one octave -> raw keypoint dicts."""
        scale_factor = 2.0 ** octave
        keypoints = []
        for s in range(1, len(dogs) - 1):
            prev, curr, nxt = dogs[s - 1], dogs[s], dogs[s + 1]
            centre = curr[1:-1, 1:-1]
            is_max = np.ones_like(centre, dtype=bool)
            is_min = np.ones_like(centre, dtype=bool)
            for layer in (prev, curr, nxt):
                patch = layer  # full image; neighbours via slices below
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if layer is curr and dy == 0 and dx == 0:
                            continue
                        neighbour = patch[1 + dy: patch.shape[0] - 1 + dy,
                                           1 + dx: patch.shape[1] - 1 + dx]
                        is_max &= centre > neighbour
                        is_min &= centre < neighbour
            mask = (is_max & (centre >= self.contrast_threshold)) | \
                   (is_min & (centre <= -self.contrast_threshold))
            ys, xs = np.nonzero(mask)
            for y, x in zip(ys, xs):
                y_i, x_i = int(y) + 1, int(x) + 1  # back to full-image coords
                if not self._passes_edge_test(curr, y_i, x_i):
                    continue
                keypoints.append({
                    "y": float(y_i * scale_factor),
                    "x": float(x_i * scale_factor),
                    "scale": float(self.sigma0 * (self._k ** s) * scale_factor),
                    "octave": octave,
                    "orientation": 0.0,
                    "response": float(curr[y_i, x_i]),
                })
        return keypoints

    def _passes_edge_test(self, dog, y, x):
        """Lowe's edge test: reject if trace^2/det is too large."""
        dxx = dog[y, x + 1] - 2 * dog[y, x] + dog[y, x - 1]
        dyy = dog[y + 1, x] - 2 * dog[y, x] + dog[y - 1, x]
        dxy = (dog[y + 1, x + 1] - dog[y + 1, x - 1]
               - dog[y - 1, x + 1] + dog[y - 1, x - 1]) / 4.0
        det = dxx * dyy - dxy * dxy
        if det <= 0:
            return False
        trace = dxx + dyy
        r = self.edge_threshold
        return (trace * trace) / det < ((r + 1.0) ** 2) / r

    def _suppress_nearby(self, keypoints, min_dist=4.0):
        """Keep the strongest keypoint within ``min_dist`` px (same octave)."""
        kept = []
        ordered = sorted(keypoints, key=lambda kp: -abs(kp["response"]))
        for kp in ordered:
            if all((kp["y"] - other["y"]) ** 2 + (kp["x"] - other["x"]) ** 2
                   >= min_dist ** 2 or kp["octave"] != other["octave"]
                   for other in kept):
                kept.append(kp)
        return kept

    # -- orientation ---------------------------------------------------------

    def _assign_orientation(self, level, kp):
        """Dominant gradient orientation around ``kp`` in octave coords."""
        scale_factor = 2.0 ** kp["octave"]
        yo = kp["y"] / scale_factor
        xo = kp["x"] / scale_factor
        gy, gx = np.gradient(level)
        magnitude = np.hypot(gx, gy)
        orientation = np.arctan2(gy, gx)
        radius = max(4, int(round(1.5 * self.patch_size / 2)))
        y0 = max(0, int(math.floor(yo)) - radius)
        y1 = min(level.shape[0], int(math.floor(yo)) + radius + 1)
        x0 = max(0, int(math.floor(xo)) - radius)
        x1 = min(level.shape[1], int(math.floor(xo)) + radius + 1)
        hist = np.zeros(36)
        sigma_w = 0.5 * radius
        for y in range(y0, y1):
            for x in range(x0, x1):
                dist2 = (y - yo) ** 2 + (x - xo) ** 2
                if dist2 > radius * radius:
                    continue
                weight = magnitude[y, x] * math.exp(-dist2 / (2 * sigma_w ** 2))
                bin_idx = int((orientation[y, x] + math.pi)
                              / (2 * math.pi) * 36) % 36
                hist[bin_idx] += weight
        if hist.sum() <= 0:
            return 0.0
        peak = int(np.argmax(hist))
        return (peak + 0.5) * (2 * math.pi / 36) - math.pi

    # -- descriptors ---------------------------------------------------------

    def _descriptor(self, level, kp):
        """128-dim descriptor for ``kp`` sampled from ``level`` (octave coords).

        Returns None when the rotated patch leaves the image.
        """
        scale_factor = 2.0 ** kp["octave"]
        yo = kp["y"] / scale_factor
        xo = kp["x"] / scale_factor
        half = self.patch_size / 2.0
        margin = half * math.sqrt(2.0) + 1.0
        if not (margin <= yo < level.shape[0] - margin
                and margin <= xo < level.shape[1] - margin):
            return None
        gy, gx = np.gradient(level)
        magnitude = np.hypot(gx, gy)
        orientation = np.arctan2(gy, gx)
        cos_o, sin_o = math.cos(-kp["orientation"]), math.sin(-kp["orientation"])
        offsets = np.arange(-half + 0.5, half, 1.0)
        dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
        # Rotate sample offsets by -orientation.
        rx = cos_o * dx_grid - sin_o * dy_grid
        ry = sin_o * dx_grid + cos_o * dy_grid
        sample_y = yo + ry
        sample_x = xo + rx
        mag = _bilinear_sample(magnitude, sample_y, sample_x)
        ori = _bilinear_sample(orientation, sample_y, sample_x)
        rel_ori = (ori - kp["orientation"]) % (2 * math.pi)
        # Gaussian weight falling off from the keypoint centre.
        sigma_w = 0.5 * self.patch_size
        weight = mag * np.exp(
            -(dx_grid ** 2 + dy_grid ** 2) / (2 * sigma_w ** 2))
        cells = self.patch_size // 4
        descriptor = np.zeros(4 * 4 * self.n_bins)
        bin_idx = (rel_ori / (2 * math.pi) * self.n_bins).astype(int) % self.n_bins
        cell_y = np.clip(((dy_grid + half) // cells).astype(int), 0, 3)
        cell_x = np.clip(((dx_grid + half) // cells).astype(int), 0, 3)
        cell = cell_y * 4 + cell_x
        np.add.at(descriptor, (cell * self.n_bins + bin_idx).ravel(),
                  weight.ravel())
        norm = np.linalg.norm(descriptor)
        if norm == 0.0 or not np.isfinite(norm):
            return None
        descriptor /= norm
        descriptor = np.clip(descriptor, 0.0, 0.2)
        norm = np.linalg.norm(descriptor)
        if norm == 0.0:
            return None
        descriptor /= norm
        return descriptor

    # -- public API ------------------------------------------------------------

    def detect(self, image):
        """Detect keypoints in a 2-D grayscale image.

        Returns a list of dicts with keys ``y``, ``x``, ``scale``,
        ``octave``, ``orientation`` (radians) and ``response``.
        """
        image = _validate_image(image)
        gaussian_levels, dog_levels = self._build_pyramid(image)
        keypoints = []
        for octave, dogs in enumerate(dog_levels):
            keypoints.extend(self._find_extrema(dogs, octave))
        keypoints = self._suppress_nearby(keypoints)
        for kp in keypoints:
            # Orientation from the Gaussian level nearest the keypoint scale.
            level_index = min(
                self.n_scales + 1,
                max(0, int(round(math.log(kp["scale"] / (2.0 ** kp["octave"])
                                     / self.sigma0, self._k))) + 1)
            )
            kp["orientation"] = self._assign_orientation(
                gaussian_levels[kp["octave"]][level_index], kp)
        keypoints.sort(key=lambda kp: (kp["y"], kp["x"]))
        return keypoints

    def compute_descriptors(self, image, keypoints):
        """Compute a (N, 128) descriptor array for ``keypoints``.

        Keypoints whose patch falls outside the image get a zero row.
        """
        image = _validate_image(image)
        if not isinstance(keypoints, (list, tuple)):
            raise ValueError("keypoints must be a list of dicts")
        gaussian_levels, _ = self._build_pyramid(image)
        descriptors = np.zeros((len(keypoints), 4 * 4 * self.n_bins))
        for i, kp in enumerate(keypoints):
            if not isinstance(kp, dict) or not all(
                    k in kp for k in ("y", "x", "scale", "octave", "orientation")):
                raise ValueError(
                    "each keypoint must be a dict with y, x, scale, "
                    "octave and orientation"
                )
            level_index = min(
                self.n_scales + 1,
                max(0, int(round(math.log(kp["scale"] / (2.0 ** kp["octave"])
                                     / self.sigma0, self._k))) + 1)
            )
            desc = self._descriptor(
                gaussian_levels[kp["octave"]][level_index], kp)
            if desc is not None:
                descriptors[i] = desc
        return descriptors

    def detect_and_describe(self, image):
        """Detect keypoints and compute their descriptors.

        Returns (keypoints, descriptors) with only keypoints that produced
        a valid (non-zero, unit-norm) descriptor.
        """
        keypoints = self.detect(image)
        descriptors = self.compute_descriptors(image, keypoints)
        norms = np.linalg.norm(descriptors, axis=1)
        valid = norms > 0
        return [kp for kp, ok in zip(keypoints, valid) if ok], \
            descriptors[valid]
