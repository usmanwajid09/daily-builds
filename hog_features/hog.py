"""Histogram of Oriented Gradients (HOG) descriptor, from scratch.

The image is split into cells; each cell votes into an orientation
histogram weighted by gradient magnitude. Cells are grouped into
overlapping blocks, each block is L2-normalised (with L2-Hys clipping),
and the blocks are concatenated into one feature vector. Only numpy.
"""

import numpy as np


class HOGDescriptor:
    """HOG feature extractor for grayscale images.

    Parameters
    ----------
    cell_size : int
        Pixels per cell side (image is cropped to a multiple of this).
    block_size : int
        Cells per block side; blocks slide by one cell.
    n_bins : int
        Orientation bins over 0..180 degrees (unsigned gradients).
    """

    def __init__(self, cell_size=8, block_size=2, n_bins=9):
        for name, value in (("cell_size", cell_size), ("block_size", block_size), ("n_bins", n_bins)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.cell_size = int(cell_size)
        self.block_size = int(block_size)
        self.n_bins = int(n_bins)

    def _validate(self, image):
        arr = np.asarray(image, dtype=float)
        if arr.ndim != 2:
            raise ValueError("image must be a 2D grayscale array")
        if arr.shape[0] < self.cell_size or arr.shape[1] < self.cell_size:
            raise ValueError(
                f"image must be at least {self.cell_size}x{self.cell_size} pixels, "
                f"got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("image must contain only finite values")
        return arr

    def feature_length(self, image_shape):
        """Length of the descriptor for an image of the given (h, w) shape."""
        h, w = image_shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        if n_blocks_y < 1 or n_blocks_x < 1:
            raise ValueError("image is too small for the chosen cell/block size")
        return n_blocks_y * n_blocks_x * (self.block_size ** 2) * self.n_bins

    def compute(self, image):
        """Return the HOG feature vector (1D float array)."""
        img = self._validate(image)
        h = (img.shape[0] // self.cell_size) * self.cell_size
        w = (img.shape[1] // self.cell_size) * self.cell_size
        img = img[:h, :w]

        gy, gx = np.gradient(img)
        magnitude = np.hypot(gx, gy)
        orientation = np.degrees(np.arctan2(gy, gx)) % 180.0

        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size
        cell_hist = np.zeros((n_cells_y, n_cells_x, self.n_bins))
        bin_width = 180.0 / self.n_bins
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                mag = magnitude[
                    cy * self.cell_size:(cy + 1) * self.cell_size,
                    cx * self.cell_size:(cx + 1) * self.cell_size,
                ].ravel()
                ori = orientation[
                    cy * self.cell_size:(cy + 1) * self.cell_size,
                    cx * self.cell_size:(cx + 1) * self.cell_size,
                ].ravel()
                bins = np.minimum((ori / bin_width).astype(int), self.n_bins - 1)
                # Split each vote between the two nearest bin centres.
                centres = (bins + 0.5) * bin_width
                frac = (ori - centres) / bin_width
                lower = bins
                upper = (bins + 1) % self.n_bins
                np.add.at(cell_hist[cy, cx], lower, mag * (1.0 - np.abs(frac)))
                np.add.at(cell_hist[cy, cx], upper, mag * np.abs(frac))

        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        if n_blocks_y < 1 or n_blocks_x < 1:
            raise ValueError("image is too small for the chosen cell/block size")
        features = []
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block = cell_hist[
                    by:by + self.block_size, bx:bx + self.block_size
                ].ravel()
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / norm
                # L2-Hys: clip at 0.2 and renormalise.
                block = np.minimum(block, 0.2)
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / norm
                features.append(block)
        return np.concatenate(features)
