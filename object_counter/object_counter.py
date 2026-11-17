"""Count objects in a binary image from scratch (numpy only).

Connected-components labelling implemented with a classic two-pass
algorithm and a union-find structure written by hand -- no OpenCV,
no scipy. Each foreground object gets its area (pixel count),
centroid (row, col) and bounding box.
"""

import numpy as np

__all__ = ["label_image", "count_objects", "ObjectInfo"]


class _UnionFind:
    """Small union-find with path compression and union by rank."""

    def __init__(self):
        self.parent = [0]  # index 0 is the background label, never used
        self.rank = [0]

    def make_set(self):
        label = len(self.parent)
        self.parent.append(label)
        self.rank.append(0)
        return label

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _validate_binary(image):
    """Return the image as a uint8 0/1 array, or raise ValueError."""
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"binary_image must be a numpy array, got {type(image).__name__}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"binary_image must be 2D, got {image.ndim}D with shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("binary_image must not be empty")
    if image.dtype == bool:
        return image.astype(np.uint8)
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(
            f"binary_image must contain 0/1 integers or booleans, "
            f"got dtype {image.dtype}"
        )
    values = np.unique(image)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(
            f"binary_image values must be 0 or 1, got values {values.tolist()}"
        )
    return image.astype(np.uint8)


def _validate_connectivity(connectivity):
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}"
        )


def _validate_min_size(min_size):
    if isinstance(min_size, bool) or not isinstance(min_size, (int, np.integer)):
        raise ValueError(f"min_size must be an integer >= 1, got {min_size!r}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")


def label_image(binary_image, connectivity=8):
    """Label the connected components of a binary image.

    Parameters
    ----------
    binary_image : (H, W) ndarray of 0/1 ints or bools
        Foreground pixels are 1, background pixels are 0.
    connectivity : {4, 8}
        4-connectivity joins only edge neighbours; 8-connectivity also
        joins diagonal neighbours.

    Returns
    -------
    labelled : (H, W) int32 ndarray
        Background is 0; each object gets a unique label 1..N in
        top-to-bottom, left-to-right order of first appearance.
    """
    img = _validate_binary(binary_image)
    _validate_connectivity(connectivity)
    rows, cols = img.shape
    provisional = np.zeros((rows, cols), dtype=np.int32)
    uf = _UnionFind()

    # Neighbours already visited in a top-to-bottom, left-to-right scan.
    if connectivity == 4:
        offsets = ((-1, 0), (0, -1))
    else:
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1))

    # Pass 1: provisional labels + record equivalences.
    for r in range(rows):
        for c in range(cols):
            if img[r, c] == 0:
                continue
            neighbour_labels = {
                provisional[r + dr, c + dc]
                for dr, dc in offsets
                if 0 <= r + dr < rows and 0 <= c + dc < cols
                and provisional[r + dr, c + dc] != 0
            }
            if not neighbour_labels:
                provisional[r, c] = uf.make_set()
            else:
                smallest = min(neighbour_labels)
                provisional[r, c] = smallest
                for other in neighbour_labels:
                    uf.union(smallest, other)

    # Pass 2: resolve each provisional label to its root and renumber
    # consecutively in scan order.
    root_to_final = {}
    labelled = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            p = provisional[r, c]
            if p == 0:
                continue
            root = uf.find(p)
            final = root_to_final.get(root)
            if final is None:
                final = len(root_to_final) + 1
                root_to_final[root] = final
            labelled[r, c] = final
    return labelled


class ObjectInfo(dict):
    """One detected object: label, area, centroid and bounding box.

    Behaves like a dict with keys ``label``, ``area``, ``centroid``
    (row, col as floats) and ``bbox`` (min_row, min_col, max_row,
    max_col, inclusive) -- plus attribute access for convenience.
    """

    def __init__(self, label, area, centroid, bbox):
        super().__init__(
            label=int(label),
            area=int(area),
            centroid=(float(centroid[0]), float(centroid[1])),
            bbox=tuple(int(v) for v in bbox),
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def count_objects(binary_image, min_size=1, connectivity=8):
    """Count the objects in a binary image.

    Parameters
    ----------
    binary_image : (H, W) ndarray of 0/1 ints or bools
    min_size : int, default 1
        Objects with fewer than ``min_size`` pixels are ignored
        (handy for dropping speckle noise).
    connectivity : {4, 8}, default 8

    Returns
    -------
    list[ObjectInfo]
        One entry per object, sorted by label (scan order). Each entry
        has ``label``, ``area`` (pixel count), ``centroid`` (row, col)
        and ``bbox`` (min_row, min_col, max_row, max_col, inclusive).
    """
    _validate_min_size(min_size)
    _validate_connectivity(connectivity)
    labelled = label_image(binary_image, connectivity=connectivity)
    objects = []
    for label in range(1, int(labelled.max()) + 1):
        rows, cols = np.nonzero(labelled == label)
        area = len(rows)
        if area < min_size:
            continue
        objects.append(
            ObjectInfo(
                label=label,
                area=area,
                centroid=(float(rows.mean()), float(cols.mean())),
                bbox=(
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
            )
        )
    return objects
