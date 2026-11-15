"""Image segmentation via k-means clustering, from scratch with numpy only.

Pixel colours are clustered with k-means (k-means++ initialisation,
Lloyd iterations to convergence). The result is a label map saying
which cluster each pixel belongs to and a segmented image where every
pixel is replaced by its cluster centre colour. Works on grayscale or
RGB images.
"""

import numpy as np


class KMeansSegmenter:
    """K-means image segmenter.

    Parameters
    ----------
    n_clusters : int
        Number of colour clusters (segments). Must be >= 2.
    max_iter : int
        Maximum Lloyd iterations. Must be >= 1.
    tol : float
        Stop when the largest centre shift is below this. Must be > 0.
    seed : int
        Random seed for k-means++ initialisation.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, seed=0):
        if not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, bool):
            raise ValueError("n_clusters must be an integer >= 2")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = seed
        self.centers_ = None
        self.labels_ = None
        self.n_iter_ = 0

    @staticmethod
    def _prepare(image):
        arr = np.asarray(image)
        if arr.ndim not in (2, 3):
            raise ValueError("image must be 2D (grayscale) or 3D (h, w, channels)")
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("image must have at least 1 pixel")
        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            raise ValueError("colour images must have 1, 3 or 4 channels")
        if not np.isfinite(arr.astype(float)).all():
            raise ValueError("image must contain only finite values")
        arr = arr.astype(float)
        # Normalise to 0..1 regardless of whether input is 0..255 or 0..1.
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.min() < 0.0 or arr.max() > 1.0:
            raise ValueError("pixel values must be in [0, 1] or [0, 255]")
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    def _init_centers(self, pixels, rng):
        n = pixels.shape[0]
        centers = [pixels[rng.integers(n)]]
        closest_sq = np.sum((pixels - centers[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                centers.append(pixels[rng.integers(n)])
            else:
                probs = closest_sq / total
                centers.append(pixels[rng.choice(n, p=probs)])
            closest_sq = np.minimum(
                closest_sq, np.sum((pixels - centers[-1]) ** 2, axis=1)
            )
        return np.vstack(centers)

    def fit(self, image):
        """Cluster the image's pixel colours. Returns self."""
        arr = self._prepare(image)
        h, w, c = arr.shape
        pixels = arr.reshape(-1, c)
        if self.n_clusters > pixels.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds the number of pixels ({pixels.shape[0]})"
            )
        rng = np.random.default_rng(self.seed)
        centers = self._init_centers(pixels, rng)
        labels = np.zeros(pixels.shape[0], dtype=int)
        for it in range(1, self.max_iter + 1):
            dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = pixels[new_labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
            shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers, labels = new_centers, new_labels
            self.n_iter_ = it
            if shift < self.tol:
                break
        self.centers_ = centers
        self.labels_ = labels.reshape(h, w)
        return self

    def _require_fit(self):
        if self.centers_ is None:
            raise ValueError("KMeansSegmenter is not fitted yet; call fit() first")

    def segment(self, image=None):
        """Return the segmented image (each pixel replaced by its centre).

        If image is given it must have the same shape as the fitted image;
        otherwise the fitted image is segmented.
        """
        self._require_fit()
        if image is None:
            labels = self.labels_
        else:
            arr = self._prepare(image)
            if arr.shape[:2] != self.labels_.shape:
                raise ValueError("image shape does not match the fitted image")
            pixels = arr.reshape(-1, arr.shape[2])
            dists = np.sum((pixels[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1).reshape(arr.shape[:2])
        segmented = self.centers_[labels]
        if segmented.shape[2] == 1:
            segmented = segmented[:, :, 0]
        return segmented

    def cluster_sizes(self):
        """Number of pixels in each cluster."""
        self._require_fit()
        return np.bincount(self.labels_.ravel(), minlength=self.n_clusters)


def segment_image(image, n_clusters=3, seed=0, max_iter=100):
    """Segment an image into ``n_clusters`` colour regions.

    Returns ``(labels, segmented, centers)`` where ``labels`` is a
    (h, w) int array, ``segmented`` is the image with each pixel
    replaced by its cluster centre (values in [0, 1]), and ``centers``
    is a (n_clusters, channels) array.
    """
    seg = KMeansSegmenter(n_clusters=n_clusters, seed=seed, max_iter=max_iter).fit(image)
    return seg.labels_, seg.segment(), seg.centers_
