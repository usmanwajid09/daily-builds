"""Document-type classifier.

Classifies synthetic document images as ``invoice``, ``letter``, ``form``
or ``receipt`` from layout / texture features (ink density, line counts,
margins, enclosed boxes) with a nearest-centroid classifier.

Everything is numpy + stdlib; the document images are generated in code,
so no external files are needed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CLASSES",
    "HEIGHT",
    "WIDTH",
    "DocumentClassifier",
    "accuracy",
    "confusion_matrix",
    "extract_features",
    "fit",
    "make_document",
    "predict",
    "predict_proba",
    "train_classifier",
]

CLASSES = ("invoice", "letter", "form", "receipt")
HEIGHT = 150
WIDTH = 200

# Feature names, in the order extract_features returns them.
FEATURE_NAMES = (
    "ink_density",
    "h_line_count",
    "v_line_count",
    "left_margin",
    "right_margin",
    "top_margin",
    "bottom_margin",
    "text_line_count",
    "mean_line_length",
    "ink_top_third",
    "ink_middle_third",
    "ink_bottom_third",
    "box_count",
    "mean_run_length",
)

_WHITE = 255
_INK = 0


# ---------------------------------------------------------------------------
# Drawing helpers (all coordinates in pixels, x = column, y = row)
# ---------------------------------------------------------------------------

def _fill(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Fill the rectangle [x0, x1) x [y0, y1) with ink, clipped to canvas."""
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = _INK


def _hline(img: np.ndarray, y: int, x0: int, x1: int, thickness: int = 1) -> None:
    _fill(img, x0, y, x1, y + thickness)


def _vline(img: np.ndarray, x: int, y0: int, y1: int, thickness: int = 1) -> None:
    _fill(img, x, y0, x + thickness, y1)


def _box(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int = 1) -> None:
    """Draw an outlined (hollow) rectangle."""
    _hline(img, y0, x0, x1, thickness)
    _hline(img, y1 - thickness, x0, x1, thickness)
    _vline(img, x0, y0, y1, thickness)
    _vline(img, x1 - thickness, y0, y1, thickness)


def _text_line(img: np.ndarray, y: int, x0: int, x1: int, rng: np.random.Generator,
               thickness: int = 2, word_min: int = 4, word_max: int = 11) -> None:
    """Draw one fake line of text: a row of short word-like ink blobs."""
    x = x0
    while x < x1 - word_min:
        w = int(rng.integers(word_min, word_max + 1))
        w = min(w, x1 - x)
        if w <= 0:
            break
        _fill(img, x, y, x + w, y + thickness)
        x += w + int(rng.integers(2, 5))


def _paragraph(img: np.ndarray, y0: int, y1: int, x0: int, x1: int,
               rng: np.random.Generator, step: int = 6) -> None:
    for y in range(y0, y1, step):
        # Last line of a paragraph is usually shorter.
        end = x1 if rng.random() > 0.25 else int(x0 + (x1 - x0) * rng.uniform(0.5, 0.85))
        _text_line(img, y, x0, end, rng)


# ---------------------------------------------------------------------------
# Synthetic document generation
# ---------------------------------------------------------------------------

def _make_invoice(rng: np.random.Generator) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH), _WHITE, dtype=np.uint8)
    # Header block: solid logo block + title line.
    _fill(img, 8, 8, 8 + int(rng.integers(38, 52)), 8 + int(rng.integers(10, 15)))
    _text_line(img, 26, 8, 90, rng)
    _hline(img, 34, 8, 192, 2)
    # Table grid: 5 columns, ~8 rows.
    x_left, x_right = 8, 192
    y_top, y_bottom = 40, 132
    col_edges = [8, 62, 108, 148, 170, 192]
    for y in range(y_top, y_bottom + 1, 11):
        _hline(img, y, x_left, x_right, 1)
    for x in col_edges:
        _vline(img, x, y_top, y_bottom, 1)
    # Short text segments inside the table cells.
    for r in range(8):
        y = y_top + 3 + r * 11
        for c in range(5):
            _text_line(img, y, col_edges[c] + 3, col_edges[c + 1] - 3, rng)
    # Total line at the bottom.
    _hline(img, 138, 120, 192, 2)
    _text_line(img, 140, 122, 190, rng)
    return img


def _make_letter(rng: np.random.Generator) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH), _WHITE, dtype=np.uint8)
    # Sender block top-left, date top-right.
    _text_line(img, 10, 12, 70, rng)
    _text_line(img, 16, 12, 60, rng)
    _text_line(img, 10, 150, 188, rng)
    _hline(img, 26, 12, 188, 1)
    # Dense paragraphs, almost full width.
    _paragraph(img, 34, 78, 12, 188, rng)
    _paragraph(img, 84, 122, 12, 188, rng)
    # Closing + signature line.
    _text_line(img, 130, 12, 70, rng)
    _hline(img, 142, 120, 188, 1)
    return img


def _make_form(rng: np.random.Generator) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH), _WHITE, dtype=np.uint8)
    # Wide margins: content lives in x = 30..170.
    _text_line(img, 10, 78, 122, rng)  # centred title
    _hline(img, 18, 30, 170, 2)
    y = 28
    for _ in range(5):
        # Label on the left, empty outlined field box on the right.
        _text_line(img, y + 3, 32, 78, rng)
        _box(img, 88, y, 168, y + 11)
        y += 16
    # Checkbox row.
    for i in range(3):
        cx = 32 + i * 46
        _box(img, cx, y + 2, cx + 7, y + 9)
        _text_line(img, y + 3, cx + 10, cx + 42, rng)
    y += 20
    # Signature box at the bottom.
    _text_line(img, y + 2, 32, 70, rng)
    _box(img, 88, y, 168, y + 12)
    return img


def _make_receipt(rng: np.random.Generator) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH), _WHITE, dtype=np.uint8)
    # Narrow column: content lives in x = 55..145 (90 px wide).
    x0, x1 = 55, 145
    _fill(img, 72, 6, 128, 16)  # centred shop-name block
    _text_line(img, 22, x0 + 8, x1 - 8, rng)
    _hline(img, 28, x0, x1, 1)
    # Many short item lines.
    for y in range(34, 108, 6):
        _text_line(img, y, x0 + 2, x1 - 2, rng)
    _hline(img, 112, x0, x1, 1)
    # Dense total block at the bottom.
    _fill(img, x0, 116, x1, 120)
    _text_line(img, 124, x0 + 4, x1 - 4, rng)
    _text_line(img, 130, x0 + 4, x1 - 4, rng)
    _hline(img, 138, x0, x1, 2)
    return img


_GENERATORS = {
    "invoice": _make_invoice,
    "letter": _make_letter,
    "form": _make_form,
    "receipt": _make_receipt,
}


def make_document(doc_type: str, seed: int = 0) -> np.ndarray:
    """Render a synthetic grayscale document image (``HEIGHT`` x ``WIDTH``).

    Parameters
    ----------
    doc_type: one of ``"invoice"``, ``"letter"``, ``"form"``, ``"receipt"``.
    seed: integer seed; the same (doc_type, seed) always gives the same image.

    Returns
    -------
    np.ndarray of shape (150, 200), dtype uint8, 255 = paper, 0 = ink.
    """
    if doc_type not in _GENERATORS:
        raise ValueError(
            f"unknown document type {doc_type!r}; expected one of {list(CLASSES)}"
        )
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    rng = np.random.default_rng(int(seed))
    return _GENERATORS[doc_type](rng)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim}-D with shape {image.shape}")
    if image.size == 0:
        raise ValueError("image is empty")
    return image


def _long_runs(mask: np.ndarray, axis: int, min_len: int) -> np.ndarray:
    """Boolean array marking rows (axis=0) / columns (axis=1) that contain a
    run of at least ``min_len`` consecutive True values."""
    n = mask.shape[axis]
    out = np.zeros(n, dtype=bool)
    moved = np.moveaxis(mask, axis, 0)
    for i in range(n):
        row = moved[i]
        run = 0
        for v in row:
            run = run + 1 if v else 0
            if run >= min_len:
                out[i] = True
                break
    return out


def _count_boxes(ink: np.ndarray) -> int:
    """Approximate number of enclosed boxes: intersections of long
    horizontal and long vertical ink runs, divided by 4 corners per box."""
    h_rows = _long_runs(ink, axis=0, min_len=25)
    v_cols = _long_runs(ink, axis=1, min_len=12)
    if not h_rows.any() or not v_cols.any():
        return 0
    corners = int(np.count_nonzero(ink[np.ix_(h_rows, v_cols)]))
    return corners // 4


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract a 14-dimensional layout/texture feature vector from a
    grayscale document image (ink = pixels < 128)."""
    image = _validate_image(image)
    ink = image < 128
    h, w = ink.shape
    total = ink.size
    ink_density = float(ink.mean())

    row_ink = ink.mean(axis=1)  # fraction of ink per row
    col_ink = ink.mean(axis=0)  # fraction of ink per column

    h_line_count = int(np.count_nonzero(row_ink > 0.5))
    v_line_count = int(np.count_nonzero(col_ink > 0.5))

    ink_rows = np.flatnonzero(row_ink > 0)
    ink_cols = np.flatnonzero(col_ink > 0)
    if ink_rows.size == 0:  # blank page
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    left_margin = float(ink_cols[0] / w)
    right_margin = float((w - 1 - ink_cols[-1]) / w)
    top_margin = float(ink_rows[0] / h)
    bottom_margin = float((h - 1 - ink_rows[-1]) / h)

    # Text lines: runs of consecutive ink rows.
    ink_row_mask = row_ink > 0
    padded = np.concatenate(([False], ink_row_mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    text_line_count = float(len(starts))
    line_lengths = []
    for s, e in zip(starts, ends):
        line_lengths.append(float(ink[s:e].sum() / max(1, e - s)))
    mean_line_length = float(np.mean(line_lengths)) if line_lengths else 0.0

    third = h // 3
    ink_top = float(ink[:third].mean())
    ink_mid = float(ink[third:2 * third].mean())
    ink_bot = float(ink[2 * third:].mean())

    box_count = float(_count_boxes(ink))

    # Mean length of horizontal ink runs (word-stroke length proxy).
    run_lengths = []
    for row in ink:
        run = 0
        for v in row:
            if v:
                run += 1
            elif run:
                run_lengths.append(run)
                run = 0
        if run:
            run_lengths.append(run)
    mean_run = float(np.mean(run_lengths)) if run_lengths else 0.0

    return np.array([
        ink_density,
        h_line_count,
        v_line_count,
        left_margin,
        right_margin,
        top_margin,
        bottom_margin,
        text_line_count,
        mean_line_length,
        ink_top,
        ink_mid,
        ink_bot,
        box_count,
        mean_run,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Nearest-centroid classifier
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """Nearest-centroid classifier over document layout features.

    Each class is summarised by the mean of its (standardised) training
    feature vectors; a new document is assigned to the nearest centroid.
    """

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.centroids_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._fitted = False

    # -- fitting -----------------------------------------------------------
    def fit(self, images, labels) -> "DocumentClassifier":
        images = list(images)
        labels = list(labels)
        if len(images) == 0:
            raise ValueError("cannot fit on an empty training set")
        if len(images) != len(labels):
            raise ValueError(
                f"{len(images)} images but {len(labels)} labels; counts must match"
            )
        for label in labels:
            if label not in CLASSES:
                raise ValueError(
                    f"unknown label {label!r}; expected one of {list(CLASSES)}"
                )
        X = np.array([extract_features(img) for img in images], dtype=float)
        y = np.array(labels)
        self.classes_ = [c for c in CLASSES if c in set(labels)]
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        self.centroids_ = np.array(
            [Xs[y == c].mean(axis=0) for c in self.classes_], dtype=float
        )
        self._fitted = True
        return self

    # -- prediction --------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("classifier is not fitted yet; call fit() first")

    def _distances(self, image: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = (extract_features(image) - self.mean_) / self.std_
        return np.linalg.norm(self.centroids_ - x, axis=1)

    def predict(self, image: np.ndarray) -> str:
        """Return the predicted document type for one image."""
        dists = self._distances(image)
        return self.classes_[int(np.argmin(dists))]

    def predict_proba(self, image: np.ndarray) -> dict[str, float]:
        """Return a {class: probability} dict from softmax over negative
        centroid distances (closer centroid -> higher probability)."""
        dists = self._distances(image)
        scores = np.exp(-dists)
        scores /= scores.sum()
        return {c: float(p) for c, p in zip(self.classes_, scores)}

    def score(self, images, labels) -> float:
        """Fraction of correctly classified images."""
        images = list(images)
        labels = list(labels)
        if len(images) != len(labels):
            raise ValueError(
                f"{len(images)} images but {len(labels)} labels; counts must match"
            )
        if len(images) == 0:
            raise ValueError("cannot score an empty set")
        correct = sum(self.predict(img) == lab for img, lab in zip(images, labels))
        return correct / len(images)


def train_classifier(images, labels) -> DocumentClassifier:
    """Fit and return a :class:`DocumentClassifier` on the given data."""
    return DocumentClassifier().fit(images, labels)


def fit(images, labels) -> DocumentClassifier:
    """Alias for :func:`train_classifier`."""
    return train_classifier(images, labels)


def predict(classifier: DocumentClassifier, image: np.ndarray) -> str:
    """Predict with an already-fitted classifier."""
    return classifier.predict(image)


def predict_proba(classifier: DocumentClassifier, image: np.ndarray) -> dict[str, float]:
    """Class probabilities from an already-fitted classifier."""
    return classifier.predict_proba(image)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(classifier: DocumentClassifier, images, labels) -> float:
    """Overall accuracy of a fitted classifier on a labelled set."""
    return classifier.score(images, labels)


def confusion_matrix(classifier: DocumentClassifier, images, labels) -> dict[str, dict[str, int]]:
    """Nested {true: {predicted: count}} confusion counts."""
    images = list(images)
    labels = list(labels)
    matrix: dict[str, dict[str, int]] = {
        t: {p: 0 for p in CLASSES} for t in CLASSES
    }
    for img, true in zip(images, labels):
        if true not in CLASSES:
            raise ValueError(f"unknown label {true!r}; expected one of {list(CLASSES)}")
        matrix[true][classifier.predict(img)] += 1
    return matrix
