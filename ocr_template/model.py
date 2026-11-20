"""Template-matching OCR for printed digits and letters (numpy only).

A 5x7 bitmap font defines templates for the digits 0-9 and the letters
A-Z. Text is rendered into a binary image with that font, characters
are segmented by fixed-pitch cells (glyph width 5, spacing 1, with a
blank cell meaning a word space), each cell is resampled to 5x7 and
matched against the templates by Hamming distance, and the best
template wins. No image files or external fonts are needed.
"""

import numpy as np

GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7
GLYPH_STEP = GLYPH_WIDTH + 1  # glyph + one blank spacing column


def _glyph(rows):
    return np.array([[1.0 if ch == "#" else 0.0 for ch in row]
                     for row in rows], dtype=np.float64)


TEMPLATES = {
    "0": _glyph([".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#",
                 ".###."]),
    "1": _glyph(["..#..", ".##..", "..#..", "..#..", "..#..", "..#..",
                 ".###."]),
    "2": _glyph([".###.", "#...#", "....#", "...#.", "..#..", ".#...",
                 "#####"]),
    "3": _glyph(["####.", "....#", "....#", ".###.", "....#", "....#",
                 "####."]),
    "4": _glyph(["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.",
                 "...#."]),
    "5": _glyph(["#####", "#....", "####.", "....#", "....#", "#...#",
                 ".###."]),
    "6": _glyph([".###.", "#...#", "#....", "####.", "#...#", "#...#",
                 ".###."]),
    "7": _glyph(["#####", "....#", "...#.", "..#..", "..#..", "..#..",
                 "..#.."]),
    "8": _glyph([".###.", "#...#", "#...#", ".###.", "#...#", "#...#",
                 ".###."]),
    "9": _glyph([".###.", "#...#", "#...#", ".####", "....#", "#...#",
                 ".###."]),
    "A": _glyph([".###.", "#...#", "#...#", "#####", "#...#", "#...#",
                 "#...#"]),
    "B": _glyph(["####.", "#...#", "#...#", "####.", "#...#", "#...#",
                 "####."]),
    "C": _glyph([".###.", "#...#", "#....", "#....", "#....", "#...#",
                 ".###."]),
    "D": _glyph(["####.", "#...#", "#...#", "#...#", "#...#", "#...#",
                 "####."]),
    "E": _glyph(["#####", "#....", "#....", "####.", "#....", "#....",
                 "#####"]),
    "F": _glyph(["#####", "#....", "#....", "####.", "#....", "#....",
                 "#...."]),
    "G": _glyph([".###.", "#...#", "#....", "#.###", "#...#", "#...#",
                 ".###."]),
    "H": _glyph(["#...#", "#...#", "#...#", "#####", "#...#", "#...#",
                 "#...#"]),
    "I": _glyph([".###.", "..#..", "..#..", "..#..", "..#..", "..#..",
                 ".###."]),
    "J": _glyph(["..###", "...#.", "...#.", "...#.", "...#.", "#..#.",
                 ".##.."]),
    "K": _glyph(["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.",
                 "#...#"]),
    "L": _glyph(["#....", "#....", "#....", "#....", "#....", "#....",
                 "#####"]),
    "M": _glyph(["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#",
                 "#...#"]),
    "N": _glyph(["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#",
                 "#...#"]),
    "O": _glyph([".###.", "#...#", "#...#", "#...#", "#...#", "#...#",
                 ".###."]),
    "P": _glyph(["####.", "#...#", "#...#", "####.", "#....", "#....",
                 "#...."]),
    "Q": _glyph([".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.",
                 ".##.#"]),
    "R": _glyph(["####.", "#...#", "#...#", "####.", "#.#..", "#..#.",
                 "#...#"]),
    "S": _glyph([".####", "#....", "#....", ".###.", "....#", "....#",
                 "####."]),
    "T": _glyph(["#####", "..#..", "..#..", "..#..", "..#..", "..#..",
                 "..#.."]),
    "U": _glyph(["#...#", "#...#", "#...#", "#...#", "#...#", "#...#",
                 ".###."]),
    "V": _glyph(["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.",
                 "..#.."]),
    "W": _glyph(["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##",
                 "#...#"]),
    "X": _glyph(["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#",
                 "#...#"]),
    "Y": _glyph(["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..",
                 "..#.."]),
    "Z": _glyph(["#####", "....#", "...#.", "..#..", ".#...", "#....",
                 "#####"]),
    " ": np.zeros((GLYPH_HEIGHT, GLYPH_WIDTH), dtype=np.float64),
}

CHARACTERS = tuple(sorted(c for c in TEMPLATES if c != " "))


def render_text(text, scale=1):
    """Render ``text`` (digits, A-Z, spaces) as a binary image.

    ``scale`` magnifies each font pixel into a ``scale x scale`` block.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if int(scale) < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    scale = int(scale)
    for ch in text:
        if ch not in TEMPLATES:
            raise ValueError(
                f"unsupported character {ch!r}: use digits, A-Z or space")
    width = len(text) * GLYPH_STEP * scale - scale
    height = GLYPH_HEIGHT * scale
    image = np.zeros((height, width), dtype=np.float64)
    for idx, ch in enumerate(text):
        glyph = TEMPLATES[ch]
        if scale > 1:
            glyph = np.repeat(np.repeat(glyph, scale, axis=0),
                              scale, axis=1)
        col = idx * GLYPH_STEP * scale
        image[:, col:col + GLYPH_WIDTH * scale] = glyph
    return image


def _resample_to_template(cell):
    """Resample a binary ``cell`` to the 5x7 template grid."""
    cell = np.asarray(cell, dtype=np.float64)
    if cell.ndim != 2 or cell.shape[0] < 1 or cell.shape[1] < 1:
        raise ValueError(f"cell must be a non-empty 2D array, got shape "
                         f"{cell.shape}")
    rows = (np.arange(GLYPH_HEIGHT) * cell.shape[0] / GLYPH_HEIGHT).astype(int)
    cols = (np.arange(GLYPH_WIDTH) * cell.shape[1] / GLYPH_WIDTH).astype(int)
    return (cell[np.ix_(rows, cols)] > 0.5).astype(np.float64)


def _validate_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.size == 0:
        raise ValueError("image is empty")
    return (image > 0.5).astype(np.float64)


def _detect_scale(binary):
    """Font scale from the ink height (glyphs are 7 rows tall)."""
    rows = np.nonzero(binary.any(axis=1))[0]
    height = int(rows[-1] - rows[0] + 1)
    scale = max(1, int(round(height / GLYPH_HEIGHT)))
    return scale, int(rows[0]), int(rows[-1]) + 1


def _is_blank(cell):
    """True if ``cell`` holds (almost) no ink — a word space, possibly
    with a few stray noise pixels (up to 4% of the cell). A real glyph
    always carries far more ink than that."""
    cell = np.asarray(cell, dtype=np.float64)
    return float(cell.sum()) <= max(2.0, 0.04 * cell.size)


def segment_characters(image):
    """Split a rendered-text ``image`` into per-character cells.

    The font scale is detected from the ink height, and the cell phase
    (where the first glyph starts) is found by trying every offset
    within one pitch and keeping the alignment with the lowest mean
    template-matching distance — so left/right margins and light
    noise do not break segmentation. A fully blank cell decodes as a
    space. Returns a list of 2D arrays.
    """
    binary = _validate_image(image)
    if binary.sum() == 0:
        raise ValueError("image has no ink: nothing to segment")
    scale, r0, r1 = _detect_scale(binary)
    cropped = binary[r0:r1, :]
    step = GLYPH_STEP * scale
    width = GLYPH_WIDTH * scale

    def _cells_for_phase(phase):
        cells = []
        col = phase
        while col + width <= cropped.shape[1]:
            cells.append(cropped[:, col:col + width])
            col += step
        if col < cropped.shape[1]:
            cell = cropped[:, col:]
            cells.append(np.pad(
                cell, ((0, 0), (0, width - cell.shape[1]))))
        return cells

    best = None
    for phase in range(step):
        cells = _cells_for_phase(phase)
        total, n_ink = 0, 0
        for cell in cells:
            if _is_blank(cell):
                continue
            total += classify_character(cell)[1]
            n_ink += 1
        if n_ink == 0:
            continue
        score = total / n_ink
        if best is None or score < best[0]:
            best = (score, cells)
    if best is None:
        raise ValueError("image has no ink: nothing to segment")
    cells = best[1]
    while cells and _is_blank(cells[0]):
        cells.pop(0)
    while cells and _is_blank(cells[-1]):
        cells.pop()
    if not cells:
        raise ValueError("image has no ink: nothing to segment")
    return cells


def classify_character(cell):
    """Best-matching template character for one glyph ``cell``.

    Returns ``(character, hamming_distance)``; the distance is the
    number of differing pixels out of 35.
    """
    grid = _resample_to_template(cell)
    best_char, best_dist = None, GLYPH_WIDTH * GLYPH_HEIGHT + 1
    for ch in CHARACTERS:
        dist = int(np.sum(grid != TEMPLATES[ch]))
        if dist < best_dist:
            best_char, best_dist = ch, dist
    return best_char, best_dist


def recognize(image):
    """Recognise the text in a rendered-text ``image``."""
    cells = segment_characters(image)
    out = []
    for cell in cells:
        if _is_blank(cell):
            out.append(" ")
        else:
            out.append(classify_character(cell)[0])
    return "".join(out).strip()


class TemplateOCR:
    """Template-matching OCR engine for the 5x7 bitmap font."""

    def __init__(self):
        self.characters = CHARACTERS

    def render(self, text, scale=1):
        """Render ``text`` to a binary image (see :func:`render_text`)."""
        return render_text(text, scale=scale)

    def recognize(self, image):
        """Recognise the text in ``image`` (see :func:`recognize`)."""
        return recognize(image)

    def recognize_with_scores(self, image):
        """Recognise text, returning ``(text, [(char, distance), ...])``."""
        cells = segment_characters(image)
        chars, scored = [], []
        for cell in cells:
            if _is_blank(cell):
                chars.append(" ")
                scored.append((" ", 0))
            else:
                ch, dist = classify_character(cell)
                chars.append(ch)
                scored.append((ch, dist))
        return "".join(chars).strip(), scored
