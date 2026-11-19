"""Decode a simple 1D barcode from a synthetic image (numpy only).

Encoding scheme
---------------
Alphabet: ``"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "`` (37 characters).
Each character maps to a fixed pattern of 6 bars, alternating
black/white starting with black. The pattern is the 6-bit binary form
of the character's index in the alphabet (most-significant bit first):
a ``0`` bit is a narrow bar (1 unit) and a ``1`` bit is a wide bar
(2 units).

Six bars — not five — are used deliberately: 5 bars give only 2**5 = 32
patterns, which cannot cover 37 characters plus two sentinels
(39 distinct patterns). Six bars give 64 patterns, so every character
plus both sentinels has a unique pattern.

Sentinels (also 6 bars, black first):
- start: ``111111`` (all wide) — index 63, outside the alphabet range.
- stop:  ``101010`` — index 42, outside the alphabet range.

Layout (in units): quiet zone (10 white) + start + one 6-bar group
per character + stop + quiet zone (10 white). The stop pattern's last
bar is white, so it merges with the trailing quiet zone into a single
white run of 11 units.

Rendering: each unit is ``unit_width`` pixels wide, the image is
``height`` rows tall, black bars are 0 and white is 255 (uint8).

Decoding: scan the middle row, threshold at 128, run-length encode
the row, infer the unit width from the shortest interior run, convert
each run to 1 or 2 units, then map each 6-bar group back to a
character, validating the start/stop sentinels.
"""

import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# Bars per character/sentinel. Each bit: 0 -> narrow (1 unit), 1 -> wide.
BARS_PER_CHAR = 6
NARROW_UNITS = 1
WIDE_UNITS = 2
QUIET_ZONE_UNITS = 10
THRESHOLD = 128

# Sentinel bit patterns (6 bits each). 63 and 42 are outside the
# alphabet index range 0-36, so neither collides with a character.
_START_BITS = (1, 1, 1, 1, 1, 1)
_STOP_BITS = (1, 0, 1, 0, 1, 0)

_PATTERN_TO_CHAR = {}
for _index, _char in enumerate(ALPHABET):
    _bits = tuple(((_index >> shift) & 1) for shift in range(BARS_PER_CHAR - 1, -1, -1))
    _PATTERN_TO_CHAR[_bits] = _char


def _char_bits(char):
    """Return the 6-bit pattern (tuple of 0/1) for ``char``."""
    index = ALPHABET.index(char)
    return tuple((index >> shift) & 1 for shift in range(BARS_PER_CHAR - 1, -1, -1))


def _bits_to_widths(bits):
    """Convert a bit pattern to bar widths in units (1 = narrow, 2 = wide)."""
    return [WIDE_UNITS if bit else NARROW_UNITS for bit in bits]


def _validate_text(text):
    if not isinstance(text, str):
        raise ValueError(f"text must be a str, got {type(text).__name__}")
    if text == "":
        raise ValueError("text must not be empty")
    for char in text:
        if char not in ALPHABET:
            raise ValueError(
                f"unsupported character {char!r} in text {text!r}; "
                f"allowed characters are {ALPHABET!r}"
            )
    return text


def _validate_render_args(unit_width, height, bar_height):
    if isinstance(unit_width, bool) or not isinstance(unit_width, int):
        raise ValueError(f"unit_width must be an int >= 1, got {unit_width!r}")
    if unit_width < 1:
        raise ValueError(f"unit_width must be >= 1, got {unit_width}")
    if isinstance(height, bool) or not isinstance(height, int):
        raise ValueError(f"height must be an int >= 1, got {height!r}")
    if height < 1:
        raise ValueError(f"height must be >= 1, got {height}")
    if bar_height is not None:
        if isinstance(bar_height, bool) or not isinstance(bar_height, int):
            raise ValueError(f"bar_height must be an int or None, got {bar_height!r}")
        if bar_height < 1 or bar_height > height:
            raise ValueError(
                f"bar_height must lie in [1, height], got {bar_height} "
                f"for height {height}"
            )


def encode(text):
    """Encode ``text`` as a list of run widths in units.

    The runs alternate white/black starting with white (the leading
    quiet zone). The final run is the trailing quiet zone merged with
    the stop pattern's last (white) bar, so it is 11 units wide.
    Raises ValueError for empty text, non-str text, or characters
    outside :data:`ALPHABET`.
    """
    _validate_text(text)
    runs = [QUIET_ZONE_UNITS]  # leading quiet zone (white)
    for bits in [_START_BITS] + [_char_bits(c) for c in text] + [_STOP_BITS]:
        runs.extend(_bits_to_widths(bits))
    # The stop pattern's last bar is white and merges with the trailing
    # quiet zone (also white) into one run.
    runs[-1] += QUIET_ZONE_UNITS
    return runs


def barcode_width(text, unit_width=3):
    """Return the expected rendered width in pixels for ``text``."""
    _validate_render_args(unit_width, 1, None)
    return sum(encode(text)) * unit_width


def render_barcode(text, unit_width=3, height=40, bar_height=None):
    """Render ``text`` as a grayscale barcode image (uint8, values 0/255).

    Returns a 2D numpy array of shape ``(height, width)``. Each unit is
    ``unit_width`` pixels wide. By default the bars span the full
    height; pass ``bar_height`` (<= ``height``) to draw them centred
    in that many rows instead, leaving white above and below.
    """
    _validate_text(text)
    _validate_render_args(unit_width, height, bar_height)
    runs = encode(text)
    width = sum(runs) * unit_width
    image = np.full((height, width), 255, dtype=np.uint8)
    rows = slice(0, height)
    if bar_height is not None:
        top = (height - bar_height) // 2
        rows = slice(top, top + bar_height)
    col = 0
    is_black = False  # first run is the white quiet zone
    for run_units in runs:
        run_px = run_units * unit_width
        if is_black:
            image[rows, col:col + run_px] = 0
        col += run_px
        is_black = not is_black
    return image


def _run_lengths(row):
    """Return ``[(is_black, length_px), ...]`` for a 1D thresholded row."""
    binary = row < THRESHOLD
    runs = []
    current = bool(binary[0])
    length = 1
    for value in binary[1:]:
        value = bool(value)
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = value, 1
    runs.append((current, length))
    return runs


def decode(image):
    """Decode a barcode image (2D array) back to its text.

    Scans the middle row, thresholds at 128, and run-length decodes
    the bars. Raises ValueError on a missing/invalid start or stop
    sentinel, an unknown character pattern, or run lengths that do
    not map to 1 or 2 units.
    """
    if isinstance(image, (str, bytes)):
        raise ValueError(f"image must be a 2D array, got {type(image).__name__}")
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must be a 2D array: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"image must be a 2D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("image must not be empty")
    try:
        row = np.asarray(array[array.shape[0] // 2], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must contain numeric values: {exc}") from exc
    if not np.isfinite(row).all():
        raise ValueError("image must contain only finite values")

    runs = _run_lengths(row)
    if len(runs) < 2:
        raise ValueError("no barcode found: image has no black bars")

    # The unit width is the shortest interior run (every character has
    # at least one narrow bar, and bars are 1 or 2 units wide). The
    # leading/trailing white runs are the quiet zones, so they are
    # excluded from the estimate.
    interior = list(runs)
    if not interior[0][0]:
        interior = interior[1:]
    if interior and not interior[-1][0]:
        interior = interior[:-1]
    if len(interior) < 2 * BARS_PER_CHAR:
        raise ValueError("invalid barcode: too few bars for start/stop sentinels")
    unit_width = min(length for _, length in interior)
    if unit_width < 1:
        raise ValueError("invalid barcode: run lengths must be positive")

    def _to_units(length):
        whole = int(round(length / unit_width))
        if whole < 1 or abs(length / unit_width - whole) > 0.34:
            raise ValueError(
                f"invalid barcode: run of {length} px does not map to "
                f"a whole number of units (unit width {unit_width} px)"
            )
        return whole

    # Strip the quiet zones. The leading white run is pure quiet zone
    # (the barcode starts with a black bar), so it is discarded. The
    # trailing white run is the stop pattern's last (white) bar merged
    # with the trailing quiet zone, so it contributes exactly one
    # 1-unit white bar.
    bar_runs = list(runs)
    if not bar_runs[0][0]:
        if _to_units(bar_runs[0][1]) < 3:
            raise ValueError("invalid barcode: barcode must start with a black bar")
        bar_runs = bar_runs[1:]
    if bar_runs and not bar_runs[-1][0]:
        _to_units(bar_runs[-1][1])  # validates the run maps to whole units
        bar_runs = bar_runs[:-1] + [(False, unit_width)]
    if len(bar_runs) < 2 * BARS_PER_CHAR:
        raise ValueError("invalid barcode: too few bars for start/stop sentinels")
    if bar_runs[0][0] is not True:
        raise ValueError("invalid barcode: first bar must be black")

    units = []
    for is_black, length in bar_runs:
        whole = _to_units(length)
        if whole not in (NARROW_UNITS, WIDE_UNITS):
            raise ValueError(
                f"invalid barcode: run of {length} px does not map to "
                f"1 or 2 units (unit width {unit_width} px)"
            )
        units.append(whole)

    if len(units) % BARS_PER_CHAR != 0:
        raise ValueError(
            f"invalid barcode: {len(units)} bars is not a multiple of "
            f"{BARS_PER_CHAR} bars per character"
        )
    groups = [
        tuple(1 if u == WIDE_UNITS else 0 for u in units[i:i + BARS_PER_CHAR])
        for i in range(0, len(units), BARS_PER_CHAR)
    ]
    if groups[0] != _START_BITS:
        raise ValueError(
            f"invalid barcode: bad start sentinel {groups[0]}, "
            f"expected {_START_BITS}"
        )
    if groups[-1] != _STOP_BITS:
        raise ValueError(
            f"invalid barcode: bad stop sentinel {groups[-1]}, "
            f"expected {_STOP_BITS}"
        )
    chars = []
    for bits in groups[1:-1]:
        char = _PATTERN_TO_CHAR.get(bits)
        if char is None:
            raise ValueError(f"invalid barcode: unknown character pattern {bits}")
        chars.append(char)
    return "".join(chars)


def decode_barcode(image):
    """Alias for :func:`decode`."""
    return decode(image)


def render_and_decode(text, unit_width=3, height=40, bar_height=None):
    """Render ``text`` to an image and decode it back; return the text."""
    return decode(render_barcode(text, unit_width, height, bar_height))
