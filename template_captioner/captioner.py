"""Template-based image captioning from scratch (numpy only).

The captioner does not use a learned model. It measures simple, exact
scene attributes — dominant colours, number of shapes, brightness,
contrast and the layout of the largest shape — and composes a
natural-language caption from templates. Everything is deterministic and
fully testable on synthetic images.
"""

import numpy as np

__all__ = [
    "analyze_image",
    "caption",
    "count_shapes",
    "dominant_colours",
    "largest_shape_quadrant",
    "make_test_image",
]

# Named colours as RGB centres in [0, 1].
_PALETTE = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "grey": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.55, 0.0),
    "purple": (0.6, 0.0, 0.8),
    "cyan": (0.0, 0.8, 0.8),
    "brown": (0.55, 0.3, 0.1),
    "pink": (1.0, 0.6, 0.8),
}


def _to_float_rgb(image):
    """Normalise an image to float RGB in [0, 1]."""
    image = np.asarray(image)
    if image.ndim == 2:  # grayscale -> RGB
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be grayscale (H, W) or RGB (H, W, 3), got shape {image.shape}."
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}.")
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(
                "float images must have values in [0, 1]; "
                f"got range [{image.min():.3f}, {image.max():.3f}]. "
                "Use uint8 for 0-255 images."
            )
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values.")
    return image


def _to_grey(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def dominant_colours(image, min_fraction=0.05):
    """Named colours covering at least ``min_fraction`` of the pixels.

    Returns a list of (name, fraction) sorted by fraction, descending.
    """
    rgb = _to_float_rgb(image)
    if not 0 < min_fraction <= 1:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction!r}.")
    pixels = rgb.reshape(-1, 3)
    names = list(_PALETTE)
    centres = np.array([_PALETTE[n] for n in names])
    # Distance from every pixel to every palette centre.
    dists = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    counts = np.bincount(nearest, minlength=len(names)).astype(np.float64)
    fractions = counts / len(pixels)
    result = [(names[i], float(fractions[i]))
              for i in range(len(names)) if fractions[i] >= min_fraction]
    return sorted(result, key=lambda item: -item[1])


def _foreground_mask(rgb):
    """Binary mask of pixels that differ clearly from the median background.

    Uses the largest per-channel difference so that bright colours on a
    bright background (e.g. yellow on white) are still foreground.
    """
    background = np.median(rgb, axis=(0, 1))
    return np.max(np.abs(rgb - background), axis=2) > 0.25


def _connected_components(mask, min_size=4):
    """Label connected (4-neighbour) components; returns list of pixel lists."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) >= min_size:
                components.append(pixels)
    return components


def count_shapes(image, min_size=4):
    """Count distinct shapes (connected foreground regions).

    Returns (count, sizes) where sizes is a list of pixel counts per shape,
    largest first.
    """
    if not isinstance(min_size, int) or min_size <= 0:
        raise ValueError(f"min_size must be a positive int, got {min_size!r}.")
    rgb = _to_float_rgb(image)
    grey = _to_grey(rgb)
    components = _connected_components(_foreground_mask(rgb), min_size=min_size)
    sizes = sorted((len(p) for p in components), reverse=True)
    return len(sizes), sizes


def largest_shape_quadrant(image):
    """Quadrant of the largest shape's centroid, or 'centre' / None.

    Returns None when the image has no shapes.
    """
    rgb = _to_float_rgb(image)
    grey = _to_grey(rgb)
    components = _connected_components(_foreground_mask(rgb))
    if not components:
        return None
    largest = max(components, key=len)
    ys = np.array([p[0] for p in largest])
    xs = np.array([p[1] for p in largest])
    cy, cx = ys.mean(), xs.mean()
    h, w = grey.shape
    vertical = "upper" if cy < h / 2 else "lower"
    horizontal = "left" if cx < w / 2 else "right"
    if abs(cy - h / 2) < h * 0.1 and abs(cx - w / 2) < w * 0.1:
        return "centre"
    return f"{vertical} {horizontal}"


def analyze_image(image):
    """Measure all scene attributes; returns a dict."""
    rgb = _to_float_rgb(image)
    grey = _to_grey(rgb)
    colours = dominant_colours(rgb)
    n_shapes, sizes = count_shapes(rgb)
    brightness = float(grey.mean())
    contrast = float(grey.std())
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    quadrant = largest_shape_quadrant(rgb)
    background = "dark" if float(np.median(grey)) < 0.5 else "light"
    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "background": background,
        "dominant_colours": colours,
        "n_shapes": n_shapes,
        "shape_sizes": sizes,
        "largest_shape_quadrant": quadrant,
        "is_colourful": saturation > 0.2,
        "brightness_word": ("bright" if brightness > 0.66 else
                            "dark" if brightness < 0.33 else "moderately bright"),
    }


def caption(image):
    """Compose a natural-language caption for the image."""
    info = analyze_image(image)
    parts = []
    if info["n_shapes"] == 0:
        colour_word = info["dominant_colours"][0][0] if info["dominant_colours"] else "plain"
        return (f"A plain {info['brightness_word']} {colour_word} image "
                f"with no distinct shapes.")
    shape_word = "shape" if info["n_shapes"] == 1 else "shapes"
    first = (f"A {info['brightness_word']} image with {info['n_shapes']} "
             f"{shape_word} on a {info['background']} background.")
    parts.append(first)
    if info["dominant_colours"]:
        names = [name for name, _ in info["dominant_colours"][:3]]
        parts.append(f"Dominant colours: {', '.join(names)}.")
    if info["largest_shape_quadrant"]:
        parts.append(
            f"The largest shape is in the {info['largest_shape_quadrant']} "
            f"of the image.")
    if info["is_colourful"]:
        parts.append("The scene is colourful.")
    elif info["contrast"] < 0.05:
        parts.append("The scene has very low contrast.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------
def make_test_image(size=32, shapes=(), background=0.0):
    """Build a synthetic RGB scene.

    ``shapes`` is a list of (kind, colour, centre, size) tuples where kind
    is 'circle' or 'square' and colour is an RGB triple in [0, 1].
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if not 0.0 <= background <= 1.0:
        raise ValueError(f"background must be in [0, 1], got {background}.")
    img = np.full((size, size, 3), background, dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    for kind, colour, centre, shape_size in shapes:
        colour = np.asarray(colour, dtype=np.float64)
        if colour.shape != (3,):
            raise ValueError(f"colour must be an RGB triple, got {colour!r}.")
        if kind == "circle":
            mask = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= shape_size ** 2
        elif kind == "square":
            mask = ((np.abs(yy - centre[0]) <= shape_size)
                    & (np.abs(xx - centre[1]) <= shape_size))
        else:
            raise ValueError(f"unknown shape kind {kind!r}; expected circle/square.")
        img[mask] = colour
    return img
