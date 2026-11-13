"""Demo: median-cut quantization of a synthetic colourful image.

Builds a 64x64 RGB image with smooth red/green gradients, a blue disc
and a yellow rectangle, quantizes it to 8 colours, and prints the
palette size and the unique colour count before and after.
"""

import numpy as np

from median_cut_quantizer.quantize import quantize, unique_color_count


def make_colourful_image(height=64, width=64):
    """Return a synthetic RGB image: gradients + disc + rectangle."""
    rows = np.linspace(0, 255, height)[:, None]
    cols = np.linspace(0, 255, width)[None, :]
    r = np.broadcast_to(cols, (height, width))
    g = np.broadcast_to(rows, (height, width))
    b = np.full((height, width), 60.0)
    image = np.stack([r, g, b], axis=2)
    rr, cc = np.mgrid[0:height, 0:width]
    disc = (rr - 20) ** 2 + (cc - 45) ** 2 <= 10 ** 2
    image[disc] = [30.0, 30.0, 230.0]
    image[40:55, 8:24] = [230.0, 220.0, 30.0]
    return image.astype(np.uint8)


def main():
    image = make_colourful_image()
    n_before = unique_color_count(image)
    quantized, palette = quantize(image, n_colors=8)
    n_after = unique_color_count(quantized)
    print(f"image: {image.shape[0]}x{image.shape[1]} RGB")
    print(f"unique colours before: {n_before}")
    print(f"unique colours after: {n_after} (palette size: {len(palette)})")
    print("palette (RGB rows):")
    for row in palette:
        print(f"  {row.tolist()}")


if __name__ == "__main__":
    main()
