"""Demo: active contour (snake) segmentation on a synthetic disc.

Builds a 64x64 grayscale image — a bright filled disc (radius 15) on a
dark background — initialises a circle of radius 25 around the same
centre, runs the snake, and prints the initial/final mean radius, the
true radius, the error, and the total energy before and after fitting.
The snake should shrink onto the disc boundary: final mean radius
close to 15 and total energy lower than at the start.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from active_contour_snake.active_contour_snake import (
    ActiveContour,
    _mean_radius,
    contour_length,
    enclosed_area,
)


def make_disc_image(height=64, width=64, center=(32.0, 32.0), radius=15.0,
                    foreground=255.0, background=0.0):
    """Return a synthetic image: a bright filled disc on a dark background.

    ``center`` is ``(x, y)``; the disc includes pixels whose centre lies
    within ``radius`` of it.
    """
    image = np.full((height, width), background, dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    image[(rows - center[1]) ** 2 + (cols - center[0]) ** 2 <= radius ** 2] \
        = foreground
    return image


def main():
    height, width = 64, 64
    center = (32.0, 32.0)
    true_radius = 15.0
    init_radius = 25.0

    image = make_disc_image(height, width, center, true_radius)
    snake = ActiveContour(alpha=0.5, beta=0.2, gamma=1.0,
                          n_iterations=100, n_points=50)
    initial = snake.initialise_circle(center, init_radius,
                                      image_shape=image.shape)
    energy_before = snake.energy(image, initial)
    initial_radius = _mean_radius(initial, center)

    final = snake.fit(image, initial)
    energy_after = snake.energy(image, final)
    final_radius = _mean_radius(final, center)
    error = abs(final_radius - true_radius)

    print("Active contour (snake) segmentation — synthetic disc")
    print(f"Image: {width}x{height}, disc centre {center}, "
          f"true radius {true_radius:.1f} px")
    print(f"Initial mean radius: {initial_radius:.2f} px")
    print(f"Final mean radius:   {final_radius:.2f} px")
    print(f"True radius:         {true_radius:.2f} px")
    print(f"Absolute error:      {error:.2f} px")
    print(f"Energy before fit:   {energy_before:.2f}")
    print(f"Energy after fit:    {energy_after:.2f}")
    print(f"Energy decreased:    {energy_after < energy_before}")
    print(f"Final contour length: {contour_length(final):.2f} px "
          f"(2*pi*r = {2 * np.pi * true_radius:.2f})")
    print(f"Final enclosed area:  {enclosed_area(final):.2f} px^2 "
          f"(pi*r^2 = {np.pi * true_radius ** 2:.2f})")


if __name__ == "__main__":
    main()
