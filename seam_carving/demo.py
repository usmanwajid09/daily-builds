"""Demo: content-aware width reduction with seam carving.

Builds a synthetic image — a smooth (low-energy) background with a bright,
sharp-edged rectangle (high-energy object) — then removes 10 vertical seams
and shows that the width shrinks while the bright object survives.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from seam_carving.seam_carving import (
    energy_map,
    find_vertical_seam,
    remove_vertical_seams,
)


def build_demo_image(height=40, width=60):
    """Smooth background (value 50) with a bright rectangle (value 200)."""
    img = np.full((height, width), 50.0)
    # Bright object: rows 10..30, columns 25..35 (sharp edges => high energy).
    img[10:30, 25:35] = 200.0
    return img


def main():
    image = build_demo_image()
    n_seams = 10

    # Energy of the seams that will be removed (computed one at a time,
    # mirroring what remove_vertical_seams does).
    removed_seam_energies = []
    work = image.copy()
    for _ in range(n_seams):
        energy = energy_map(work)
        seam = find_vertical_seam(energy)
        removed_seam_energies.append(float(energy[np.arange(work.shape[0]), seam].mean()))
        work = remove_vertical_seams(work, 1)

    result = work
    full_energy = energy_map(image)

    # Object-preservation check: the bright rectangle should still be
    # present at full brightness somewhere in the result.
    obj_max_before = float(image[10:30, 25:35].max())
    obj_max_after = float(result.max())
    bright_cols_after = int(np.sum(result.max(axis=0) == 200.0))

    print("Seam carving demo")
    print(f"Original shape:        {image.shape}")
    print(f"New shape:             {result.shape}")
    print(f"Seams removed:         {n_seams}")
    print(f"Width reduced by:      {image.shape[1] - result.shape[1]} columns")
    print(f"Height unchanged:      {image.shape[0] == result.shape[0]}")
    print(f"Mean image energy:     {full_energy.mean():.4f}")
    print(f"Mean removed-seam energy: {np.mean(removed_seam_energies):.4f}")
    print(f"Removed seams are low-energy: "
          f"{np.mean(removed_seam_energies) < full_energy.mean()}")
    print(f"Object max before:     {obj_max_before:.0f}")
    print(f"Object max after:      {obj_max_after:.0f}")
    print(f"Bright columns remaining: {bright_cols_after} "
          f"(object was 10 columns wide)")
    print(f"Object preserved:      {obj_max_after == obj_max_before}")


if __name__ == "__main__":
    main()
