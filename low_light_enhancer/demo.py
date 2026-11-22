"""Demo: enhance a synthetic dark scene and measure the improvement."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from low_light_enhancer.enhancer import enhance, make_dark_image


def main():
    dark = make_dark_image(size=64, seed=0)
    enhanced, before, after, gamma = enhance(dark, target_mean=0.5)
    print(f"Dark scene: {dark.shape}")
    print(f"Adaptive gamma used: {gamma:.3f} (< 1 brightens)")
    print(f"Brightness: {before['brightness']:.3f} -> {after['brightness']:.3f}")
    print(f"Contrast:   {before['contrast']:.3f} -> {after['contrast']:.3f}")
    print(f"Dynamic range: {before['dynamic_range']:.3f} -> {after['dynamic_range']:.3f}")
    print(f"Brightness improved: {after['brightness'] > before['brightness']}")
    print(f"Contrast improved:   {after['contrast'] > before['contrast']}")
    print(f"Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}] (stays in [0, 1])")


if __name__ == "__main__":
    main()
