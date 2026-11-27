"""Demo: HDR merge of a synthetic exposure stack + Reinhard tone map.

Run from the repo root:  python hdr_merge/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hdr_merge import hat_weight, merge_hdr, psnr, reinhard_tone_map


def build_scene(height=48, width=64):
    """Synthetic radiance scene: a dark room, a bright window, a lamp."""
    yy, xx = np.mgrid[0:height, 0:width]
    radiance = np.full((height, width, 3), 0.15)
    radiance[:, :, 0] = 0.12
    radiance[:, :, 1] = 0.15
    radiance[:, :, 2] = 0.18
    radiance[8:24, 8:28] = (0.8, 0.7, 0.5)      # bright window
    disc = (yy - 36) ** 2 + (xx - 48) ** 2 <= 7 ** 2
    radiance[disc] = (3.0, 2.4, 1.2)             # very bright lamp
    radiance[30:40, 6:14] = (0.25, 0.2, 0.3)     # mid-tone object
    return radiance


def expose(radiance, exposure_time):
    """Simulate a camera exposure: radiance * time, clipped to 8-bit."""
    return np.clip(np.round(radiance * exposure_time), 0, 255).astype(np.uint8)


def main():
    scene = build_scene()
    times = [0.25, 1.0, 4.0, 16.0]
    exposures = [expose(scene, t) for t in times]
    hdr = merge_hdr(exposures, times)
    tone_mapped = reinhard_tone_map(hdr)

    print(f"Scene shape: {scene.shape}, radiance range "
          f"[{scene.min():.3f}, {scene.max():.2f}]")
    print(f"Exposures: {len(exposures)} at times {times}")
    print(f"Hat weights at 0 / 127.5 / 255: "
          f"{hat_weight(0.0):.1f} / {hat_weight(127.5):.1f} / "
          f"{hat_weight(255.0):.1f}")
    print(f"HDR radiance range: [{hdr.min():.4f}, {hdr.max():.2f}]")
    print(f"HDR vs true scene (log domain) correlation: "
          f"{np.corrcoef(np.log(hdr.ravel()), np.log(scene.ravel()))[0, 1]:.4f}")
    print(f"Tone-mapped shape: {tone_mapped.shape}, dtype: {tone_mapped.dtype}")
    print(f"Tone-mapped range: [{tone_mapped.min()}, {tone_mapped.max()}]")
    print(f"PSNR tone-mapped vs mid exposure: "
          f"{psnr(exposures[1], tone_mapped):.2f} dB")


if __name__ == "__main__":
    main()
