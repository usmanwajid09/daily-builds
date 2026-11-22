"""Demo: haze a synthetic scene, then dehaze it and measure the recovery."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from dark_channel_dehazing.dehazing import (
    dark_channel,
    dehaze,
    make_clean_scene,
    make_hazy_image,
)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    clean = make_clean_scene(size=64, seed=0)
    true_A = np.array([0.9, 0.9, 0.9])
    hazy = make_hazy_image(clean, transmission=0.55, atmospheric_light=true_A)
    print(f"Clean scene: {clean.shape}, dark-channel mean {dark_channel(clean).mean():.4f}")
    print(f"Hazy scene:  dark-channel mean {dark_channel(hazy).mean():.4f} "
          f"(haze lifts the dark channel)")

    recovered, info = dehaze(hazy, omega=0.95, patch_size=15, t0=0.1)
    print(f"Estimated atmospheric light: {np.round(info['atmospheric_light'], 3)} "
          f"(true {true_A})")
    print(f"Mean transmission: {info['mean_transmission']:.3f} (true 0.55)")
    print(f"MSE hazy vs clean:     {mse(hazy, clean):.5f}")
    print(f"MSE recovered vs clean: {mse(recovered, clean):.5f}")
    print(f"Recovery improves MSE: {mse(recovered, clean) < mse(hazy, clean)}")


if __name__ == "__main__":
    main()
