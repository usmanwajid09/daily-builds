"""Track a target moving at constant velocity through noisy measurements.

Simulates a target starting at position 0 moving at 2.0 units/step, adds
Gaussian measurement noise (std 1.0), and runs the 1D Kalman filter over
the noisy readings. Prints the true vs. filtered position at a few steps
and the final tracking error (RMSE of the filter vs. the truth).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kalman_filter import KalmanFilter1D, track

TRUE_VELOCITY = 2.0
N_STEPS = 50
MEASUREMENT_NOISE = 1.0
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    true_positions = TRUE_VELOCITY * np.arange(N_STEPS, dtype=float)
    measurements = true_positions + rng.normal(0.0, MEASUREMENT_NOISE, N_STEPS)
    positions, velocities = track(
        measurements,
        dt=1.0,
        process_noise=0.1,
        measurement_noise=MEASUREMENT_NOISE**2,
        initial_position=0.0,
        initial_velocity=0.0,
    )
    rmse_raw = float(np.sqrt(np.mean((measurements - true_positions) ** 2)))
    rmse_filtered = float(np.sqrt(np.mean((positions - true_positions) ** 2)))
    print(f"Target: constant velocity {TRUE_VELOCITY}, {N_STEPS} steps, seed {SEED}")
    print(f"Measurement noise std: {MEASUREMENT_NOISE}")
    print(f"{'step':>4} {'true':>8} {'measured':>9} {'filtered':>9}")
    for i in (0, 9, 24, 49):
        print(
            f"{i:>4} {true_positions[i]:>8.2f} {measurements[i]:>9.2f} "
            f"{positions[i]:>9.2f}"
        )
    print(f"\nFinal estimated velocity: {velocities[-1]:.3f} (true {TRUE_VELOCITY})")
    print(f"Raw measurement RMSE:   {rmse_raw:.3f}")
    print(f"Filtered position RMSE: {rmse_filtered:.3f}")
    print(f"Filter beats raw noise by {100 * (1 - rmse_filtered / rmse_raw):.1f}%")


if __name__ == "__main__":
    main()
