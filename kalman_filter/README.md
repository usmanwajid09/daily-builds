# Kalman Filter (1D)

A 1D Kalman filter for tracking a target moving at **constant velocity**,
implemented from scratch with numpy. The state is `[position, velocity]`;
only the position is measured, and the measurement is noisy.

## How it works

Each step runs a **predict** then an **update**:

- **Predict** — advance the state with the motion model
  `F = [[1, dt], [0, 1]]` and grow the uncertainty by the process-noise
  covariance (discrete white-noise acceleration model).
- **Update** — compare the new measurement with the predicted position,
  compute the Kalman gain from the state and measurement uncertainties,
  and correct the state (and shrink the uncertainty) by the innovation.

Because the gain adapts, the filter trusts measurements more when the
state is uncertain and trusts its own prediction more when the
measurements are noisy — so the filtered track is smoother than the raw
readings.

## Usage

```python
from kalman_filter import KalmanFilter1D, track

# One measurement at a time
kf = KalmanFilter1D(dt=1.0, process_noise=0.1, measurement_noise=1.0)
kf.predict()
kf.update(3.7)
print(kf.position, kf.velocity)

# Or a whole series at once
positions, velocities = track(measurements, dt=1.0, process_noise=0.1,
                              measurement_noise=1.0)
```

Constructor arguments: `dt` (time step, > 0), `process_noise` and
`measurement_noise` (variances, > 0), `initial_position`,
`initial_velocity`, `initial_covariance`. Bad values (non-positive
`dt`/noises, non-finite or non-numeric measurements, empty measurement
lists) raise `ValueError`/`TypeError` with a clear message.

## Run the demo

```bash
python kalman_filter/demo.py
```

Simulates a target moving at 2.0 units/step for 50 steps with noisy
position readings (std 1.0) and prints the filtered track: final
estimated velocity ≈ 2.0 and a filtered position RMSE well below the
raw measurement RMSE.

## Run the tests

```bash
pytest tests/test_kalman_filter.py -q
```
