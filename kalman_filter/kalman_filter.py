"""1D Kalman filter for tracking a target moving at constant velocity.

State vector: ``[position, velocity]``.

Motion model (discrete, step ``dt``)::

    x_{k+1} = F x_k + w,   F = [[1, dt], [0, 1]]

Measurement model (only the position is observed)::

    z_k = H x_k + v,       H = [1, 0]

``w`` and ``v`` are zero-mean Gaussian noise with variances set by
``process_noise`` and ``measurement_noise``. The process-noise covariance
uses the standard discrete white-noise acceleration model.
"""

import numpy as np

__all__ = ["KalmanFilter1D", "track"]


def _positive_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__!r}")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def _finite_number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__!r}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


class KalmanFilter1D:
    """Constant-velocity Kalman filter for one-dimensional tracking."""

    def __init__(
        self,
        dt=1.0,
        process_noise=1.0,
        measurement_noise=1.0,
        initial_position=0.0,
        initial_velocity=0.0,
        initial_covariance=1.0,
    ):
        self.dt = _positive_finite(dt, "dt")
        self.process_noise = _positive_finite(process_noise, "process_noise")
        self.measurement_noise = _positive_finite(
            measurement_noise, "measurement_noise"
        )
        self.initial_covariance = _positive_finite(
            initial_covariance, "initial_covariance"
        )
        self.x = np.array(
            [
                _finite_number(initial_position, "initial_position"),
                _finite_number(initial_velocity, "initial_velocity"),
            ],
            dtype=float,
        )
        # State transition and observation matrices.
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        # Discrete white-noise acceleration covariance.
        dt = self.dt
        q = self.process_noise
        self.Q = q * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
        )
        self.R = np.array([[self.measurement_noise]])
        self.P = self.initial_covariance * np.eye(2)
        self.n_steps_ = 0

    @property
    def position(self):
        """Current estimated position."""
        return float(self.x[0])

    @property
    def velocity(self):
        """Current estimated velocity."""
        return float(self.x[1])

    def predict(self):
        """Advance the state estimate one step (no measurement)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement):
        """Correct the state estimate with one position measurement.

        Returns the Kalman gain used for this update.
        """
        z = self._validate_measurement(measurement)
        y = z - float((self.H @ self.x)[0])  # innovation
        S = float((self.H @ self.P @ self.H.T + self.R)[0, 0])
        K = (self.P @ self.H.T) / S  # shape (2, 1)
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.n_steps_ += 1
        return K.flatten().copy()

    def step(self, measurement):
        """One predict + update cycle; returns (position, velocity)."""
        self.predict()
        self.update(measurement)
        return self.position, self.velocity

    @staticmethod
    def _validate_measurement(measurement):
        if isinstance(measurement, bool):
            raise TypeError("measurement must be a real number, got bool")
        if isinstance(measurement, (list, tuple, np.ndarray)):
            arr = np.asarray(measurement, dtype=float)
            if arr.size != 1:
                raise ValueError(
                    f"measurement must be a single number, got {arr.size} values"
                )
            value = float(arr.reshape(-1)[0])
        else:
            try:
                value = float(measurement)
            except (TypeError, ValueError):
                raise TypeError(
                    "measurement must be a real number, "
                    f"got {type(measurement).__name__!r}"
                )
        if not np.isfinite(value):
            raise ValueError(f"measurement must be finite, got {value!r}")
        return value


def track(measurements, **kwargs):
    """Run a Kalman filter over a sequence of position measurements.

    Returns ``(positions, velocities)`` as numpy arrays of the same length
    as ``measurements``.
    """
    if isinstance(measurements, (str, bytes)):
        raise TypeError("measurements must be a sequence of numbers, got a string")
    try:
        measurements = list(measurements)
    except TypeError:
        raise TypeError("measurements must be a sequence of numbers")
    if len(measurements) == 0:
        raise ValueError("measurements must not be empty")
    kf = KalmanFilter1D(**kwargs)
    positions = np.empty(len(measurements))
    velocities = np.empty(len(measurements))
    for i, z in enumerate(measurements):
        pos, vel = kf.step(z)
        positions[i] = pos
        velocities[i] = vel
    return positions, velocities
