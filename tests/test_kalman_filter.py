import numpy as np
import pytest

from kalman_filter import KalmanFilter1D, track


def _noisy_track(seed=42, n_steps=50, noise=1.0, velocity=2.0):
    rng = np.random.default_rng(seed)
    truth = velocity * np.arange(n_steps, dtype=float)
    measurements = truth + rng.normal(0.0, noise, n_steps)
    positions, velocities = track(
        measurements,
        dt=1.0,
        process_noise=0.1,
        measurement_noise=noise**2,
    )
    return truth, measurements, positions, velocities


def test_track_returns_arrays_matching_measurements():
    positions, velocities = track([0.0, 2.1, 3.9, 6.2])
    assert len(positions) == 4 and len(velocities) == 4


def test_filter_beats_raw_noise_rmse():
    truth, measurements, positions, _ = _noisy_track()
    rmse_raw = float(np.sqrt(np.mean((measurements - truth) ** 2)))
    rmse_filtered = float(np.sqrt(np.mean((positions - truth) ** 2)))
    assert rmse_filtered < rmse_raw
    assert rmse_filtered < 0.6  # headline number from the demo


def test_final_velocity_close_to_true():
    _, _, _, velocities = _noisy_track()
    assert abs(velocities[-1] - 2.0) < 0.5


def test_final_position_close_to_truth():
    truth, _, positions, _ = _noisy_track()
    assert abs(positions[-1] - truth[-1]) < 2.0


def test_step_returns_position_and_velocity():
    kf = KalmanFilter1D()
    pos, vel = kf.step(1.0)
    assert isinstance(pos, float) and isinstance(vel, float)
    assert kf.n_steps_ == 1


def test_perfect_measurements_tracked_exactly():
    # Noiseless measurements of a target moving at 1.0/step.
    positions, velocities = track(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        process_noise=1e-6,
        measurement_noise=1e-6,
        initial_covariance=1e-6,
    )
    assert abs(positions[-1] - 4.0) < 0.1
    assert abs(velocities[-1] - 1.0) < 0.2


def test_predict_moves_position_by_velocity():
    kf = KalmanFilter1D(dt=0.5, initial_position=10.0, initial_velocity=4.0)
    kf.predict()
    assert kf.position == pytest.approx(12.0)
    assert kf.velocity == pytest.approx(4.0)


def test_empty_measurements_raise():
    with pytest.raises(ValueError, match="must not be empty"):
        track([])


def test_string_measurements_raise():
    with pytest.raises(TypeError):
        track("not a sequence")


def test_non_numeric_measurement_raises():
    with pytest.raises(TypeError):
        KalmanFilter1D().update("hello")


def test_nan_measurement_raises():
    with pytest.raises(ValueError, match="finite"):
        KalmanFilter1D().update(float("nan"))


def test_vector_measurement_raises():
    with pytest.raises(ValueError, match="single number"):
        KalmanFilter1D().update([1.0, 2.0])


def test_invalid_dt_raises():
    with pytest.raises(ValueError, match="dt"):
        KalmanFilter1D(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        KalmanFilter1D(dt=-1.0)


def test_invalid_noises_raise():
    with pytest.raises(ValueError, match="process_noise"):
        KalmanFilter1D(process_noise=0.0)
    with pytest.raises(ValueError, match="measurement_noise"):
        KalmanFilter1D(measurement_noise=-2.0)


def test_covariance_stays_symmetric_and_finite():
    kf = KalmanFilter1D()
    for z in [0.5, 1.5, 2.5, 3.5, 4.5]:
        kf.step(z)
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.isfinite(kf.P))


def test_deterministic_repeat_run():
    _, _, pos_a, _ = _noisy_track()
    _, _, pos_b, _ = _noisy_track()
    assert np.allclose(pos_a, pos_b)
