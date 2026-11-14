import numpy as np
import pytest

from blob_detection.blobs import detect_blobs, log_kernel, log_response
from blob_detection.demo import DISCS, make_test_image


def _disc_image(height=60, width=60, centre=(30, 30), radius=8, amplitude=1.0):
    """One bright disc of ``radius`` on a black background."""
    image = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.mgrid[0:height, 0:width]
    image[(rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2] = amplitude
    return image


def _gaussian_blob(height=60, width=60, centre=(30, 30), sigma=4.0, amplitude=1.0):
    """A smooth Gaussian blob (no hard edge)."""
    rows, cols = np.mgrid[0:height, 0:width]
    r2 = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma ** 2))


def test_single_disc_detected_at_centre():
    blobs = detect_blobs(_disc_image(radius=8))
    assert len(blobs) == 1
    row, col, radius, response = blobs[0]
    assert (row, col) == (30, 30)
    # Best-matching scale is sigma = radius / sqrt(2) ~= 5.66, so the
    # sampled scale 5.5 (radius 7.78) should win.
    assert radius == pytest.approx(7.78, abs=0.01)
    assert response > 0.5


def test_demo_image_finds_all_three_discs():
    blobs = detect_blobs(make_test_image())
    assert len(blobs) == 3
    found = sorted((row, col) for row, col, _r, _resp in blobs)
    assert found == sorted((row, col) for row, col, _r in DISCS)
    found_radii = sorted(radius for _r, _c, radius, _resp in blobs)
    true_radii = sorted(radius for _r, _c, radius in DISCS)
    for found_radius, true_radius in zip(found_radii, true_radii):
        # Sampled scales quantise the radius; allow one scale step.
        assert abs(found_radius - true_radius) <= 1.5


def test_gaussian_blob_radius_matches_sigma():
    # A Gaussian blob of sigma s responds maximally at LoG scale s,
    # so the reported radius should be sqrt(2) * s.
    blobs = detect_blobs(
        _gaussian_blob(sigma=4.0), min_sigma=1.0, max_sigma=8.0, num_sigma=8
    )
    assert len(blobs) == 1
    row, col, radius, _response = blobs[0]
    assert (row, col) == (30, 30)
    assert radius == pytest.approx(np.sqrt(2) * 4.0, abs=0.6)


def test_constant_image_has_no_blobs():
    assert detect_blobs(np.full((20, 20), 128.0)) == []
    assert detect_blobs(np.zeros((20, 20))) == []
    assert detect_blobs(np.ones((20, 20))) == []


def test_threshold_filters_weak_blobs():
    image = _disc_image(radius=8, amplitude=1.0)
    assert len(detect_blobs(image, threshold=0.1)) == 1
    # The disc's peak response is ~0.73, so a threshold above it
    # removes it entirely.
    assert detect_blobs(image, threshold=1.0) == []


def test_blobs_sorted_by_response_descending():
    blobs = detect_blobs(make_test_image())
    responses = [response for _r, _c, _rad, response in blobs]
    assert responses == sorted(responses, reverse=True)


def test_output_types_and_radius_convention():
    blobs = detect_blobs(_disc_image(radius=8))
    row, col, radius, response = blobs[0]
    assert isinstance(row, int) and isinstance(col, int)
    assert isinstance(radius, float) and isinstance(response, float)
    # radius = sqrt(2) * sigma for one of the sampled scales.
    sigmas = np.linspace(1.0, 10.0, 5)
    assert any(radius == pytest.approx(np.sqrt(2) * s) for s in sigmas)


def test_num_sigma_one_uses_min_sigma():
    blobs = detect_blobs(_disc_image(radius=4), min_sigma=3.0, max_sigma=3.0,
                         num_sigma=1)
    assert len(blobs) == 1
    assert blobs[0][2] == pytest.approx(np.sqrt(2) * 3.0)


def test_overlap_pruning_removes_rim_blobs():
    # Without pruning, the sharp rim of a large disc fires small-scale
    # blobs; with the default overlap pruning only the disc survives.
    image = _disc_image(height=80, width=80, centre=(40, 40), radius=14)
    assert len(detect_blobs(image, overlap=1.0)) > 1
    assert len(detect_blobs(image)) == 1


def test_log_kernel_shape_and_zero_sum():
    kernel = log_kernel(2.0)
    assert kernel.shape == (13, 13)  # 2 * ceil(3 * 2) + 1
    assert kernel.sum() == pytest.approx(0.0, abs=1e-12)
    # The LoG centre is a negative peak (bright blob -> positive
    # response only after the minus sign in log_response).
    assert kernel[6, 6] < 0
    assert kernel[6, 6] == kernel.min()


def test_log_kernel_explicit_size():
    kernel = log_kernel(1.5, kernel_size=9)
    assert kernel.shape == (9, 9)


def test_log_response_constant_image_is_zero():
    response = log_response(np.full((10, 10), 77.0), sigma=2.0)
    assert response.shape == (10, 10)
    assert np.abs(response).max() < 1e-10


def test_log_response_peaks_at_blob_centre():
    response = log_response(_gaussian_blob(sigma=4.0), sigma=4.0)
    assert response[30, 30] == response.max()
    # Matched-scale response of a unit-amplitude Gaussian blob is
    # ~0.5 (slightly less for the truncated discrete kernel).
    assert response[30, 30] > 0.4


def test_deterministic():
    image = make_test_image()
    assert detect_blobs(image) == detect_blobs(image)


def test_invalid_images_raise():
    with pytest.raises(ValueError):
        detect_blobs(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        detect_blobs(np.zeros((5, 5, 3)))  # RGB not supported
    with pytest.raises(ValueError):
        detect_blobs(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        detect_blobs(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        detect_blobs(np.full((5, 5), np.inf))  # non-finite
    with pytest.raises(ValueError):
        detect_blobs(np.array([["a", "b"]]))  # non-numeric


def test_invalid_parameters_raise():
    image = np.zeros((5, 5))
    with pytest.raises(ValueError):
        detect_blobs(image, min_sigma=0)
    with pytest.raises(ValueError):
        detect_blobs(image, min_sigma=-1.0)
    with pytest.raises(ValueError):
        detect_blobs(image, min_sigma=5.0, max_sigma=2.0)
    with pytest.raises(ValueError):
        detect_blobs(image, num_sigma=0)
    with pytest.raises(ValueError):
        detect_blobs(image, num_sigma=2.5)
    with pytest.raises(ValueError):
        detect_blobs(image, threshold=-0.1)
    with pytest.raises(ValueError):
        detect_blobs(image, overlap=0)
    with pytest.raises(ValueError):
        detect_blobs(image, overlap=1.5)
    with pytest.raises(ValueError):
        log_kernel(0)
    with pytest.raises(ValueError):
        log_kernel(1.0, kernel_size=4)  # even size
    with pytest.raises(ValueError):
        log_response(image, sigma=-2.0)
