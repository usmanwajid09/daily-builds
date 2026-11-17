import numpy as np
import pytest

from image_similarity.image_similarity import (
    average_hash,
    difference_hash,
    find_most_similar,
    hamming_distance,
)


def _gradient(shape=(32, 32)):
    return np.tile(np.linspace(0, 255, shape[1]), (shape[0], 1))


def test_identical_images_have_zero_distance():
    img = _gradient()
    assert hamming_distance(average_hash(img), average_hash(img)) == 0
    assert hamming_distance(difference_hash(img), difference_hash(img)) == 0


def test_near_duplicate_is_closer_than_noise():
    rng = np.random.default_rng(7)
    base = _gradient()
    near = np.clip(base + rng.normal(0, 3, base.shape), 0, 255)
    noise = rng.uniform(0, 255, base.shape)
    assert hamming_distance(average_hash(base), average_hash(near)) < 5
    assert hamming_distance(average_hash(base), average_hash(noise)) > 10


def test_solid_image_hash_is_zero():
    solid = np.full((16, 16), 128.0)
    assert average_hash(solid) == 0  # every pixel equals the mean
    assert difference_hash(solid) == 0  # no pixel is brighter than its neighbour


def test_average_hash_known_pattern():
    # Left half dark, right half bright: bits are 0 then 1 in each row.
    img = np.zeros((8, 8))
    img[:, 4:] = 255.0
    assert average_hash(img) == int("00001111" * 8, 2)


def test_difference_hash_known_pattern():
    # Strictly increasing left to right: every step is a rise, so all bits 0.
    # Use a 32-wide image so the 8x9 resize never duplicates a column.
    img = _gradient((8, 32))
    assert difference_hash(img) == 0
    # Reversed gradient: every step is a fall, so all bits 1.
    assert difference_hash(img[:, ::-1]) == (1 << 64) - 1


def test_rgb_and_grayscale_agree():
    gray = _gradient()
    rgb = np.stack([gray, gray, gray], axis=2)
    assert average_hash(rgb) == average_hash(gray)
    assert difference_hash(rgb) == difference_hash(gray)


def test_hash_is_resolution_invariant():
    small = _gradient((8, 8))
    big = _gradient((64, 64))
    assert average_hash(small) == average_hash(big)
    assert difference_hash(small) == difference_hash(big)


def test_hash_size_changes_bit_length():
    img = _gradient()
    assert average_hash(img, hash_size=4) < (1 << 16)
    assert difference_hash(img, hash_size=4) < (1 << 20)  # 4 x 5 bits


def test_hamming_distance_known_values():
    assert hamming_distance(0b1010, 0b0101) == 4
    assert hamming_distance(0b1111, 0b1100) == 2
    assert hamming_distance(12345, 12345) == 0


def test_find_most_similar_picks_near_duplicate():
    rng = np.random.default_rng(3)
    base = _gradient()
    near = np.clip(base + rng.normal(0, 2, base.shape), 0, 255)
    noise = rng.uniform(0, 255, base.shape)
    solid = np.full((32, 32), 200.0)
    index, distance = find_most_similar(base, [noise, solid, near])
    assert index == 2
    assert distance < 5
    # For dHash, a solid image ties with the near-duplicate at distance 0
    # (both hashes are all zeros), so check the two-way race without it.
    index, distance = find_most_similar(base, [noise, near], method="difference")
    assert index == 1
    assert distance == 0


def test_find_most_similar_single_candidate():
    img = _gradient()
    assert find_most_similar(img, [img]) == (0, 0)


def test_find_most_similar_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        find_most_similar(_gradient(), [])


def test_find_most_similar_bad_method_raises():
    with pytest.raises(ValueError, match="method"):
        find_most_similar(_gradient(), [_gradient()], method="bogus")


def test_invalid_hash_size_raises():
    with pytest.raises(ValueError, match="hash_size"):
        average_hash(_gradient(), hash_size=1)
    with pytest.raises(ValueError, match="hash_size"):
        difference_hash(_gradient(), hash_size=0)


def test_invalid_image_dimensions_raise():
    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        average_hash(np.zeros(8))  # 1D
    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        average_hash(np.zeros((4, 4, 3, 2)))  # 4D


def test_invalid_channel_count_raises():
    with pytest.raises(ValueError, match="1 or 3 channels"):
        average_hash(np.zeros((4, 4, 2)))


def test_empty_image_raises():
    with pytest.raises(ValueError, match="empty"):
        average_hash(np.zeros((0, 0)))


def test_nan_image_raises():
    img = _gradient()
    img[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        average_hash(img)


def test_hamming_distance_rejects_bad_hashes():
    with pytest.raises(ValueError, match="non-negative integer"):
        hamming_distance(-1, 5)
    with pytest.raises(ValueError, match="non-negative integer"):
        hamming_distance(1.5, 5)


def test_find_most_similar_propagates_bad_candidate():
    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        find_most_similar(_gradient(), [_gradient(), np.zeros(4)])
