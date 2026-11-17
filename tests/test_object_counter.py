"""Tests for object_counter (connected-components labelling, numpy only)."""

import numpy as np
import pytest

from object_counter.object_counter import count_objects, label_image


def test_empty_image_has_no_objects():
    img = np.zeros((5, 5), dtype=np.uint8)
    assert count_objects(img) == []
    assert label_image(img).max() == 0


def test_single_pixel_is_one_object():
    img = np.zeros((4, 4), dtype=np.uint8)
    img[2, 3] = 1
    objects = count_objects(img)
    assert len(objects) == 1
    obj = objects[0]
    assert obj["area"] == 1
    assert obj["centroid"] == (2.0, 3.0)
    assert obj["bbox"] == (2, 3, 2, 3)


def test_two_separate_blobs_counted():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 1   # 2x2 blob, area 4
    img[6:9, 6:9] = 1   # 3x3 blob, area 9
    objects = count_objects(img)
    assert len(objects) == 2
    assert [o["area"] for o in objects] == [4, 9]
    assert objects[0]["centroid"] == (1.5, 1.5)
    assert objects[1]["centroid"] == (7.0, 7.0)
    assert objects[0]["bbox"] == (1, 1, 2, 2)
    assert objects[1]["bbox"] == (6, 6, 8, 8)


def test_touching_blobs_merge_into_one():
    img = np.zeros((6, 6), dtype=np.uint8)
    img[1:3, 1:3] = 1
    img[2:4, 3:5] = 1  # shares an edge with the first blob
    objects = count_objects(img)
    assert len(objects) == 1
    assert objects[0]["area"] == 8


def test_diagonal_touch_depends_on_connectivity():
    img = np.zeros((4, 4), dtype=np.uint8)
    img[0, 0] = 1
    img[1, 1] = 1  # diagonal neighbour only
    assert len(count_objects(img, connectivity=8)) == 1
    assert len(count_objects(img, connectivity=4)) == 2


def test_min_size_filters_small_objects():
    img = np.zeros((8, 8), dtype=np.uint8)
    img[0, 0] = 1            # 1-px speck
    img[2:4, 2:4] = 1        # 4-px blob
    img[5:8, 5:8] = 1        # 9-px blob
    assert len(count_objects(img, min_size=1)) == 3
    assert len(count_objects(img, min_size=2)) == 2
    assert len(count_objects(img, min_size=5)) == 1
    assert count_objects(img, min_size=5)[0]["area"] == 9


def test_boolean_input_accepted():
    img = np.zeros((3, 3), dtype=bool)
    img[0, :] = True
    objects = count_objects(img)
    assert len(objects) == 1
    assert objects[0]["area"] == 3


def test_label_image_labels_are_consecutive_and_match_count():
    rng = np.random.default_rng(7)
    img = (rng.random((12, 12)) > 0.7).astype(np.uint8)
    labelled = label_image(img)
    objects = count_objects(img)
    assert labelled.shape == img.shape
    assert set(np.unique(labelled)) == set(range(len(objects) + 1))
    assert (labelled > 0).sum() == int(img.sum())
    for obj in objects:
        assert (labelled == obj["label"]).sum() == obj["area"]


def test_full_image_is_one_object():
    img = np.ones((3, 5), dtype=np.uint8)
    objects = count_objects(img)
    assert len(objects) == 1
    assert objects[0]["area"] == 15
    assert objects[0]["centroid"] == (1.0, 2.0)
    assert objects[0]["bbox"] == (0, 0, 2, 4)


def test_random_blobs_agree_with_brute_force():
    # Independent check: flood-fill from scratch on a random image.
    rng = np.random.default_rng(123)
    img = (rng.random((15, 15)) > 0.6).astype(np.uint8)
    seen = np.zeros_like(img, dtype=bool)
    expected_areas = []
    for r in range(15):
        for c in range(15):
            if img[r, c] == 1 and not seen[r, c]:
                stack, area = [(r, c)], 0
                seen[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    area += 1
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < 15 and 0 <= nc < 15
                                    and img[nr, nc] == 1 and not seen[nr, nc]):
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                expected_areas.append(area)
    got = sorted(o["area"] for o in count_objects(img))
    assert got == sorted(expected_areas)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="must be 2D"):
        count_objects(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="must be 2D"):
        count_objects(np.zeros(5, dtype=np.uint8))


def test_invalid_values_raise():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        count_objects(np.array([[0, 2], [1, 0]]))
    with pytest.raises(ValueError, match="0/1"):
        count_objects(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_invalid_min_size_raises():
    img = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="min_size must be >= 1"):
        count_objects(img, min_size=0)
    with pytest.raises(ValueError, match="min_size must be >= 1"):
        count_objects(img, min_size=-3)


def test_invalid_connectivity_raises():
    img = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        count_objects(img, connectivity=6)
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        label_image(img, connectivity=2)


def test_non_array_input_raises():
    with pytest.raises(ValueError, match="must be a numpy array"):
        count_objects([[0, 1], [1, 0]])


def test_empty_array_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        count_objects(np.zeros((0, 0), dtype=np.uint8))
