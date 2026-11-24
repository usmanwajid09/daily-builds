"""Tests for plate_detector.plate (licence-plate detection)."""

import numpy as np
import pytest

from plate_detector.plate import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    box_iou,
    component_boxes,
    detect_plates,
    extract_plate_crop,
    label_components,
    make_plate_scene,
    match_detections,
    threshold_image,
)

PLATES = [(30, 40, 72, 18), (180, 90, 60, 16), (90, 170, 80, 20)]


def _scene(seed=7):
    return make_plate_scene(240, 320, plate_specs=PLATES, seed=seed)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_single_plate_detected_with_exact_box():
    image, truth = make_plate_scene(120, 200, plate_specs=[(20, 30, 60, 16)],
                                    seed=1)
    detections = detect_plates(image)
    assert len(detections) == 1
    det = detections[0]
    assert (det["x"], det["y"], det["width"], det["height"]) == (20, 30, 60, 16)
    assert det["aspect_ratio"] == pytest.approx(3.75)
    assert det["area"] > 300


def test_three_plates_all_detected():
    image, truth = _scene()
    detections = detect_plates(image)
    assert len(detections) == 3
    n_matched, pairs = match_detections(detections, truth)
    assert n_matched == 3
    assert all(iou == pytest.approx(1.0) for _, _, iou in pairs)


def test_random_scenes_detect_all_plates():
    for seed in range(5):
        image, truth = make_plate_scene(240, 320, n_plates=5, seed=seed)
        detections = detect_plates(image)
        assert len(detections) == 5
        n_matched, _ = match_detections(detections, truth)
        assert n_matched == 5


def test_detections_sorted_top_to_bottom():
    image, _ = _scene()
    detections = detect_plates(image)
    assert [(d["y"], d["x"]) for d in detections] == sorted(
        (d["y"], d["x"]) for d in detections)


def test_crop_matches_detected_box():
    image, _ = _scene()
    detections = detect_plates(image)
    crop = extract_plate_crop(image, detections[0])
    assert crop.shape == (18, 72)
    # The crop is the bright plate: mean well above the dark background.
    assert crop.mean() > 150
    # ... and it still contains the dark characters.
    assert (crop < 100).sum() > 0


def test_crop_accepts_tuple_box():
    image, _ = _scene()
    crop = extract_plate_crop(image, (30, 40, 72, 18))
    assert crop.shape == (18, 72)
    assert crop.mean() > 150


def test_closing_bridges_character_gaps():
    # A plate-shaped bright mask with a 2 px dark vertical slit: closing
    # must merge it into one component, plain thresholding must not.
    mask = np.zeros((40, 100), dtype=bool)
    mask[10:26, 10:70] = True
    mask[10:26, 38:40] = False  # the "character" gap
    labels, n = label_components(mask)
    assert n == 2
    closed = binary_closing(mask, kernel_size=3)
    _, n_closed = label_components(closed)
    assert n_closed == 1


# ---------------------------------------------------------------------------
# Edge cases / rejection
# ---------------------------------------------------------------------------

def test_blank_scene_has_no_detections():
    image, _ = make_plate_scene(240, 320, plate_specs=[], seed=7)
    assert detect_plates(image) == []


def test_pure_noise_image_has_no_detections():
    rng = np.random.default_rng(0)
    image = rng.normal(40, 8, size=(200, 200)).clip(0, 255).astype(np.uint8)
    assert detect_plates(image) == []


def test_square_bright_blob_rejected_by_aspect_ratio():
    image = np.full((120, 120), 40, dtype=np.uint8)
    image[40:70, 40:70] = 220  # 30x30 square, aspect 1.0
    assert detect_plates(image) == []


def test_tiny_bright_strip_rejected_by_area():
    image = np.full((120, 200), 40, dtype=np.uint8)
    image[50:58, 20:50] = 220  # aspect 3.75 but only 240 px of bright area
    assert detect_plates(image, min_area=300) == []
    # ... unless the area filter is relaxed.
    assert len(detect_plates(image, min_area=100)) == 1


def test_box_iou():
    box = {"x": 10, "y": 10, "width": 20, "height": 10}
    assert box_iou(box, box) == pytest.approx(1.0)
    assert box_iou(box, {"x": 100, "y": 100, "width": 20, "height": 10}) == 0.0
    assert box_iou(box, (10, 10, 10, 10)) == pytest.approx(0.5)


def test_match_detections_counts_each_truth_once():
    truth = [{"x": 0, "y": 0, "width": 60, "height": 16}]
    detections = [dict(truth[0]), dict(truth[0])]  # duplicate detection
    n_matched, pairs = match_detections(detections, truth)
    assert n_matched == 1
    assert len(pairs) == 1


def test_scene_is_deterministic_for_same_seed():
    image_a, truth_a = make_plate_scene(120, 200, n_plates=2, seed=3)
    image_b, truth_b = make_plate_scene(120, 200, n_plates=2, seed=3)
    assert np.array_equal(image_a, image_b)
    assert truth_a == truth_b


def test_different_seeds_give_different_scenes():
    image_a, _ = make_plate_scene(120, 200, n_plates=2, seed=3)
    image_b, _ = make_plate_scene(120, 200, n_plates=2, seed=4)
    assert not np.array_equal(image_a, image_b)


def test_morphology_helpers():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    assert binary_dilation(mask, 3).sum() == 9
    assert binary_erosion(mask, 3).sum() == 0
    block = np.zeros((7, 7), dtype=bool)
    block[2:5, 2:5] = True
    assert binary_erosion(block, 3).sum() == 1
    assert binary_closing(block, 3).sum() == 9


def test_threshold_and_labelling():
    image = np.full((10, 10), 40, dtype=np.uint8)
    image[2:4, 2:5] = 200
    image[7:9, 7:9] = 200
    mask = threshold_image(image, 150)
    assert mask.sum() == 10
    labels, n = label_components(mask)
    assert n == 2
    boxes = component_boxes(labels, n)
    assert sorted(b["area"] for b in boxes) == [4, 6]


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_detect_rejects_non_2d_image():
    with pytest.raises(ValueError, match="2-D"):
        detect_plates(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        detect_plates(np.zeros((0, 0), dtype=np.uint8))


def test_detect_rejects_bad_threshold():
    image, _ = _scene()
    with pytest.raises(ValueError, match="threshold"):
        detect_plates(image, threshold=300)
    with pytest.raises(ValueError, match="threshold"):
        detect_plates(image, threshold=-1)


def test_detect_rejects_bad_filter_params():
    image, _ = _scene()
    with pytest.raises(ValueError, match="min_area"):
        detect_plates(image, min_area=0)
    with pytest.raises(ValueError, match="max_aspect"):
        detect_plates(image, min_aspect=2.0, max_aspect=1.0)


def test_crop_rejects_out_of_bounds_box():
    image, _ = _scene()
    with pytest.raises(ValueError, match="exceeds"):
        extract_plate_crop(image, (300, 200, 50, 20))
    with pytest.raises(ValueError, match=">= 0"):
        extract_plate_crop(image, (-5, 10, 20, 10))
    with pytest.raises(ValueError, match="> 0"):
        extract_plate_crop(image, (10, 10, 0, 10))


def test_crop_rejects_malformed_box():
    image, _ = _scene()
    with pytest.raises(ValueError, match="box"):
        extract_plate_crop(image, {"x": 10})
    with pytest.raises(ValueError, match="box"):
        extract_plate_crop(image, "not-a-box")


def test_scene_rejects_plate_outside_image():
    with pytest.raises(ValueError, match="does not fit"):
        make_plate_scene(100, 100, plate_specs=[(90, 90, 60, 16)], seed=0)
    with pytest.raises(ValueError, match="too small"):
        make_plate_scene(100, 100, plate_specs=[(10, 10, 5, 3)], seed=0)


def test_match_rejects_bad_iou_threshold():
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([], [], iou_threshold=0)
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([], [], iou_threshold=1.5)


def test_morphology_rejects_even_kernel():
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="kernel_size"):
        binary_dilation(mask, kernel_size=4)
    with pytest.raises(ValueError, match="kernel_size"):
        binary_erosion(mask, kernel_size=0)
