import numpy as np
import pytest

from video_summarizer.summarizer import (
    combined_change,
    extract_keyframes,
    frame_difference,
    frame_histogram,
    histogram_change,
    make_synthetic_video,
    summarize_video,
)


def solid_frame(value, height=8, width=8):
    return np.full((height, width, 3), value, dtype=np.uint8)


# --- make_synthetic_video ---

def test_synthetic_video_shape_dtype_and_count():
    frames = make_synthetic_video(num_frames=90, height=48, width=64,
                                  n_scenes=3, fps=30, seed=42)
    assert len(frames) == 90
    assert all(f.shape == (48, 64, 3) for f in frames)
    assert all(f.dtype == np.uint8 for f in frames)


def test_synthetic_video_is_deterministic_for_a_seed():
    a = make_synthetic_video(seed=7)
    b = make_synthetic_video(seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = make_synthetic_video(seed=8)
    assert not np.array_equal(a[0], c[0])


def test_synthetic_video_scenes_have_distinct_colours():
    frames = make_synthetic_video(num_frames=90, n_scenes=3, seed=42)
    means = [frames[s * 30].mean(axis=(0, 1)) for s in range(3)]
    # Scene means must be far apart (red vs green vs blue base colours).
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.abs(means[i] - means[j]).max() > 80


def test_synthetic_video_invalid_args_raise():
    with pytest.raises(ValueError):
        make_synthetic_video(num_frames=0)
    with pytest.raises(ValueError):
        make_synthetic_video(num_frames=10, n_scenes=11)
    with pytest.raises(ValueError):
        make_synthetic_video(height=0)
    with pytest.raises(ValueError):
        make_synthetic_video(fps=0)
    with pytest.raises(ValueError):
        make_synthetic_video(num_frames=9, n_scenes=3,
                             scene_colours=[(1, 2, 3)])


# --- frame_difference ---

def test_frame_difference_identical_frames_is_zero():
    f = solid_frame(100)
    assert frame_difference(f, f) == 0.0


def test_frame_difference_known_value():
    a = solid_frame(0)
    b = solid_frame(100)
    assert frame_difference(a, b) == 100.0
    c = np.zeros((8, 8, 3), dtype=np.uint8)
    c[:, :, 0] = 255  # only the red channel differs
    assert frame_difference(a, c) == pytest.approx(255 / 3)


def test_frame_difference_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        frame_difference(solid_frame(0), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        frame_difference(solid_frame(0), np.zeros((8, 8), dtype=np.uint8))


# --- histogram_change ---

def test_histogram_identical_frames_is_zero():
    f = solid_frame(77)
    assert histogram_change(f, f) == 0.0
    assert histogram_change(f, f, method="chi_square") == 0.0


def test_histogram_change_black_vs_white_is_maximal():
    assert histogram_change(solid_frame(0), solid_frame(255)) == pytest.approx(1.0)
    assert histogram_change(solid_frame(0), solid_frame(255),
                            method="chi_square") == pytest.approx(1.0)


def test_histogram_change_invalid_method_raises():
    with pytest.raises(ValueError):
        histogram_change(solid_frame(0), solid_frame(1), method="bogus")
    with pytest.raises(ValueError):
        frame_histogram(solid_frame(0), bins=1)


def test_combined_change_range_and_ordering():
    same = combined_change(solid_frame(50), solid_frame(50))
    assert same == 0.0
    small = combined_change(solid_frame(50), solid_frame(60))
    big = combined_change(solid_frame(0), solid_frame(255))
    assert 0.0 < small < big <= 1.0
    with pytest.raises(ValueError):
        combined_change(solid_frame(0), solid_frame(1), diff_weight=1.5)


# --- extract_keyframes ---

def test_extract_keyframes_finds_true_scene_boundaries():
    frames = make_synthetic_video(num_frames=90, n_scenes=3, fps=30, seed=42)
    assert extract_keyframes(frames, fps=30) == [0, 30, 60]


def test_extract_keyframes_single_scene_only_frame_zero():
    frames = make_synthetic_video(num_frames=60, n_scenes=1, fps=30, seed=42)
    assert extract_keyframes(frames, fps=30) == [0]


def test_extract_keyframes_min_gap_suppresses_duplicates():
    frames = make_synthetic_video(num_frames=90, n_scenes=3, fps=30, seed=42)
    # A huge min_gap lets only the first keyframe through.
    assert extract_keyframes(frames, fps=30, min_gap=100) == [0]
    # A tiny threshold with min_gap=1 still finds the boundaries first.
    keys = extract_keyframes(frames, fps=30, threshold=0.05, min_gap=1)
    assert keys[:3] == [0, 30, 60]


def test_extract_keyframes_invalid_input_raises():
    with pytest.raises(ValueError):
        extract_keyframes([])
    with pytest.raises(ValueError):
        extract_keyframes([solid_frame(0), np.zeros((4, 4, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        extract_keyframes(make_synthetic_video(num_frames=10), fps=0)
    with pytest.raises(ValueError):
        extract_keyframes(make_synthetic_video(num_frames=10), threshold=0)
    with pytest.raises(ValueError):
        extract_keyframes(make_synthetic_video(num_frames=10), threshold=1.5)
    with pytest.raises(ValueError):
        extract_keyframes(make_synthetic_video(num_frames=10), min_gap=0)


# --- summarize_video ---

def test_summarize_video_shots_cover_all_frames():
    frames = make_synthetic_video(num_frames=90, n_scenes=3, fps=30, seed=42)
    summary = summarize_video(frames, fps=30)
    assert summary["total_frames"] == 90
    assert summary["num_keyframes"] == 3
    assert summary["duration_s"] == pytest.approx(3.0)
    assert summary["compression_ratio"] == pytest.approx(30.0)
    shots = summary["shots"]
    assert len(shots) == 3
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [
        (0, 29), (30, 59), (60, 89)]
    assert [s["duration_s"] for s in shots] == [1.0, 1.0, 1.0]
    assert [s["start_time_s"] for s in shots] == [0.0, 1.0, 2.0]
    assert [s["keyframe_index"] for s in shots] == [0, 30, 60]
    # Shots tile the video with no gaps or overlaps.
    assert sum(s["end_frame"] - s["start_frame"] + 1 for s in shots) == 90


def test_summarize_video_empty_frames_raises():
    with pytest.raises(ValueError):
        summarize_video([])
