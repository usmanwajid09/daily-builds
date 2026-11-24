"""Demo: summarise a 90-frame synthetic video with 3 scenes.

Run from the repo root:

    python video_summarizer/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from video_summarizer.summarizer import (
    combined_change,
    extract_keyframes,
    make_synthetic_video,
    summarize_video,
)

FPS = 30
NUM_FRAMES = 90
N_SCENES = 3


def main():
    frames = make_synthetic_video(
        num_frames=NUM_FRAMES, height=48, width=64,
        n_scenes=N_SCENES, fps=FPS, seed=42,
    )
    print(f"Video: {len(frames)} frames, {frames[0].shape}, "
          f"{FPS} fps, {NUM_FRAMES / FPS:.1f}s, {N_SCENES} true scenes "
          f"(boundaries at frames 30 and 60)")

    # Show the change score around each true boundary vs inside a scene.
    for i in (15, 30, 45, 60, 75):
        score = combined_change(frames[i - 1], frames[i])
        print(f"  combined change at frame {i:2d}: {score:.3f}")
    print(f"  mean frame colour per scene: "
          f"{[tuple(np.round(frames[s * 30].mean(axis=(0, 1))).astype(int)) for s in range(N_SCENES)]}")

    keyframes = extract_keyframes(frames, fps=FPS)
    summary = summarize_video(frames, fps=FPS)

    print(f"\nKeyframes: {len(keyframes)} -> {keyframes} "
          f"(timestamps {[round(k / FPS, 2) for k in keyframes]}s)")
    print("Shots:")
    for shot in summary["shots"]:
        print(f"  shot {shot['shot']}: frames {shot['start_frame']:2d}-"
              f"{shot['end_frame']:2d}, {shot['start_time_s']:.2f}s-"
              f"{shot['end_time_s']:.2f}s, duration {shot['duration_s']:.2f}s, "
              f"keyframe {shot['keyframe_index']}")
    print(f"\nTotal frames: {summary['total_frames']}")
    print(f"Keyframes: {summary['num_keyframes']}")
    print(f"Duration: {summary['duration_s']}s")
    print(f"Compression ratio: {summary['compression_ratio']}x "
          f"({summary['total_frames']} frames -> "
          f"{summary['num_keyframes']} keyframes)")
    print("Keyframes land on the true scene boundaries (0, 30, 60): "
          f"{keyframes == [0, 30, 60]}")


if __name__ == "__main__":
    main()
