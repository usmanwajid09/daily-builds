# video_summarizer

Summarise a video by keyframe extraction — built from scratch with
**numpy + stdlib only** (no OpenCV, no scipy, no data files).

A video is a list of HxWx3 `uint8` frames. Scene changes are detected
from two complementary signals:

* **Frame difference** (`frame_difference`) — mean absolute pixel
  difference between consecutive frames. Catches motion and abrupt cuts.
* **Colour-histogram change** (`histogram_change`) — per-channel
  histogram distance (intersection or chi-square). Catches colour /
  content changes even when the average pixel value barely moves.

`extract_keyframes` keeps frame 0 plus every frame whose combined
score (`combined_change`, a weighted average of the two signals)
exceeds a threshold, enforcing a minimum gap (default `fps // 2`
frames) so one cut does not produce a burst of near-duplicate
keyframes. `summarize_video` turns the keyframes into a shot list
with timestamps plus summary stats (total frames, keyframes,
compression ratio).

Because there are no video files, the demo video is **generated in
code** (`make_synthetic_video`): K scenes, each a distinct base colour
with a fixed per-scene texture, small per-frame noise and a slow
brightness drift — so frames inside a scene change only slightly,
while scene boundaries are abrupt colour changes. Deterministic for a
given `seed`.

## How to run

From the repo root (`daily-builds`):

```bash
/home/hatch/workspace/db-venv/bin/python video_summarizer/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_video_summarizer.py -q
```

## Usage

```python
from video_summarizer.summarizer import make_synthetic_video, summarize_video

frames = make_synthetic_video(num_frames=90, n_scenes=3, fps=30, seed=42)
summary = summarize_video(frames, fps=30)
print(summary["shots"])            # one dict per shot, with timestamps
print(summary["compression_ratio"])  # 30.0
```

## Demo output

```
Video: 90 frames, (48, 64, 3), 30 fps, 3.0s, 3 true scenes (boundaries at frames 30 and 60)
  combined change at frame 15: 0.019
  combined change at frame 30: 0.574
  combined change at frame 45: 0.017
  combined change at frame 60: 0.509
  combined change at frame 75: 0.023
  mean frame colour per scene: [(196, 56, 56), (56, 176, 76), (57, 86, 196)]

Keyframes: 3 -> [0, 30, 60] (timestamps [0.0, 1.0, 2.0]s)
Shots:
  shot 0: frames  0-29, 0.00s-1.00s, duration 1.00s, keyframe 0
  shot 1: frames 30-59, 1.00s-2.00s, duration 1.00s, keyframe 30
  shot 2: frames 60-89, 2.00s-3.00s, duration 1.00s, keyframe 60

Total frames: 90
Keyframes: 3
Duration: 3.0s
Compression ratio: 30.0x (90 frames -> 3 keyframes)
Keyframes land on the true scene boundaries (0, 30, 60): True
```

The change score is ~0.02 inside a scene and ~0.5–0.6 at a boundary,
so the default threshold (0.25) separates them with a wide margin.

## Tests

17 pytest tests in `tests/test_video_summarizer.py` — all passing —
covering synthetic-video shape/determinism/scene colours, frame
difference and histogram change (exact values, identical frames,
black vs white), combined-score ordering, keyframes landing exactly
on the true boundaries, single-scene videos, min-gap duplicate
suppression, shot tiling/durations, and invalid input (empty frame
list, mismatched shapes, bad fps/threshold/min_gap/bins/method).
