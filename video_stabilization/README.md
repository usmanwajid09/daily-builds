# Video Stabilization

Simple video stabilisation from scratch (numpy only).

Frame-to-frame translation is estimated with phase correlation (the
peak of the inverse FFT of the normalised cross-power spectrum), the
resulting camera trajectory is accumulated and smoothed with a moving
average, and each frame is warped back by the difference between the
raw and smoothed trajectories. Random shake is removed while smooth,
deliberate camera motion is kept.

## Run

```bash
python video_stabilization/demo.py
python -m pytest tests/test_video_stabilization.py -q
```

## Example output

Demo on 21 synthetic 48×64 frames (a bright square on a textured
background, drifting smoothly with ±2 px random jitter):

```
frames: 21 of 48x64 grayscale
frame-to-frame jitter std before: 2.30 px
frame-to-frame jitter std after:  0.48 px
jitter reduced by: 79%
```

Stabilisation cuts the frame-to-frame jitter by about four fifths on
this sequence; the exact figure varies with the random seed.

## API

- `estimate_translation(prev, curr)` → `(dy, dx)` integer shift from `prev` to `curr`
- `smooth_trajectory(trajectory, window=5)` → moving-average smoothed (N, 2) trajectory
- `stabilize_frames(frames, window=5)` → list of stabilised frames, same shape and count
- `shift_image(image, dy, dx)` → image translated by `(dy, dx)`, gap filled with 0
