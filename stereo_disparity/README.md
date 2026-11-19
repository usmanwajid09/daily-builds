# Stereo Disparity

Stereo disparity by SAD block matching from scratch (numpy only). For
each pixel in the left image of a rectified stereo pair, windows are
compared against the right image shifted left by d = 0..max_disparity;
the shift with the smallest sum of absolute differences wins. Larger
disparity means closer to the camera, so the map orders scene depth.

## Run

```bash
python stereo_disparity/demo.py
python -m pytest tests/test_stereo_disparity.py -q
```

## Example

Demo on a synthetic pair (near square disparity 8, far square 3):

```
Image size: 60x80
Valid pixels: 3584/4800
Disparity accuracy vs ground truth: 0.884
Median disparity — near square: 8 (true 8), far square: 3 (true 3)
Depth ordering check (near > far): PASS
```

Both squares are recovered at their true disparities and the depth
order is correct.

## API

- `make_synthetic_pair(height=60, width=80, seed=0)` → (left, right, ground_truth)
- `block_match(left, right, max_disparity=16, window_radius=2)` → int32 disparity map (-1 = unknown)
- `disparity_accuracy(disparity, ground_truth)` → fraction correct
- `depth_order_ok(disparity, near_region, far_region)` → bool
