# background_subtraction

Running-average background subtraction from scratch (numpy only).

The background model is the running average of the frames seen so
far: `bg = (1 - alpha) * bg + alpha * frame`. A pixel is foreground
when `|frame - bg| > threshold`. Isolated foreground pixels (sensor
noise) are removed by requiring each foreground pixel to have at
least `min_neighbors` foreground neighbours in its 3x3 neighbourhood.

## API

- `BackgroundSubtractor(learning_rate=0.05, threshold=25.0,
  min_neighbors=2)` — `apply(frame)` updates the model and returns a
  boolean foreground mask (the first frame initialises the model and
  returns an all-False mask); `get_background()` returns a copy of the
  model; `reset()` forgets it.
- `running_average_update(background, frame, learning_rate)` — one
  update step.
- `foreground_mask(frame, background, threshold, min_neighbors=2)` —
  mask for a single frame against a given background.

## How to run

```bash
python background_subtraction/demo.py
python -m pytest tests/test_background_subtraction.py -q
```

## Demo output

```
video: 10 frames of (40, 60)
  frame 0: 0 foreground pixels
  frame 1: 0 foreground pixels
  frame 2: 0 foreground pixels
  frame 3: 0 foreground pixels
  frame 4: 0 foreground pixels
  frame 5: 100 foreground pixels
  frame 6: 100 foreground pixels
  frame 7: 130 foreground pixels
  frame 8: 160 foreground pixels
  frame 9: 160 foreground pixels
background model mean: 52.51 (true background ~50)
```

The demo video is static for 5 frames, then a bright 10x10 square
moves 3 pixels right per frame. Frames 5-6 detect exactly the 100
square pixels; later frames count more because the square's old
position is still brighter than the adapting background (a ghost of
the trail), which the running average absorbs over time. 12 tests pass.
