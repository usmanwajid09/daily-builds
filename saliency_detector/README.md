# Saliency Detector

Spectral-residual saliency detection from scratch (numpy only).

Implements Hou & Zhang (2007): the log amplitude spectrum of a natural
image is mostly smooth, so subtracting a blurred copy of it leaves a
"spectral residual" that marks the unusual parts of the scene.
Recombining the residual with the original phase and transforming back
gives a saliency map whose peaks are the regions a human eye would
look at first — useful for thumbnail cropping, object proposals, and
guiding downstream vision pipelines.

## Run

```bash
python saliency_detector/demo.py
python -m pytest tests/test_saliency_detector.py -q
```

## Example output

Demo on a 64×64 smooth sinusoidal texture with a bright disk (radius
8) planted at (32, 40):

```
image: 64x64 grayscale, disk at (32, 40) (radius 8)
saliency range: min=0.014 max=1.000
saliency peak at: (35, 33)
distance from disk centre: 7.6 px
```

The saliency peak lands on the disk's rim, 7.6 px from its centre —
inside the planted disk. Spectral residual highlights the boundary
of the unusual object rather than its flat interior, which is exactly
where the scene differs most from the smooth background.

## API

- `spectral_residual_saliency(image, blur_size=3, smooth_sigma=2.0)` → float64 saliency map in [0, 1], same shape as the input
- `saliency_peak(saliency_map)` → (row, col) of the most salient pixel
