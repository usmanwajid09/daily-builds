# Non-Local Means Denoising

Non-local means image denoising from scratch (numpy only). Each output
pixel is a weighted average of pixels in a search window, weighted by
how similar the small patch around each neighbour is to the patch
around the target pixel — so repeated texture is averaged together
while sharp edges are barely averaged across. Works on grayscale and
RGB images.

## Run

```bash
python non_local_means/demo.py
python -m pytest tests/test_non_local_means.py -q
```

## Example

Demo on a 40×60 step image (50 / 200) with σ=15 Gaussian noise:

```
PSNR before denoising: 24.60 dB
PSNR after denoising:  35.67 dB
    noisy: flat-region std left=14.92 right=14.99, step height=149.6 (true 150.0)
 denoised: flat-region std left=3.77 right=3.95, step height=149.6 (true 150.0)
```

Flat-region noise drops to about a third while the step height is
preserved almost exactly.

## API

- `non_local_means(image, search_radius=5, patch_radius=1, h=10.0)` → float64 array, same shape as the input
- `psnr(reference, denoised, peak=255.0)` → PSNR in dB (inf for identical images)
