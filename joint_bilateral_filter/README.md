# Joint Bilateral Filter

Joint (cross) bilateral filtering from scratch (numpy only).

A bilateral filter weights each neighbour by spatial closeness and by
intensity similarity. A *joint* bilateral filter takes the intensity
similarity from a separate **guidance** image instead of the image
being filtered, so a clean guide can steer the smoothing of a noisy
input — edges present in the guide are preserved even when the input
is too noisy to show them. The classic use is denoising or upsampling
a depth map guided by a colour photo.

## Run

```bash
python joint_bilateral_filter/demo.py
python -m pytest tests/test_joint_bilateral_filter.py -q
```

## Example output

Demo on a noisy 40×60 step image (50 on the left, 200 on the right,
σ=15 noise), filtered with the clean step image as guidance:

```
image: 40x60 grayscale
flat-region std before: 15.05
flat-region std after (self-guided): 5.67
flat-region std after (clean-guided): 3.14
step height before: 148.4
step height after (clean-guided): 150.6
```

The clean guide cuts flat-region noise to about a fifth of its
original level — roughly twice as effective as self-guided filtering —
while the ~150-unit step survives almost exactly.

## API

- `joint_bilateral_filter(image, guidance, diameter=5, sigma_color=25.0, sigma_space=2.0)` → float64 array, same shape as the input; `image` and `guidance` must be 2D arrays of the same shape
- `gaussian_kernel_2d(diameter, sigma)` → normalised 2D Gaussian kernel
