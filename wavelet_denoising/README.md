# Wavelet Denoising

Denoises a grayscale image with a multi-level 2D Haar discrete wavelet
transform and coefficient thresholding.

## How it works

1. **Haar DWT** (`haar_dwt2`): at each level the image is split into an
   approximation subband (local averages) and three detail subbands
   (horizontal / vertical / diagonal differences), using the
   orthonormal Haar step `(a+b)/sqrt(2)`, `(a-b)/sqrt(2)`.
2. **Noise estimate**: sigma is estimated from the median absolute
   deviation of the finest detail subband (HH), sigma = MAD / 0.6745.
3. **Thresholding**: detail coefficients below the universal threshold
   sigma * sqrt(2 ln N) are zeroed (hard) or shrunk towards zero
   (soft); the approximation subband is untouched.
4. **Inverse DWT** (`haar_idwt2`) reconstructs the denoised image.

Images whose dimensions are not multiples of 2**levels are padded
with edge replication and cropped back after inversion.

## Run

```bash
python wavelet_denoising/demo.py
python -m pytest tests/test_wavelet_denoising.py -q
```

## Example output

```
Image shape: (64, 64)
PSNR noisy vs clean:          22.31 dB
PSNR soft-thresholded:        29.80 dB
PSNR hard-thresholded:        30.49 dB
Denoised (soft) dtype: uint8, range [50, 251]
```
