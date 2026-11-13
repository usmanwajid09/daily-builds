# Median-Cut Colour Quantization, from scratch

Reduces an RGB image to at most `n_colors` colours using the classic
median-cut algorithm (Heckbert, 1982), implemented with numpy only —
no PIL, no scikit-learn. Pixels are repeatedly split into boxes along
their widest colour channel at the median, so the palette spends more
colours where the image actually has pixels, unlike a uniform grid.
Each pixel is then replaced by its box's mean colour.

## Usage

```bash
pip install -r requirements.txt
python -m median_cut_quantizer.demo
```

This builds a synthetic colourful 64x64 image, quantizes it to 8
colours, and prints the palette size and the unique colour count
before and after.

## Run tests

```bash
python -m pytest tests/test_median_cut_quantizer.py -v
```
