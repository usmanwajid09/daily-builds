# Gaussian kernel density estimator, from scratch

A small Gaussian KDE — no scikit-learn, just numpy. It fits on samples,
evaluates density / log-density at new points, picks a bandwidth with
Silverman's rule of thumb (or a fixed value you supply), draws new samples
from the fitted density, and approximates the mode on a grid.

## Why

KDE turns a pile of samples into a smooth probability density without
assuming a parametric shape, which makes it handy for spotting modes,
outliers, and the shape of a distribution from data alone.

## Usage

```bash
pip install -r requirements.txt
python kernel_density_estimator/demo.py
```

This fits a KDE on a bimodal 1D mixture (modes near -2 and +2.5) and
prints the density at a few points plus the estimated mode.

## Demo output (seed 42)

```
Samples: 300, bandwidth (Silverman): 0.6947
  x= -2.0  density=0.2051  log-density=-1.5842
  x=  0.0  density=0.0402  log-density=-3.2128
  x=  2.5  density=0.1690  log-density=-1.7781
Estimated mode: -2.026
New samples: [3.35, 1.948, 2.265, -0.729, -1.681]
```

## Run tests

```bash
python -m pytest tests/test_kernel_density_estimator.py -v
```
