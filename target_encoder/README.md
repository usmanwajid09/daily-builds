# target_encoder

Smoothed target encoding for categorical features (numpy only).

Target encoding replaces each category with the mean of the target for that
category, shrunk toward the global target mean so rare categories don't get
extreme values from a handful of samples:

```
enc(c) = (count_c * mean_c + smoothing * global_mean) / (count_c + smoothing)
```

Categories with fewer than `min_samples_leaf` samples are encoded with the
global mean exactly.

## Features

- `smoothing` (default 1.0): `0` gives raw per-category means; larger values
  pull every encoding closer to the global mean.
- `min_samples_leaf` (default 1): categories below this count map to the
  global mean.
- `handle_unknown="value"` (default) encodes unseen categories with the
  global mean; `"error"` raises `ValueError`.
- `handle_missing="value"` (default) treats `None`/`NaN` as its own
  category; `"error"` raises `ValueError`.
- Works with 1D or 2D `X` (one encoder per column), string or int labels,
  binary or continuous `y`.

## How to run

```bash
python target_encoder/demo.py
python -m pytest tests/test_target_encoder.py -q
```

## Example

```python
from target_encoder import TargetEncoder

enc = TargetEncoder(smoothing=10.0).fit(cities, converted)
print(enc.transform(["Lahore", "Peshawar"]))  # unseen -> global mean
```

## Example output (demo)

```
Rows: 200   Global mean (conversion rate): 0.5500

City          count  raw mean   smoothed
----------------------------------------
Lahore          100    0.6000     0.5955
Karachi          60    0.3500     0.3786
Islamabad        30    0.7000     0.6625
Quetta            8    1.0000     0.7500
Murree            2    0.0000     0.4583

Shrinkage in action:
  Quetta   (n=8,  raw mean 1.00) -> 0.7500  (pulled down toward global mean)
  Murree   (n=2,  raw mean 0.00) -> 0.4583  (pulled up toward global mean)
  Lahore   (n=100, raw mean 0.6000) -> 0.5955  (barely moved: plenty of data)

Unseen category 'Peshawar' encodes to the global mean: 0.5500 (Lahore -> 0.5955)
```

## Tests

25 pytest tests in `tests/test_target_encoder.py`, covering the exact
encoding formula, shrinkage of rare categories, unseen/missing handling,
`min_samples_leaf`, multi-column input, and input validation.
