# Digit Recogniser (template matching)

Recognise synthetic 8x8 digit patterns (0–9) by nearest-template
matching. Each digit is a hand-defined 8x8 binary template; recognition
is the smallest Euclidean distance to a template, so it is fast,
deterministic, and robust to light pixel noise.

## Why

Template matching is the simplest form of OCR: it shows how far a
nearest-prototype classifier gets before you need learned features, and
it gives a baseline any fancier recogniser has to beat.

## Usage

```bash
pip install -r requirements.txt
python -m digit_recognizer.demo
```

The demo generates 20 noisy variants (5% pixel flips) of each digit and
prints overall and per-digit accuracy, plus example predictions with
their winning and runner-up distances.

```python
from digit_recognizer import recognize, template, add_noise

noisy = add_noise(template(7), flip_prob=0.05)
digit, scores = recognize(noisy)   # digit == 7, scores maps 0-9 -> distance
```

## Run tests

```bash
python -m pytest tests/test_digit_recognizer.py -v
```
