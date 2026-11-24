# Signature Verifier

Verify handwritten-style signatures from synthetic pen-stroke
trajectories, using only numpy and the standard library.

## How it works

A signature is an `(N, 2)` array of `(x, y)` pen points.

1. **Synthetic signatures** — `make_signature(person_id, genuine=True, seed=...)`
   builds a smooth parametric stroke that is distinctive per person
   (frequencies, amplitudes and phases seeded by the person id).
   Genuine samples add only small jitter and a mild speed warp;
   forgeries use a different person's stroke shape plus extra wobble
   and a slower, uneven speed profile — a forger can imitate the
   shape but not the dynamics.
2. **Stroke features** — `extract_features(trajectory)` returns the
   aspect ratio, total path length, mean/std curvature, speed mean/std
   (from point spacing, or from timestamps when given), number of
   direction changes, and bounding-box fill.
3. **DTW distance** — `dtw_distance(a, b)` is classic dynamic time
   warping between two point sequences on normalised coordinates
   (centred, unit spread, so position and size don't matter), with a
   Euclidean point cost, returned as the mean cost per aligned step.
   An optional Sakoe-Chiba `window` limits the warping band.
4. **Enroll / verify** — `enroll(store, person_id, trajectories)`
   stores reference trajectories; `verify(store, person_id, trajectory,
   threshold=0.12)` returns the mean DTW distance to the references
   and the decision `'genuine'` (distance ≤ threshold) or `'forgery'`,
   plus a score that is 1 at distance 0 and falls to 0 at the
   threshold. `SignatureVerifier` is a stateful wrapper around the
   same two functions. The default threshold 0.12 sits well between
   genuine distances (~0.03–0.05) and forgery distances (~0.27–0.36).

All data is synthetic and generated in code; no external files are
read and no new dependencies are needed.

## Usage

```python
from signature_verifier.verifier import (
    SignatureVerifier, make_signature, extract_features,
)

verifier = SignatureVerifier()
verifier.enroll("usman", [make_signature("usman", seed=100 + i) for i in range(3)])

result = verifier.verify("usman", make_signature("usman", seed=200))
print(result)
# {'person_id': 'usman', 'distance': 0.0449, 'threshold': 0.12,
#  'decision': 'genuine', 'score': 0.63, 'n_references': 3}

features = extract_features(make_signature("usman", seed=100))
print(features["aspect_ratio"], features["mean_curvature"])
```

## Demo

Run from the repo root:

```bash
python signature_verifier/demo.py
```

It enrolls 3 references for `"usman"`, then verifies 5 genuine
samples and 5 forgeries:

```
enrolled 'usman' with 3 references (threshold 0.12)
reference features: aspect_ratio=2.29, path_length=4.57, mean_curvature=0.794, mean_speed=0.038

genuine samples:
  genuine 1: distance=0.0449 score=0.63 decision=genuine OK
  genuine 2: distance=0.0303 score=0.75 decision=genuine OK
  genuine 3: distance=0.0373 score=0.69 decision=genuine OK
  genuine 4: distance=0.0401 score=0.67 decision=genuine OK
  genuine 5: distance=0.0417 score=0.65 decision=genuine OK
forgery samples:
  forgery 1: distance=0.3564 score=0.00 decision=forgery OK
  forgery 2: distance=0.3334 score=0.00 decision=forgery OK
  forgery 3: distance=0.3596 score=0.00 decision=forgery OK
  forgery 4: distance=0.3300 score=0.00 decision=forgery OK
  forgery 5: distance=0.3449 score=0.00 decision=forgery OK

accuracy: 10/10 = 100%
```

Genuine distances (0.030–0.045) sit far below forgeries (0.330–0.360).
A robustness sweep over three people (10 genuine + 10 forgery samples
each) gave genuine max ≤ 0.047 and forgery min ≥ 0.275.

## Tests

22 pytest tests in `tests/test_signature_verifier.py`, covering
determinism, per-person distinctness, feature ranges, DTW symmetry
and translation/scale invariance, genuine-vs-forgery decisions,
custom thresholds, multi-person enrollment, and invalid input
(too few points, non-2D trajectories, verifying before enrolling,
bad thresholds, bad timestamps):

```bash
python -m pytest tests/test_signature_verifier.py -q
# 22 passed
```
