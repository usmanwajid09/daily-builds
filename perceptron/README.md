# Perceptron, from scratch

The classic Rosenblatt perceptron, implemented with numpy only — no
scikit-learn. It learns a linear decision boundary `sign(w . x + b)` by
updating the weights on every misclassified sample. On linearly
separable data it is guaranteed to converge; on non-separable data it
keeps the *pocket* — the best weights seen so far (fewest training
errors) — instead of the last, possibly worse, weights.

## Usage

```bash
pip install -r requirements.txt
python -m perceptron.demo
```

This generates two synthetic, linearly separable 2D blobs, fits the
perceptron, and prints test accuracy and how many epochs it took to
converge.

## Run tests

```bash
python -m pytest tests/test_perceptron.py -v
```
