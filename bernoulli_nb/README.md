# Bernoulli Naive Bayes

A Bernoulli naive Bayes text/binary classifier built from scratch with
numpy only.

## What it does

Models binary features (0/1), such as word-presence vectors in text
classification. Each feature is treated as a coin flip whose
probability of being 1 depends on the class, with Laplace smoothing
(`alpha`, default 1.0) so unseen feature/class combinations still get a
non-zero probability. Predicts the class with the highest log posterior.

## How to run

```bash
python bernoulli_nb/demo.py
```

## Demo result

A tiny synthetic spam/ham dataset: 6 binary word-presence features
(`free`, `win`, `money`, `meeting`, `project`, `lunch`), 100 messages
per class, 80/20 train/test split.

- Test accuracy: **0.975**
- Learned P(word=1 | spam) ≈ [0.82, 0.72, 0.72, 0.05, 0.10, 0.06]
- Learned P(word=1 | ham) ≈ [0.10, 0.06, 0.12, 0.78, 0.69, 0.69]
