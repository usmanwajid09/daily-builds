# MLP Autoencoder

A multilayer-perceptron autoencoder built from scratch with numpy
only: `input -> tanh hidden -> tanh latent -> tanh hidden -> linear
output`, trained by minibatch gradient descent on mean squared
reconstruction error.

The bottleneck forces the network to compress its input, so the
per-sample reconstruction error works as an anomaly score — data the
network has seen reconstructs well, unfamiliar data does not.

## Run

```bash
python mlp_autoencoder/demo.py
python -m pytest tests/test_mlp_autoencoder.py -q
```

## Example

```python
from mlp_autoencoder import MLPAutoencoder

model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2)
model.fit(X, epochs=300, lr=0.01)
codes = model.encode(X)              # (n_samples, 2)
recon = model.reconstruct(X)         # (n_samples, 8)
scores = model.anomaly_score(X)      # ~1 for normal data, >>1 for anomalies
```

Demo on synthetic 8-D data living on a 2-D manifold:

```
train MSE before training: 1.3715
train MSE after 300 epochs: 0.003553
median anomaly score, normal data: 0.61
median anomaly score, outliers (+3 sigma noise): 2664.05
outliers score 4388.4x higher than normal
```

## API

- `MLPAutoencoder(input_dim, hidden_dim=32, latent_dim=2, seed=0)` — `latent_dim` must be smaller than `input_dim`
- `fit(X, epochs=200, lr=0.01, batch_size=64)` → self; `history_` holds the MSE after each epoch (Adam optimiser, implemented from scratch)
- `encode(X)` → latent codes; `decode(Z)` → reconstructions from codes
- `reconstruct(X)` → reconstructions; `reconstruction_error(X)` → per-sample MSE
- `anomaly_score(X)` → reconstruction error divided by the median training error
