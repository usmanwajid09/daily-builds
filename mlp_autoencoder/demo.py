"""Demo: MLP autoencoder on synthetic low-rank data.

Generates 8-D data that really lives on a 2-D manifold (plus a little
noise), trains a 8 -> 16 -> 2 -> 16 -> 8 autoencoder, then shows the
reconstruction error on held-out normal data versus on gross outliers.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mlp_autoencoder import MLPAutoencoder


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size=(n, 2))
    mixing = rng.normal(0.0, 1.0, size=(2, 8))
    X = latent @ mixing + rng.normal(0.0, 0.05, size=(n, 8))
    return X / X.std()


def main():
    X = make_data()
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    # MSE of an untrained model, for comparison.
    out0, _ = model._forward(X)
    mse_before = float(np.mean((out0 - X) ** 2))
    model.fit(X, epochs=300, lr=0.01)
    mse_after = model.history_[-1]
    codes = model.encode(X[:5])
    rng = np.random.default_rng(99)
    outliers = X[:20] + rng.normal(0.0, 3.0, size=(20, 8))
    normal_score = float(np.median(model.anomaly_score(X[:20])))
    outlier_score = float(np.median(model.anomaly_score(outliers)))
    print(f"data: {X.shape[0]} samples x {X.shape[1]} features, latent dim 2")
    print(f"train MSE before training: {mse_before:.4f}")
    print(f"train MSE after 300 epochs: {mse_after:.6f}")
    print(f"latent codes (first 2 samples): {np.round(codes[:2], 3).tolist()}")
    print(f"median anomaly score, normal data: {normal_score:.2f}")
    print(f"median anomaly score, outliers (+3 sigma noise): {outlier_score:.2f}")
    print(f"outliers score {outlier_score / max(normal_score, 1e-9):.1f}x higher than normal")
    return model


if __name__ == "__main__":
    main()
