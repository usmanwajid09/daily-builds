"""Generates correlated synthetic data, fits PCA, reports variance explained."""
import numpy as np

from pca_from_scratch.model import PCA


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 200
    # Strongly correlated 4D data: two latent factors drive the features.
    factor1 = rng.normal(size=n)
    factor2 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=(n, 4))
    x = np.column_stack([
        2.0 * factor1 + 0.5 * factor2,
        2.0 * factor1 - 0.5 * factor2,
        0.5 * factor1 + 2.0 * factor2,
        -0.5 * factor1 + 2.0 * factor2,
    ]) + noise
    return x


def main():
    x = make_data()
    model = PCA(n_components=2).fit(x)
    projected = model.transform(x)
    reconstructed = model.inverse_transform(projected)
    mse = float(np.mean((x - reconstructed) ** 2))
    var2 = float(model.explained_variance_ratio_[:2].sum())

    print(f"Samples: {len(x)}, features: {x.shape[1]}, components kept: 2")
    print(f"Variance explained by first 2 components: {var2:.3f}")
    print(f"Explained variance ratios: {np.round(model.explained_variance_ratio_, 3).tolist()}")
    print(f"Reconstruction MSE with 2 components: {mse:.4f}")
    print(f"Projected shape: {projected.shape}")


if __name__ == "__main__":
    main()
