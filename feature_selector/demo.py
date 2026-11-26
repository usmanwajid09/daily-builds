"""Shows mutual-information selection finding the 2 informative features among 6."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from feature_selector.model import (
    correlation_scores,
    mutual_info_scores,
    select_top_k,
    variance_threshold,
)


def make_data(seed: int = 42, n: int = 400):
    rng = np.random.default_rng(seed)
    informative = rng.normal(0, 1, size=(n, 2))
    # Class depends on the sign of the first informative feature plus noise.
    y = (informative[:, 0] + 0.5 * informative[:, 1] > 0).astype(int)
    noise = rng.normal(0, 1, size=(n, 4))
    x = np.hstack([informative, noise])
    return x, y


def main():
    x, y = make_data()
    mi = mutual_info_scores(x, y)
    corr = correlation_scores(x, y)
    top2 = select_top_k(mi, 2)
    kept = variance_threshold(x, threshold=0.0)

    print(f"Samples: {len(y)}, features: {x.shape[1]}")
    print(f"Mutual information scores: {np.round(mi, 4).tolist()}")
    print(f"Correlation scores:        {np.round(corr, 4).tolist()}")
    print(f"Top-2 features by MI: {top2.tolist()} (informative features are 0 and 1)")
    print(f"Features kept by variance threshold: {kept.sum()} of {len(kept)}")


if __name__ == "__main__":
    main()
