"""Demo: recognise synthetic faces with eigenfaces."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from eigenfaces.model import Eigenfaces


def make_faces(seed=0, size=16, n_identities=3, per_identity=5):
    """Synthetic face-like images: each identity has a smooth random
    pattern (blobs) plus per-sample noise."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for ident in range(n_identities):
        # Smooth base pattern: sum of a few random gaussian blobs.
        base = np.zeros((size, size))
        for _ in range(4):
            cy, cx = rng.integers(0, size, 2)
            amp = rng.uniform(0.4, 1.0)
            yy, xx = np.mgrid[0:size, 0:size]
            base += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
        for _ in range(per_identity):
            images.append(base + rng.normal(scale=0.05, size=(size, size)))
            labels.append(ident)
    return np.array(images), np.array(labels)


def main():
    images, labels = make_faces()
    model = Eigenfaces(n_components=8).fit(images, labels)
    # A fresh noisy copy of identity 1 (sample 7 is identity 1).
    rng = np.random.default_rng(99)
    query = images[7] + rng.normal(scale=0.05, size=images[7].shape)
    predicted = model.predict(query)[0]
    train_pred = model.predict(images)
    accuracy = float(np.mean(train_pred == labels))
    errors = model.reconstruction_error(images)
    print(f"Training faces: {len(images)} ({len(set(labels))} identities), image size: {images.shape[1:]}")
    print(f"Eigenfaces kept: 8, variance explained: {model.explained_variance_ratio_.sum():.3f}")
    print(f"Training recognition accuracy: {accuracy:.2f}")
    print(f"Query face (true identity 1) recognised as: {predicted}")
    print(f"Mean reconstruction MSE: {errors.mean():.5f} (max {errors.max():.5f})")


if __name__ == "__main__":
    main()
