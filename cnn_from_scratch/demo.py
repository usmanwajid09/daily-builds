"""Demo: train the tiny CNN to tell circles from squares."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from cnn_from_scratch.cnn import TinyCNN, make_shape_dataset


def main():
    X_train, y_train = make_shape_dataset(n_per_class=40, size=12, seed=0)
    X_test, y_test = make_shape_dataset(n_per_class=10, size=12, seed=99)
    print(f"Train set: {X_train.shape}, test set: {X_test.shape} (0=circle, 1=square)")

    net = TinyCNN(input_shape=(12, 12), n_filters=4, kernel_size=3, n_classes=2, seed=0)
    losses = net.fit(X_train, y_train, epochs=30, lr=0.05)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    print(f"Train accuracy: {net.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {net.score(X_test, y_test):.3f}")

    # A fresh circle and a fresh square, never seen in training.
    X_fresh, y_fresh = make_shape_dataset(n_per_class=1, size=12, seed=123)
    preds = net.predict(X_fresh)
    print(f"Fresh samples (true {y_fresh.tolist()}): predicted {preds.tolist()}")
    probs = net.forward(X_fresh[0])
    print(f"Confidence on first fresh sample: {probs.max():.3f} "
          f"(probabilities sum to {probs.sum():.3f})")


if __name__ == "__main__":
    main()
