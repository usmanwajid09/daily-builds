"""Demo: train the from-scratch RNN on the parity task.

Each sample is a sequence of 6 bits (one-hot encoded); the label is the
parity of the bits (0 = even number of 1s, 1 = odd). No single time step
carries the answer, so the network must remember what it has seen.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from rnn_from_scratch.model import RNNClassifier, make_parity_dataset


def main():
    X_train, y_train = make_parity_dataset(400, seq_len=6, seed=42)
    X_test, y_test = make_parity_dataset(100, seq_len=6, seed=123)
    print(f"Train set: {X_train.shape}, test set: {X_test.shape} "
          f"(0=even parity, 1=odd parity)")

    clf = RNNClassifier(hidden_size=32, epochs=150, seed=42)
    clf.fit(X_train, y_train)
    print(f"Initial loss: {clf.loss_history_[0]:.4f}")
    print(f"Final loss:   {clf.loss_history_[-1]:.4f}")
    print(f"Loss decreased: {clf.loss_history_[-1] < clf.loss_history_[0]}")
    print(f"Train accuracy: {clf.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {clf.score(X_test, y_test):.3f}")

    # A few fresh sequences, never seen in training.
    X_fresh, y_fresh = make_parity_dataset(5, seq_len=6, seed=999)
    preds = clf.predict(X_fresh)
    print(f"Fresh sequences (true {y_fresh.tolist()}): predicted {preds.tolist()}")
    probs = clf.predict_proba(X_fresh)
    print(f"Confidence on first fresh sample: {probs[0].max():.3f} "
          f"(probabilities sum to {probs[0].sum():.3f})")


if __name__ == "__main__":
    main()
