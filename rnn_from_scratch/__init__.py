"""RNN sequence classifier from scratch (numpy only).

A vanilla recurrent network with a tanh hidden state, trained with full
backpropagation through time (BPTT) and plain SGD. See ``model.py``.
"""

from .model import RNNClassifier

__all__ = ["RNNClassifier"]
