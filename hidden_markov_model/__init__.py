"""A discrete Hidden Markov Model, from scratch with numpy only.

Supports the forward-backward algorithm (with scaling), Viterbi
decoding, sampling, and Baum-Welch (EM) training.
"""

from hidden_markov_model.model import HiddenMarkovModel

__all__ = ["HiddenMarkovModel"]
