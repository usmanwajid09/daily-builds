"""Discrete Hidden Markov Model, from scratch with numpy only.

Implements the forward-backward algorithm (with scaling, so long
sequences do not underflow), Viterbi decoding, sampling, and
Baum-Welch (EM) training.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _dirichlet_row(rng: np.random.Generator, n: int) -> np.ndarray:
    """One random probability row (all entries > 0, sums to 1)."""
    row = rng.dirichlet(np.ones(n))
    return np.maximum(row, _EPS) / np.maximum(row, _EPS).sum()


class HiddenMarkovModel:
    """Discrete Hidden Markov Model with ``n_states`` hidden states
    and ``n_symbols`` observable symbols (0 .. n_symbols - 1).

    Parameters
    ----------
    n_states : int
        Number of hidden states (>= 1).
    n_symbols : int
        Number of observable symbols (>= 1).
    random_state : int | None
        Seed for the random initial parameters (and for ``sample`` when
        no ``random_state`` is passed there).

    Attributes
    ----------
    start_prob_ : np.ndarray, shape (n_states,)
        Initial-state distribution (pi).
    trans_mat_ : np.ndarray, shape (n_states, n_states)
        State transition matrix (A); rows sum to 1.
    emission_mat_ : np.ndarray, shape (n_states, n_symbols)
        Emission matrix (B); rows sum to 1.
    log_likelihood_history_ : list[float]
        Total log-likelihood of the training sequences after each
        Baum-Welch iteration (set by :meth:`fit`).
    """

    def __init__(self, n_states: int = 2, n_symbols: int = 2,
                 random_state: int | None = None):
        if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
            raise ValueError("n_states must be an integer")
        if int(n_states) < 1:
            raise ValueError("n_states must be at least 1")
        if isinstance(n_symbols, bool) or not isinstance(n_symbols, (int, np.integer)):
            raise ValueError("n_symbols must be an integer")
        if int(n_symbols) < 1:
            raise ValueError("n_symbols must be at least 1")
        self.n_states = int(n_states)
        self.n_symbols = int(n_symbols)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.start_prob_ = _dirichlet_row(self._rng, self.n_states)
        self.trans_mat_ = np.array(
            [_dirichlet_row(self._rng, self.n_states) for _ in range(self.n_states)]
        )
        self.emission_mat_ = np.array(
            [_dirichlet_row(self._rng, self.n_symbols) for _ in range(self.n_states)]
        )
        self.log_likelihood_history_: list[float] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # parameter helpers
    # ------------------------------------------------------------------
    def set_params(self, start_prob=None, trans_mat=None, emission_mat=None):
        """Set the model parameters (each optional) and return self.

        Every supplied array is checked for the right shape, non-negative
        entries, and rows that sum to ~1 (then re-normalised).
        """
        if start_prob is not None:
            pi = np.asarray(start_prob, dtype=float)
            if pi.shape != (self.n_states,):
                raise ValueError(
                    f"start_prob must have shape ({self.n_states},), got {pi.shape}")
            if np.any(pi < 0):
                raise ValueError("start_prob must not contain negative values")
            if pi.sum() <= 0:
                raise ValueError("start_prob must not be all zeros")
            self.start_prob_ = pi / pi.sum()
        if trans_mat is not None:
            A = np.asarray(trans_mat, dtype=float)
            if A.shape != (self.n_states, self.n_states):
                raise ValueError(
                    f"trans_mat must have shape ({self.n_states}, {self.n_states}), "
                    f"got {A.shape}")
            if np.any(A < 0):
                raise ValueError("trans_mat must not contain negative values")
            if np.any(A.sum(axis=1) <= 0):
                raise ValueError("trans_mat must not have an all-zero row")
            self.trans_mat_ = A / A.sum(axis=1, keepdims=True)
        if emission_mat is not None:
            B = np.asarray(emission_mat, dtype=float)
            if B.shape != (self.n_states, self.n_symbols):
                raise ValueError(
                    f"emission_mat must have shape ({self.n_states}, {self.n_symbols}), "
                    f"got {B.shape}")
            if np.any(B < 0):
                raise ValueError("emission_mat must not contain negative values")
            if np.any(B.sum(axis=1) <= 0):
                raise ValueError("emission_mat must not have an all-zero row")
            self.emission_mat_ = B / B.sum(axis=1, keepdims=True)
        # Parameters are now fully specified, so the model is usable.
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    @staticmethod
    def _validate_sequence(seq) -> np.ndarray:
        seq = np.asarray(seq)
        if seq.ndim != 1:
            raise ValueError("sequence must be a 1D array of symbol ids")
        if seq.size == 0:
            raise ValueError("sequence must not be empty")
        if not np.issubdtype(seq.dtype, np.integer):
            if not np.all(np.equal(np.mod(seq, 1), 0)):
                raise ValueError("sequence symbols must be integers")
            seq = seq.astype(int)
        return seq.astype(int)

    def _validate_sequences(self, sequences):
        """Accept a single 1D array or a list of 1D arrays."""
        arr = np.asarray(sequences)
        if arr.ndim == 1:
            seqs = [self._validate_sequence(arr)]
        else:
            if isinstance(sequences, (list, tuple)) and len(sequences) == 0:
                raise ValueError("sequences must not be empty")
            seqs = [self._validate_sequence(s) for s in sequences]
        for s in seqs:
            if s.min() < 0 or s.max() >= self.n_symbols:
                raise ValueError(
                    f"symbols must be in [0, {self.n_symbols}), "
                    f"got values in [{s.min()}, {s.max()}]")
        return seqs

    # ------------------------------------------------------------------
    # scaled forward-backward
    # ------------------------------------------------------------------
    def _forward(self, seq):
        """Scaled forward pass.

        Returns (alpha, scales) with alpha[t, i] = P(state_i at t | obs up
        to t) and scales[t] = sum of the unscaled alpha row at t, so
        log P(obs) = sum(log(scales)).
        """
        T = len(seq)
        alpha = np.zeros((T, self.n_states))
        scales = np.zeros(T)
        alpha[0] = self.start_prob_ * self.emission_mat_[:, seq[0]]
        scales[0] = alpha[0].sum()
        if scales[0] <= 0:
            scales[0] = _EPS
        alpha[0] /= scales[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.trans_mat_) * self.emission_mat_[:, seq[t]]
            scales[t] = alpha[t].sum()
            if scales[t] <= 0:
                scales[t] = _EPS
            alpha[t] /= scales[t]
        return alpha, scales

    def _backward(self, seq, scales):
        """Scaled backward pass, using the same scales as _forward."""
        T = len(seq)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.trans_mat_ @ (self.emission_mat_[:, seq[t + 1]] * beta[t + 1]))
            beta[t] /= scales[t + 1]
        return beta

    def _forward_backward(self, seq):
        alpha, scales = self._forward(seq)
        beta = self._backward(seq, scales)
        log_lik = float(np.sum(np.log(scales)))
        # gamma[t, i] = P(state_i at t | whole sequence)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)
        # xi[t, i, j] = P(state_i at t, state_j at t+1 | whole sequence)
        T = len(seq)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            numer = (
                alpha[t][:, None]
                * self.trans_mat_
                * (self.emission_mat_[:, seq[t + 1]] * beta[t + 1])[None, :]
            )
            denom = numer.sum()
            xi[t] = numer / denom if denom > 0 else numer
        return alpha, beta, gamma, xi, log_lik

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, sequences, n_iter: int = 50, tol: float = 1e-4):
        """Fit by Baum-Welch (EM). Returns self.

        Parameters
        ----------
        sequences : list of 1D int arrays, or a single 1D int array
            Observed symbol sequences.
        n_iter : int
            Maximum number of EM iterations (>= 1).
        tol : float
            Stop early when the total log-likelihood improves by less
            than ``tol`` between iterations (> 0).
        """
        if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)):
            raise ValueError("n_iter must be an integer")
        if int(n_iter) < 1:
            raise ValueError("n_iter must be at least 1")
        if tol is None or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive number")
        seqs = self._validate_sequences(sequences)
        self.log_likelihood_history_ = []

        prev_ll = None
        for _ in range(int(n_iter)):
            # E-step: accumulate expected counts over all sequences.
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_symbols))
            B_den = np.zeros(self.n_states)
            total_ll = 0.0
            for seq in seqs:
                _, _, gamma, xi, ll = self._forward_backward(seq)
                total_ll += ll
                pi_acc += gamma[0]
                A_num += xi.sum(axis=0)
                A_den += gamma[:-1].sum(axis=0)
                for k in range(self.n_symbols):
                    B_num[:, k] += gamma[seq == k].sum(axis=0)
                B_den += gamma.sum(axis=0)
            # M-step (floor at _EPS so no probability ever becomes exactly 0).
            self.start_prob_ = np.maximum(pi_acc, _EPS)
            self.start_prob_ /= self.start_prob_.sum()
            self.trans_mat_ = np.maximum(A_num, _EPS)
            self.trans_mat_ /= np.maximum(A_den, _EPS)[:, None]
            self.trans_mat_ /= self.trans_mat_.sum(axis=1, keepdims=True)
            self.emission_mat_ = np.maximum(B_num, _EPS)
            self.emission_mat_ /= np.maximum(B_den, _EPS)[:, None]
            self.emission_mat_ /= self.emission_mat_.sum(axis=1, keepdims=True)

            self.log_likelihood_history_.append(total_ll)
            if prev_ll is not None and total_ll - prev_ll < tol:
                break
            prev_ll = total_ll
        self._fitted = True
        return self

    def log_likelihood(self, sequence) -> float:
        """Log probability of ``sequence`` under the fitted model."""
        self._check_fitted()
        seq = self._validate_sequence(sequence)
        if seq.min() < 0 or seq.max() >= self.n_symbols:
            raise ValueError(
                f"symbols must be in [0, {self.n_symbols}), "
                f"got values in [{seq.min()}, {seq.max()}]")
        _, scales = self._forward(seq)
        return float(np.sum(np.log(scales)))

    def score(self, sequences) -> float:
        """Mean log-likelihood per sequence (over a list, or one sequence)."""
        self._check_fitted()
        seqs = self._validate_sequences(sequences)
        return float(np.mean([self.log_likelihood(s) for s in seqs]))

    def forward_backward(self, sequence):
        """Return (alpha, beta, gamma, xi) posterior quantities.

        ``gamma[t, i]`` is P(state i at time t | whole sequence) and its
        rows sum to 1; ``xi[t, i, j]`` is P(state i at t, state j at t+1
        | whole sequence).
        """
        self._check_fitted()
        seq = self._validate_sequence(sequence)
        if seq.min() < 0 or seq.max() >= self.n_symbols:
            raise ValueError(
                f"symbols must be in [0, {self.n_symbols}), "
                f"got values in [{seq.min()}, {seq.max()}]")
        alpha, beta, gamma, xi, _ = self._forward_backward(seq)
        return alpha, beta, gamma, xi

    def predict(self, sequence) -> np.ndarray:
        """Viterbi decoding: most likely hidden-state sequence."""
        self._check_fitted()
        seq = self._validate_sequence(sequence)
        if seq.min() < 0 or seq.max() >= self.n_symbols:
            raise ValueError(
                f"symbols must be in [0, {self.n_symbols}), "
                f"got values in [{seq.min()}, {seq.max()}]")
        T = len(seq)
        log_pi = np.log(np.maximum(self.start_prob_, _EPS))
        log_A = np.log(np.maximum(self.trans_mat_, _EPS))
        log_B = np.log(np.maximum(self.emission_mat_, _EPS))
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = log_pi + log_B[:, seq[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_A  # [prev, curr]
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.n_states)] + log_B[:, seq[t]]
        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def sample(self, n_steps: int, random_state: int | None = None):
        """Generate (observations, hidden_states) of length ``n_steps``.

        Uses the current parameters (no fitting needed), so this also
        works on a freshly constructed model to generate training data.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
            raise ValueError("n_steps must be an integer")
        if int(n_steps) < 1:
            raise ValueError("n_steps must be at least 1")
        rng = (np.random.default_rng(random_state)
               if random_state is not None else self._rng)
        states = np.zeros(int(n_steps), dtype=int)
        obs = np.zeros(int(n_steps), dtype=int)
        states[0] = rng.choice(self.n_states, p=self.start_prob_)
        obs[0] = rng.choice(self.n_symbols, p=self.emission_mat_[states[0]])
        for t in range(1, int(n_steps)):
            states[t] = rng.choice(self.n_states, p=self.trans_mat_[states[t - 1]])
            obs[t] = rng.choice(self.n_symbols, p=self.emission_mat_[states[t]])
        return obs, states
