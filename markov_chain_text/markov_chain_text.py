"""Order-N Markov chain text generator (word level), standard library only.

Learns, for every sequence of ``order`` words, which words can follow it,
then generates new text by walking the chain. Generation is deterministic
when a ``random_state`` seed is given.
"""

import random

__all__ = ["MarkovChainText"]


class MarkovChainText:
    """Word-level Markov chain of a fixed order (default 2)."""

    def __init__(self, order=2, random_state=None):
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError(f"order must be an integer >= 1, got {order!r}")
        self.order = order
        self._rng = random.Random(random_state)
        self._chain = {}
        self._starts = []
        self._fitted = False

    def fit(self, text):
        """Learn transition counts from ``text`` (a non-empty string).

        Returns ``self``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__!r}")
        tokens = text.split()
        if len(tokens) == 0:
            raise ValueError("text must not be empty or blank")
        if len(tokens) <= self.order:
            raise ValueError(
                f"text has {len(tokens)} words, need more than order={self.order} "
                "to learn any transition"
            )
        chain = {}
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            nxt = tokens[i + self.order]
            chain.setdefault(state, []).append(nxt)
        self._chain = chain
        self._starts = [tuple(tokens[i : i + self.order]) for i in range(0, 1)]
        self._fitted = True
        return self

    @property
    def n_states(self):
        """Number of distinct states learned."""
        return len(self._chain)

    def generate(self, length=20, seed_words=None, random_state=None):
        """Generate ``length`` words of text.

        ``seed_words`` (optional) must be exactly ``order`` words that form
        a known state; otherwise generation starts from the first state of
        the training text. ``random_state`` overrides the instance seed for
        this call only.
        """
        if not self._fitted:
            raise RuntimeError("chain is not fitted yet; call fit() first")
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError(f"length must be an integer >= 1, got {length!r}")
        rng = random.Random(random_state) if random_state is not None else self._rng
        if seed_words is None:
            state = self._starts[0]
        else:
            if isinstance(seed_words, str):
                seed_words = seed_words.split()
            try:
                seed_words = list(seed_words)
            except TypeError:
                raise TypeError("seed_words must be a string or a sequence of words")
            if len(seed_words) != self.order:
                raise ValueError(
                    f"seed_words must contain exactly {self.order} words, "
                    f"got {len(seed_words)}"
                )
            state = tuple(seed_words)
            if state not in self._chain:
                raise ValueError(
                    f"seed words {list(state)!r} never occur in the training text"
                )
        words = list(state)
        while len(words) < length:
            options = self._chain.get(state)
            if not options:
                # Dead end: restart from the beginning state.
                state = self._starts[0]
                words.extend(state)
                continue
            nxt = rng.choice(options)
            words.append(nxt)
            state = tuple(words[-self.order :])
        return " ".join(words[:length])
