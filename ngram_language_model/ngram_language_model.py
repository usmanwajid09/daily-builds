"""N-gram language model (orders 1-3) with add-k smoothing and perplexity.

Standard library only.

Tokenisation: text is lowercased and split into lowercase alphanumeric
words (``re.findall(r"[a-z0-9]+", text.lower())``). Punctuation is dropped,
so "The Cat!" becomes ["the", "cat"].

Each training sentence is padded with ``<s>`` start tokens (``order - 1``
of them) and a single ``</s>`` end token, and n-gram counts are built for
every order from 1 up to ``order``.

Conditional probabilities use add-k smoothing::

    P(w | context) = (count(context, w) + k) / (count(context) + k * V)

where ``V`` is the vocabulary size (all distinct training words plus
``</s>``). Because of the smoothing, every word in the vocabulary -- and
any unseen word handled through the vocabulary -- gets a non-zero
probability, so perplexity is always finite.
"""

import math
import random
import re
from collections import Counter, defaultdict

__all__ = ["NGramLanguageModel"]

_START = "<s>"
_END = "</s>"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    """Lowercase alphanumeric word tokens of *text*."""
    if not isinstance(text, str):
        raise TypeError("expected a string, got %s" % type(text).__name__)
    return _TOKEN_RE.findall(text.lower())


class NGramLanguageModel:
    """An n-gram language model with add-k smoothing.

    Parameters
    ----------
    order : int, default 2
        The n-gram order, 1 (unigram) to 3 (trigram).
    k : float, default 0.1
        Add-k smoothing constant; must be > 0.
    """

    def __init__(self, order=2, k=0.1):
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError("order must be an int, got %s" % type(order).__name__)
        if order < 1 or order > 3:
            raise ValueError("order must be between 1 and 3, got %r" % (order,))
        if not isinstance(k, (int, float)) or isinstance(k, bool):
            raise TypeError("k must be a number, got %s" % type(k).__name__)
        if k <= 0:
            raise ValueError("k must be > 0, got %r" % (k,))
        self.order = order
        self.k = float(k)
        # counts[n][context_tuple][word] -> int, for n in 1..order.
        # For n == 1 the context is the empty tuple.
        self.counts = {}
        # context_counts[n][context_tuple] -> total tokens seen after context
        self.context_counts = {}
        self.vocab = set()
        self.vocab_size = 0
        self.total_tokens = 0
        self._fitted = False

    # -- fitting ---------------------------------------------------------
    def fit(self, sentences):
        """Build n-gram counts from *sentences*, a list of strings.

        Returns self, so calls can be chained.
        """
        if isinstance(sentences, str) or not hasattr(sentences, "__iter__"):
            raise TypeError("sentences must be a list (or other iterable) of strings")
        sentences = list(sentences)
        if not sentences:
            raise ValueError("cannot fit on an empty corpus")
        tokenised = []
        for sentence in sentences:
            tokens = _tokenize(sentence)
            if not tokens:
                raise ValueError("empty sentence in corpus: %r" % (sentence,))
            tokenised.append(tokens)

        self.counts = {n: defaultdict(Counter) for n in range(1, self.order + 1)}
        self.context_counts = {n: Counter() for n in range(1, self.order + 1)}
        self.vocab = {_END}
        self.total_tokens = 0
        for tokens in tokenised:
            self.vocab.update(tokens)
            self.total_tokens += len(tokens)
            padded = [_START] * (self.order - 1) + tokens + [_END]
            for n in range(1, self.order + 1):
                for i in range(len(padded) - n + 1):
                    gram = padded[i:i + n]
                    context = tuple(gram[:-1])
                    word = gram[-1]
                    # <s> is a context only, never a predicted word: skip
                    # grams ending in <s> so every conditional distribution
                    # over the vocabulary sums to exactly 1.
                    if word == _START:
                        continue
                    self.counts[n][context][word] += 1
                    self.context_counts[n][context] += 1
        self.vocab_size = len(self.vocab)
        self._fitted = True
        return self

    # -- helpers ---------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("model has not been fitted yet; call fit() first")

    def _context_for(self, history):
        """Last (order-1) tokens of *history* as a context tuple."""
        if self.order == 1:
            return ()
        return tuple(history[-(self.order - 1):])

    def _padded_tokens(self, sentence):
        tokens = _tokenize(sentence)
        if not tokens:
            raise ValueError("empty sentence: %r" % (sentence,))
        return tokens

    # -- probabilities ---------------------------------------------------
    def cond_prob(self, word, context=()):
        """P(word | context) with add-k smoothing.

        *word* is a single word (lowercased automatically); *context* is a
        sequence of preceding words -- only the last ``order - 1`` are used.
        Unknown words get a small non-zero probability via smoothing.
        """
        self._check_fitted()
        if not isinstance(word, str) or not word:
            raise ValueError("word must be a non-empty string")
        word = word.lower()
        context = tuple(w.lower() for w in context)
        context = self._context_for(context)
        n = len(context) + 1
        count_cw = self.counts[n][context].get(word, 0)
        count_c = self.context_counts[n][context]
        return (count_cw + self.k) / (count_c + self.k * self.vocab_size)

    def log_prob(self, sentence):
        """Natural-log probability of *sentence* (including its </s>)."""
        self._check_fitted()
        tokens = self._padded_tokens(sentence)
        history = [_START] * (self.order - 1)
        total = 0.0
        for word in tokens + [_END]:
            total += math.log(self.cond_prob(word, history))
            history.append(word)
        return total

    def prob(self, sentence):
        """Probability of *sentence* (including its </s>)."""
        return math.exp(self.log_prob(sentence))

    def perplexity(self, sentences):
        """Perplexity of *sentences*: exp(-mean log prob per token).

        The token count includes one </s> token per sentence. Lower is
        better: text like the training corpus scores much lower than
        unrelated or garbled text.
        """
        self._check_fitted()
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        if not sentences:
            raise ValueError("perplexity needs at least one sentence")
        total_log_prob = 0.0
        total_tokens = 0
        for sentence in sentences:
            tokens = self._padded_tokens(sentence)
            total_log_prob += self.log_prob(sentence)
            total_tokens += len(tokens) + 1  # +1 for </s>
        return math.exp(-total_log_prob / total_tokens)

    # -- generation ------------------------------------------------------
    def generate(self, seed="", max_len=20, sample=False, random_state=None):
        """Generate text continuing *seed*.

        Greedy argmax decoding by default (deterministic: ties are broken
        by taking the alphabetically first word). With ``sample=True``,
        each next word is drawn from the conditional distribution using
        ``random_state`` as the seed (an int, or a ``random.Random``).
        Unknown seed words are ignored; a seed with no known words starts
        from the ``<s>`` padding. Generation stops at ``</s>`` (only when
        it is strictly more likely than the best real word) or after
        ``max_len`` words. The returned string starts with the seed words
        that were recognised (or is the generated text alone if the seed
        was empty/unknown).
        """
        self._check_fitted()
        if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 1:
            raise ValueError("max_len must be a positive int, got %r" % (max_len,))
        seed_tokens = _tokenize(seed) if seed else []
        known_seed = [w for w in seed_tokens if w in self.vocab]
        history = ([_START] * (self.order - 1) + known_seed)[-(self.order - 1):] \
            if self.order > 1 else []
        rng = None
        if sample:
            rng = random_state if isinstance(random_state, random.Random) \
                else random.Random(random_state)
        generated = []
        for _ in range(max_len):
            context = self._context_for(history)
            # Greedy argmax over real words; </s> only ends the sentence
            # when it is strictly more likely than the best real word,
            # so a tie with </s> does not cut a sentence short.
            word_candidates = sorted(self.vocab - {_END})
            if sample:
                weights = [self.cond_prob(w, context) for w in word_candidates]
                word = rng.choices(word_candidates, weights=weights, k=1)[0]
                if rng.random() < self.cond_prob(_END, context):
                    break
            else:
                # Greedy argmax; sorted candidates make ties deterministic
                # (lowest probability-negated, then alphabetically first).
                word = min(word_candidates,
                           key=lambda w: (-self.cond_prob(w, context), w))
                if self.cond_prob(_END, context) > self.cond_prob(word, context):
                    break
            generated.append(word)
            history.append(word)
        return " ".join(known_seed + generated)
