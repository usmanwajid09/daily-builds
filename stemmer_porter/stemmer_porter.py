"""Porter stemmer (Porter, 1980) implemented from scratch.

Reduces English words to their stem by applying the classic five-step
suffix-stripping algorithm, exactly as published by Martin Porter in
"An algorithm for suffix stripping" (Program, 1980) and documented at
snowball.tartarus.org/algorithms/porter/stemmer.html.

The algorithm works on the *measure* ``m`` of a word part: writing a word
as ``[C](VC)^m[V]`` (optional consonants, then ``m`` vowel-consonant
sequences, then optional vowels), ``m`` counts the VC sequences.  Each
step applies at most one rule -- the longest matching suffix whose
condition (usually ``m > 0`` or ``m > 1`` on the stem) holds.

Standard library only; no third-party dependencies.

Example:
    >>> from stemmer_porter import stem
    >>> stem("caresses")
    'caress'
    >>> stem("relational")
    'relat'
    >>> stem("vietnamization")
    'vietnam'
"""

__all__ = ["PorterStemmer", "stem", "stem_text"]

_VOWELS = "aeiou"


def _is_consonant(word, i):
    """True if ``word[i]`` is a consonant (Porter's definition).

    A letter other than A, E, I, O, U is a consonant, except Y, which is
    a consonant only at the start of a word or after a vowel -- otherwise
    (after a consonant) Y counts as a vowel.
    """
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem):
    """Measure ``m`` of a word part: the number of VC sequences in it."""
    n = 0
    i = 0
    length = len(stem)
    # Skip any leading consonants.
    while i < length and _is_consonant(stem, i):
        i += 1
    while i < length:
        # Skip a vowel sequence; if a consonant sequence follows, that
        # is one more VC pair.
        while i < length and not _is_consonant(stem, i):
            i += 1
        if i < length and _is_consonant(stem, i):
            n += 1
            while i < length and _is_consonant(stem, i):
                i += 1
    return n


def _contains_vowel(stem):
    """True if the stem contains a vowel (``*v*``)."""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word):
    """True if the word ends with a double consonant (``*d``)."""
    return (
        len(word) >= 2
        and word[-1] == word[-2]
        and _is_consonant(word, len(word) - 1)
    )


def _ends_cvc(word):
    """True if the word ends CVC, with the final C not W, X or Y (``*o``)."""
    return (
        len(word) >= 3
        and _is_consonant(word, len(word) - 1)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 3)
        and word[-1] not in "wxy"
    )


# Step 2 rules: (suffix, replacement), applied when m(stem) > 0.
# Ordered longest suffix first so the longest match wins.
_STEP2 = [
    ("ational", "ate"),
    ("ization", "ize"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("tional", "tion"),
    ("biliti", "ble"),
    ("alism", "al"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("entli", "ent"),
    ("ousli", "ous"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("eli", "e"),
]

# Step 3 rules: (suffix, replacement), applied when m(stem) > 0.
_STEP3 = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ness", ""),
    ("ful", ""),
]

# Step 4 suffixes removed when m(stem) > 1 (longest first).  "ion" is
# handled separately: it is removed only when the stem ends in S or T.
_STEP4 = [
    "ement",
    "ance",
    "ence",
    "able",
    "ible",
    "ment",
    "ant",
    "ent",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
    "ion",
    "al",
    "er",
    "ic",
    "ou",
]


class PorterStemmer:
    """The classic Porter stemmer (Porter, 1980).

    Create one instance and reuse it; :meth:`stem` is stateless, so a
    single shared instance is fine.  Module-level :func:`stem` and
    :func:`stem_text` use a shared instance.
    """

    def stem(self, word):
        """Return the Porter stem of ``word`` (lowercased).

        Raises:
            TypeError: if ``word`` is not a string.
            ValueError: if ``word`` is empty or only whitespace.
        """
        if not isinstance(word, str):
            raise TypeError(
                "word must be a string, got %s" % type(word).__name__
            )
        word = word.strip().lower()
        if not word:
            raise ValueError("word must not be empty")
        # As in Porter's reference implementation, words of length 1 or 2
        # pass through unchanged.
        if len(word) <= 2:
            return word
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        return word

    def stem_text(self, text):
        """Stem every whitespace-separated word in ``text``, in order.

        Returns the stems joined by single spaces.  An empty string stems
        to an empty string.

        Raises:
            TypeError: if ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string, got %s" % type(text).__name__
            )
        return " ".join(self.stem(word) for word in text.split())

    # -- Step 1a: plurals -------------------------------------------------
    @staticmethod
    def _step1a(word):
        if word.endswith("sses"):
            return word[:-2]  # caresses -> caress
        if word.endswith("ies"):
            return word[:-3] + "i"  # ponies -> poni
        if word.endswith("ss"):
            return word  # caress -> caress
        if word.endswith("s"):
            return word[:-1]  # cats -> cat
        return word

    # -- Step 1b: -ed / -ing ----------------------------------------------
    @staticmethod
    def _step1b(word):
        if word.endswith("eed"):
            # (m>0) EED -> EE ; "feed" stays, "agreed" -> "agree".
            if _measure(word[:-3]) > 0:
                return word[:-3] + "ee"
            return word
        stripped = None
        if word.endswith("ed") and _contains_vowel(word[:-2]):
            stripped = word[:-2]  # plastered -> plaster
        elif word.endswith("ing") and _contains_vowel(word[:-3]):
            stripped = word[:-3]  # motoring -> motor
        if stripped is None:
            return word
        return PorterStemmer._step1b_extra(stripped)

    @staticmethod
    def _step1b_extra(word):
        # Run only when an -ed / -ing was removed; exactly one rule fires.
        if word.endswith("at"):
            return word + "e"  # conflat(ed) -> conflate
        if word.endswith("bl"):
            return word + "e"  # troubl(ed) -> trouble
        if word.endswith("iz"):
            return word + "e"  # siz(ed) -> size
        if _ends_double_consonant(word) and word[-1] not in "lsz":
            return word[:-1]  # hopp(ing) -> hop; fall(ing) stays fall
        if _measure(word) == 1 and _ends_cvc(word):
            return word + "e"  # fil(ing) -> file; fail(ing) stays fail
        return word

    # -- Step 1c: terminal y -> i -----------------------------------------
    @staticmethod
    def _step1c(word):
        if word.endswith("y") and _contains_vowel(word[:-1]):
            return word[:-1] + "i"  # happy -> happi; sky stays sky
        return word

    # -- Step 2: derivational suffixes (m>0) ------------------------------
    @staticmethod
    def _step2(word):
        for suffix, replacement in _STEP2:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if _measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    # -- Step 3: more derivational suffixes (m>0) -------------------------
    @staticmethod
    def _step3(word):
        for suffix, replacement in _STEP3:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if _measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    # -- Step 4: remove remaining derivational suffixes (m>1) ------------
    @staticmethod
    def _step4(word):
        for suffix in _STEP4:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if suffix == "ion":
                # (m>1 and (*S or *T)) ION -> ''
                if _measure(stem) > 1 and stem.endswith(("s", "t")):
                    return stem
                return word
            if _measure(stem) > 1:
                return stem
            return word
        return word

    # -- Step 5: tidy up ---------------------------------------------------
    @staticmethod
    def _step5(word):
        # Step 5a: drop a final E.
        if word.endswith("e"):
            stem = word[:-1]
            m = _measure(stem)
            if m > 1 or (m == 1 and not _ends_cvc(stem)):
                word = stem  # probate -> probat; cease -> ceas
                # rate stays rate: m == 1 and *o, so the E is kept.
        # Step 5b: drop one of a final double L when m > 1.
        if (
            _measure(word) > 1
            and _ends_double_consonant(word)
            and word.endswith("l")
        ):
            word = word[:-1]  # controll -> control; roll stays roll
        return word


_DEFAULT_STEMMER = PorterStemmer()


def stem(word):
    """Return the Porter stem of ``word`` (lowercased).

    Raises:
        TypeError: if ``word`` is not a string.
        ValueError: if ``word`` is empty or only whitespace.
    """
    return _DEFAULT_STEMMER.stem(word)


def stem_text(text):
    """Stem every whitespace-separated word in ``text``, in order.

    Returns the stems joined by single spaces.  An empty string stems
    to an empty string.

    Raises:
        TypeError: if ``text`` is not a string.
    """
    return _DEFAULT_STEMMER.stem_text(text)
