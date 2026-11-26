"""Edit-distance spell checker, from scratch (Norvig-style).

A vocabulary with word frequencies is built from a small built-in
corpus (or a custom corpus supplied by the caller). To correct a word,
all candidates within edit distance 1 (deletes, transposes, replaces,
inserts) are generated; if none is in the vocabulary, candidates at
edit distance 2 are tried. Candidates are ranked by frequency, so the
most common plausible word wins. Only the standard library is used.
"""

import re
from collections import Counter

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_WORD_RE = re.compile(r"[a-z]+")

# Small built-in corpus: ordinary English prose, repeated common words
# give realistic frequencies ("the" is by far the most common word).
# The target demo words appear with healthy counts so their misspellings
# resolve to them: spelling, receive, the.
_DEFAULT_CORPUS = """
the quick brown fox jumps over the lazy dog and the dog runs to the river
the cat sits on the mat in the house by the garden and the bird sings in
the morning the sun rises over the hills and the people walk to the market
to buy bread and cheese and fruit for the family the children play in the
school and the teacher helps the students with spelling and reading every
day the students learn spelling and writing and the teacher checks the
spelling of every word in the book the spelling test is on friday and the
students must practise spelling carefully i will receive the letter from
my friend tomorrow please receive the package when it arrives we receive
many emails every day and i hope to receive good news about the job i
received the message yesterday and she received the book last week the
weather is fine today and the rain will come later in the week the train
leaves the station at eight and the bus stops near the library the doctor
said the medicine will help and the nurse will visit in the evening we
went to the cinema to watch a film and then we ate dinner at home the
computer is on the desk and the phone is in the bag the music plays and
the people dance and sing together in the evening the garden grows flowers
and the trees give shade in the summer the water is cold and the food is
hot and the coffee is strong the children love the story and the teacher
reads the story to the class the book is on the table and the pen is in
the drawer the house is big and the room is clean the car is fast and the
road is long the time is short and the work is hard but the result is good
the address is on the letter and the house is separate from the school
hello world how are you today the cats play in the garden and the dogs run
in the park the birds sing and the children laugh and play all day
"""



def _words_from_text(text):
    """Lowercase alphabetic tokens from a string."""
    return _WORD_RE.findall(text.lower())


class SpellChecker:
    """Edit-distance spell checker with a frequency-ranked vocabulary.

    Parameters
    ----------
    corpus : str, dict, or None
        Text to build the vocabulary from (word frequencies are counted),
        or a mapping of word -> count. ``None`` uses the built-in corpus.
    max_edit_distance : int
        Furthest edit distance searched for candidates (1 or 2).
        Must be 1 or 2.
    """

    def __init__(self, corpus=None, max_edit_distance=2):
        if max_edit_distance not in (1, 2):
            raise ValueError("max_edit_distance must be 1 or 2")
        self.max_edit_distance = max_edit_distance
        if corpus is None:
            counts = Counter(_words_from_text(_DEFAULT_CORPUS))
        elif isinstance(corpus, dict):
            counts = Counter()
            for word, count in corpus.items():
                if not isinstance(word, str) or not word.isalpha():
                    raise ValueError(
                        f"corpus keys must be alphabetic words, got {word!r}"
                    )
                if not isinstance(count, int) or isinstance(count, bool) \
                        or count <= 0:
                    raise ValueError(
                        f"corpus counts must be positive integers, "
                        f"got {count!r} for word {word!r}"
                    )
                counts[word.lower()] += count
        elif isinstance(corpus, str):
            counts = Counter(_words_from_text(corpus))
        else:
            raise TypeError(
                "corpus must be a string, a dict of word -> count, or None"
            )
        if not counts:
            raise ValueError("corpus is empty: no words to build a vocabulary from")
        self.word_counts = dict(counts)
        self.vocab = set(counts)
        self.total_words = sum(counts.values())

    # -- candidate generation -------------------------------------------
    @staticmethod
    def edits1(word):
        """All strings one edit away: deletes, transposes, replaces, inserts."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
        inserts = [a + c + b for a, b in splits for c in _LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All strings two edits away (edits of edits), as a generator."""
        for e1 in self.edits1(word):
            yield from self.edits1(e1)

    def known(self, words):
        """The subset of `words` that appear in the vocabulary."""
        return {w for w in words if w in self.vocab}

    def candidates(self, word):
        """Best candidate set for `word`: itself if known, else the
        closest known words at edit distance 1, then 2 (if enabled)."""
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        word = word.lower()
        if not word:
            raise ValueError("word must not be empty")
        if not word.isalpha():
            raise ValueError(f"word must contain only letters, got {word!r}")
        if word in self.vocab:
            return {word}
        known1 = self.known(self.edits1(word))
        if known1:
            return known1
        if self.max_edit_distance == 2:
            known2 = self.known(self.edits2(word))
            if known2:
                return known2
        return {word}

    def correct(self, word):
        """Most likely correction of `word` (unchanged if already known
        or if no candidate exists). Preserves a leading capital."""
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        if word == "":
            return ""
        if not word.isalpha():
            raise ValueError(f"word must contain only letters, got {word!r}")
        lower = word.lower()
        best = max(self.candidates(lower),
                   key=lambda w: self.word_counts.get(w, 0))
        if word[0].isupper():
            best = best.capitalize()
        return best

    def correct_sentence(self, text):
        """Correct every word in `text`, keeping punctuation, spacing
        and capitalisation of the first letter of each word."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if text == "":
            return ""

        def _fix(match):
            return self.correct(match.group(0))

        return re.sub(r"[A-Za-z]+", _fix, text)

    def frequency(self, word):
        """How many times `word` occurs in the corpus (0 if unknown)."""
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        return self.word_counts.get(word.lower(), 0)
