"""Demo: Porter stemmer on varied words and a sentence.

Stems a list of words covering plurals, -ing / -ed forms, -ational,
-ness and other derivational suffixes, then stems a short sentence and
shows how stemming collapses distinct raw words onto fewer stems.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stemmer_porter import stem, stem_text

WORDS = [
    "caresses",
    "ponies",
    "agreed",
    "motoring",
    "hopping",
    "relational",
    "conditional",
    "vietnamization",
    "hopefulness",
    "communism",
    "bowdlerize",
    "happiness",
]

SENTENCE = "Connecting the connections and the connected systems relationally"


def main():
    print("Porter stemmer demo")
    print("===================")
    print()
    print("Word -> stem:")
    for word in WORDS:
        print(f"  {word:<16} -> {stem(word)}")
    print()
    print(f"Sentence: {SENTENCE!r}")
    stemmed = stem_text(SENTENCE)
    print(f"Stemmed:  {stemmed!r}")
    raw_words = SENTENCE.split()
    stem_words = stemmed.split()
    print(
        f"Distinct raw words: {len(set(raw_words))} "
        f"-> distinct stems: {len(set(stem_words))}"
    )


if __name__ == "__main__":
    main()
