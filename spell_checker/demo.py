"""Demo: correct common misspellings with the edit-distance spell checker."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_checker.model import SpellChecker

MISSPELLED = ["speling", "recieve", "teh", "wrting", "adress", "seperate"]
SENTENCE = "Teh students practise speling and hope to recieve good news."


def main():
    checker = SpellChecker()
    print(f"Vocabulary: {len(checker.vocab)} words from "
          f"{checker.total_words} corpus tokens")
    print(f"Most common words: the={checker.frequency('the')}, "
          f"and={checker.frequency('and')}, "
          f"spelling={checker.frequency('spelling')}, "
          f"receive={checker.frequency('receive')}")
    print("\nSingle-word corrections:")
    for word in MISSPELLED:
        cands = sorted(checker.candidates(word),
                       key=lambda w: -checker.frequency(w))[:3]
        print(f"  {word!r:>10} -> {checker.correct(word)!r} "
              f"(candidates: {cands})")
    print(f"\nSentence correction:\n  before: {SENTENCE}\n  after:  "
          f"{checker.correct_sentence(SENTENCE)}")
    print(f"\nKnown words stay unchanged: "
          f"{checker.correct('spelling')!r}, {checker.correct('the')!r}")
    print(f"Unknown word with no close match stays as-is: "
          f"{checker.correct('zxqv')!r}")


if __name__ == "__main__":
    main()
