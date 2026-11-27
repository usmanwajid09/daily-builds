"""Demo: score the readability of a simple and a complex paragraph."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from readability_scorer.model import ReadabilityScorer, score_all

SIMPLE = (
    "The cat sat on the mat. It was a warm day. "
    "The sun shone in the sky. The cat was glad. "
    "It slept all day long."
)

COMPLEX = (
    "The implementation of comprehensive regulatory frameworks "
    "necessitates the consideration of multifaceted socioeconomic "
    "variables. Consequently, policymakers must evaluate the "
    "interdependencies between institutional structures and "
    "macroeconomic indicators. Furthermore, the quantification of "
    "long-term consequences requires sophisticated analytical "
    "methodologies and extensive empirical investigation."
)


def _show(title, text):
    result = score_all(text)
    print(f"{title}:")
    print(f"  Words: {result['words']}, sentences: {result['sentences']}, "
          f"syllables: {result['syllables']}")
    print(f"  Flesch Reading Ease: {result['flesch_reading_ease']} "
          f"({result['ease_label']})")
    print(f"  Flesch-Kincaid Grade: {result['flesch_kincaid_grade']}")
    print(f"  Gunning Fog: {result['gunning_fog']}")
    print(f"  SMOG: {result['smog']}")
    print()


def main():
    scorer = ReadabilityScorer()
    _show("Simple text", SIMPLE)
    _show("Complex text", COMPLEX)
    print("Scorer class agrees:",
          scorer.score(SIMPLE)["flesch_reading_ease"] ==
          score_all(SIMPLE)["flesch_reading_ease"])


if __name__ == "__main__":
    main()
