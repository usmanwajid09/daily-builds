"""Demo: summarize an 8-sentence paragraph on photosynthesis to 2 sentences."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_summarizer_extractive.model import (
    ExtractiveSummarizer,
    score_sentences,
    split_sentences,
    summarize,
)

PARAGRAPH = (
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
    "Plants capture sunlight with chlorophyll in their leaves. "
    "During photosynthesis, plants absorb carbon dioxide from the air and water from the soil. "
    "The captured sunlight drives chemical reactions that produce glucose and release oxygen. "
    "Glucose provides the energy plants need to grow and reproduce. "
    "Farmers rely on photosynthesis because healthy crops depend on strong sunlight and water. "
    "Without photosynthesis, nearly all life on Earth would lose its main source of food and oxygen. "
    "Scientists study photosynthesis to improve crop yields and fight climate change."
)

KEY_TERMS = ["photosynthesis", "plants", "sunlight"]


def main():
    sentences = split_sentences(PARAGRAPH)
    scores = score_sentences(sentences)
    print(f"Original paragraph: {len(sentences)} sentences")
    for i, (sentence, score) in enumerate(zip(sentences, scores), start=1):
        print(f"  [{i}] score={score:.4f}  {sentence}")
    summary = summarize(PARAGRAPH, n_sentences=2)
    print(f"\nSummary ({len(summary)} sentences, original order kept):")
    for sentence in summary:
        print(f"  - {sentence}")
    joined = " ".join(s.lower() for s in summary)
    missing = [term for term in KEY_TERMS if term not in joined]
    print(f"\nKey terms checked: {KEY_TERMS}")
    print(f"Key terms present in summary: {[t for t in KEY_TERMS if t in joined]}")
    assert not missing, f"summary is missing key terms: {missing}"
    print("All key terms present in the summary.")

    # Same result through the class API.
    clf = ExtractiveSummarizer(n_sentences=2)
    assert clf.summarize(PARAGRAPH) == summary
    print("ExtractiveSummarizer class API agrees with summarize().")


if __name__ == "__main__":
    main()
