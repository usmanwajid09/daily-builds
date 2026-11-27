"""Demo: extract keyphrases from a paragraph about machine learning."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyword_extractor.model import KeywordExtractor, extract_keywords

TEXT = (
    "Machine learning models learn patterns from training data. "
    "Deep neural networks use many layers of artificial neurons to "
    "recognise complex patterns in images and text. "
    "Training a neural network requires large datasets and powerful "
    "graphics processors. "
    "Supervised learning trains models on labelled examples, while "
    "unsupervised learning discovers hidden structure without labels."
)


def main():
    extractor = KeywordExtractor(top_n=8)
    print("Top keyphrases:")
    for phrase, score in extractor.extract(TEXT):
        print(f"  {score:6.2f}  {phrase}")
    print()
    print("Top 5 single words (min_words=1, max_words=1):")
    for phrase, score in extract_keywords(TEXT, top_n=5, min_words=1, max_words=1):
        print(f"  {score:6.2f}  {phrase}")


if __name__ == "__main__":
    main()
