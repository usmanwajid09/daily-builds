"""Demo: score the sentiment of a few sample reviews."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer.model import SentimentAnalyzer, analyze

TEXTS = [
    "I love this phone. The camera is amazing and the battery lasts all day.",
    "This was a terrible experience. The food was awful and the service was slow.",
    "The package arrived on time. It works as described.",
    "I do not like this at all. It is not good and very disappointing.",
    "Absolutely fantastic! I am extremely happy with the quality.",
]


def main():
    analyzer = SentimentAnalyzer()
    for text in TEXTS:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"  Score: {result['score']:.3f}  Label: {result['label']}  "
              f"(sentiment words: {result['sentiment_words']})")
        for s in result["sentences"]:
            print(f"    [{s['label']:>8}] {s['score']:+.3f}  {s['text']}")
        print()
    mixed = analyze("I love the design but I hate the price.")
    print(f"Mixed review -> score {mixed['score']:.3f}, label {mixed['label']}")


if __name__ == "__main__":
    main()
