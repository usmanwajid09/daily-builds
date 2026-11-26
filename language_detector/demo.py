"""Demo: identify the language of one sentence per language with trigrams."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language_detector.model import LANGUAGES, LanguageDetector

# Held-out demo sentences (none of these appear in the training texts).
DEMO_SENTENCES = {
    "en": "The sun rises early in the summer and the birds sing in the trees behind our house every morning.",
    "fr": "Le soleil se lève tôt en été et les oiseaux chantent dans les arbres derrière notre maison chaque matin.",
    "es": "El sol sale temprano en verano y los pájaros cantan en los árboles detrás de nuestra casa cada mañana.",
    "de": "Die Sonne geht im Sommer früh auf und die Vögel singen jeden Morgen in den Bäumen hinter unserem Haus.",
}


def main():
    detector = LanguageDetector()
    print(f"Languages: {', '.join(f'{code} ({name})' for code, name in LANGUAGES.items())}")
    print(f"Profile size: top {detector.top_n} trigrams per language")
    correct = 0
    for code, sentence in DEMO_SENTENCES.items():
        predicted = detector.classify(sentence)
        dists = detector.distances(sentence)
        scores = detector.scores(sentence)
        ok = predicted == code
        correct += ok
        ranked = sorted(dists, key=dists.get)
        print(f"\n[{LANGUAGES[code]}] {sentence}")
        print(f"  -> {predicted} ({detector.language_name(predicted)})"
              f" {'OK' if ok else 'WRONG'}; closest: "
              + ", ".join(f"{c}={dists[c]}" for c in ranked))
        print(f"     scores: " + ", ".join(f"{c}={scores[c]:.2f}" for c in ranked))
    print(f"\nDemo accuracy: {correct}/{len(DEMO_SENTENCES)}")
    print(f"Short text: 'hello' -> {detector.classify('hello')} (too little evidence)")
    print(f"Empty text raises ValueError: ", end="")
    try:
        detector.classify("")
    except ValueError as exc:
        print(exc)


if __name__ == "__main__":
    main()
