"""Demo: classify toy sports vs tech documents with Naive Bayes."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from naive_bayes_text.model import MultinomialNB

TRAIN = [
    ("the team won the match with a great goal", "sports"),
    ("the striker scored two goals in the final", "sports"),
    ("the championship game ended in a draw", "sports"),
    ("the player received a red card", "sports"),
    ("the coach praised the team after the win", "sports"),
    ("the new phone has a faster processor", "tech"),
    ("the software update fixes the battery bug", "tech"),
    ("the laptop screen and keyboard are excellent", "tech"),
    ("the app crashes when opening the camera", "tech"),
    ("the developer released a new version today", "tech"),
]

TEST = [
    ("the goalkeeper saved the penalty", "sports"),
    ("the match was decided in extra time", "sports"),
    ("the new tablet has a bright display", "tech"),
    ("the update improves the wifi connection", "tech"),
]


def main():
    clf = MultinomialNB(alpha=1.0).fit([t for t, _ in TRAIN], [y for _, y in TRAIN])
    texts, labels = [t for t, _ in TEST], [y for _, y in TEST]
    preds = clf.predict(texts)
    probs = clf.predict_proba(texts)
    print(f"Vocabulary size: {len(clf.vocab_)} words, classes: {list(clf.classes_)}")
    print(f"Training accuracy: {clf.score([t for t, _ in TRAIN], [y for _, y in TRAIN]):.2f}")
    for text, true, pred, prob in zip(texts, labels, preds, probs):
        print(f"  '{text}' -> {pred} (true: {true}, P={prob.max():.2f})")
    print(f"Test accuracy: {float(np.mean(preds == np.array(labels))):.2f}")
    print(f"Unseen words handled: {clf.predict(['zebra quantum foxtrot'])[0]} (no crash on unknown words)")


if __name__ == "__main__":
    main()
