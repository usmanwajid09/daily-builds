"""Demo: logistic-regression text classifier over TF-IDF features.

Trains on 24 short texts across three clearly separable topics
(sports / cooking / technology), holds out the last two documents of
each class as a test set, and reports train/test accuracy plus a few
example predictions with probabilities.

Run from the repo root:  python logreg_text_classifier/demo.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logreg_text_classifier import LogisticRegressionTextClassifier

SPORTS = [
    "the striker scored a brilliant goal in the football match",
    "the goalkeeper saved the penalty and the crowd cheered loudly",
    "the football team won the championship after a thrilling match",
    "the coach praised the players after the victory in the stadium",
    "the striker headed the ball into the goal in the final minute",
    "the referee showed a red card during the tense football match",
    "the football fans celebrated the goal in the stadium",
    "the midfielder passed the ball and the striker scored the winning goal",
]

COOKING = [
    "bake the bread in the oven until the crust turns golden",
    "chop the garlic and onion then stir them into the tomato sauce",
    "the chef seasoned the chicken with garlic herbs and olive oil",
    "simmer the tomato sauce and stir in fresh basil leaves",
    "whisk the eggs and sugar then bake the cake in the oven",
    "the recipe calls for garlic onion tomato and fresh herbs",
    "roast the vegetables in the oven with olive oil and herbs",
    "taste the sauce and add garlic salt and a pinch of sugar",
]

TECHNOLOGY = [
    "the software engineer wrote python code for the database server",
    "the algorithm runs on the cloud server and processes the data",
    "the developer fixed the bug in the software and deployed the code",
    "the neural network model was trained on the computer cluster",
    "the database server crashed so the engineer restarted the software",
    "the programmer wrote an algorithm to sort the data efficiently",
    "the cloud platform hosts the software application and database",
    "machine learning code trains the neural network on the computer",
]

DATASET = [(doc, "sports") for doc in SPORTS] + \
          [(doc, "cooking") for doc in COOKING] + \
          [(doc, "technology") for doc in TECHNOLOGY]


def split_by_class(data, n_test_per_class=2):
    """Hold out the last ``n_test_per_class`` docs of each class."""
    train, test = [], []
    by_class = {}
    for doc, label in data:
        by_class.setdefault(label, []).append((doc, label))
    for label, items in by_class.items():
        train.extend(items[:-n_test_per_class])
        test.extend(items[-n_test_per_class:])
    return train, test


def main():
    train, test = split_by_class(DATASET)
    X_train = [doc for doc, _ in train]
    y_train = [label for _, label in train]
    X_test = [doc for doc, _ in test]
    y_test = [label for _, label in test]

    clf = LogisticRegressionTextClassifier(
        learning_rate=0.5, epochs=300, reg=0.01, random_state=0)
    clf.fit(X_train, y_train)

    vocab_size = len(clf.vectorizer_.vocabulary_)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print("Logistic-regression text classifier (TF-IDF + softmax)")
    print(f"Classes: {[str(c) for c in clf.classes_]}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train documents: {len(X_train)}  Test documents: {len(X_test)}")
    print(f"Train accuracy: {train_acc:.2f}")
    print(f"Test accuracy:  {test_acc:.2f}")
    print()
    print("Example predictions:")
    examples = [
        "the striker scored a goal in the final match",
        "bake the cake in the oven with sugar and eggs",
        "the python code runs the algorithm on the server",
        "the goalkeeper saved the penalty in the stadium",
        "chop the onion and stir the garlic into the sauce",
        "the neural network trains on the cloud computer",
    ]
    probs = clf.predict_proba(examples)
    preds = clf.predict(examples)
    for doc, pred, row in zip(examples, preds, probs):
        best = {str(label): round(float(p), 3)
                for label, p in zip(clf.classes_, row)}
        print(f"  {doc!r}\n    -> {pred}  {best}")


if __name__ == "__main__":
    main()
