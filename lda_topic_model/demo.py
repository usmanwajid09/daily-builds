"""Demo: LDA discovers sports vs cooking topics from a small corpus."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from lda_topic_model.model import LatentDirichletAllocation

SPORTS_DOCS = [
    "the striker scored a brilliant goal in the final match",
    "the team won the championship after a dramatic penalty shootout",
    "the coach praised the goalkeeper for saving the penalty",
    "the referee showed a red card during the tense match",
    "fans cheered in the stadium as the striker scored the winning goal",
    "the championship trophy went to the team with the best defence",
    "the midfielder passed the ball and the striker finished the goal",
    "the coach changed tactics at half time to win the match",
]

COOKING_DOCS = [
    "chop the garlic and onion then fry them in olive oil",
    "preheat the oven and bake the bread until golden brown",
    "add tomato sauce and simmer the pasta for ten minutes",
    "season the chicken with garlic herbs and lemon juice",
    "mix the flour sugar and butter then bake the cake in the oven",
    "stir the sauce slowly while the rice steams in the pot",
    "slice the vegetables and roast them in the hot oven",
    "taste the soup and add salt pepper and fresh herbs",
]

CORPUS = SPORTS_DOCS + COOKING_DOCS


def main():
    lda = LatentDirichletAllocation(
        n_topics=2, alpha=0.1, beta=0.1, n_iter=500, random_state=42
    ).fit(CORPUS)
    print(f"Corpus: {len(CORPUS)} documents, vocabulary: {len(lda.vocab_)} words")
    print(f"Topics: {lda.n_topics}, Gibbs sweeps: {lda.n_iter}")
    for topic in range(lda.n_topics):
        words = lda.top_words(topic, n=8)
        print(f"Topic {topic}: {', '.join(words)}")
    print()
    for label, idx in [("sports doc 0", 0), ("sports doc 4", 4),
                       ("cooking doc 0", 8), ("cooking doc 12", 12)]:
        dist = lda.doc_topic_dist(idx)
        print(f"{label:15s} -> topic 0: {dist[0]:.2f}, topic 1: {dist[1]:.2f}")
    print()
    new_docs = [
        "the striker scored the winning goal in the stadium",
        "bake the cake in the oven with butter and sugar",
    ]
    for doc, dist in zip(new_docs, lda.transform(new_docs)):
        print(f"'{doc}' -> topic 0: {dist[0]:.2f}, topic 1: {dist[1]:.2f}")
    print()
    print(f"All topic-word rows sum to 1: "
          f"{np.allclose(lda.topic_word_dist_.sum(axis=1), 1.0)}")
    print(f"All doc-topic rows sum to 1: "
          f"{np.allclose(lda.doc_topic_dist_.sum(axis=1), 1.0)}")


if __name__ == "__main__":
    main()
