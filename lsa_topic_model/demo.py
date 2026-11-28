"""Demo: LSA topic modelling on 12 short documents in 3 themes.

Themes: pets, cooking, and space/astronomy (4 documents each).
Run from the repo root:  python lsa_topic_model/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lsa_topic_model import LSATopicModel

PETS = [
    "My pet cat and dog play together in the garden every morning",
    "The puppy barked at the kitten while the cat chased the dog around the house",
    "A vet treats sick pets, from a coughing puppy to a limping cat or dog",
    "Cats purr and dogs wag their tails when their owners pet them",
]

COOKING = [
    "The chef baked bread in the oven and served it hot with butter",
    "This recipe simmers garlic, tomato and onion in a pan for pasta sauce",
    "The chef chopped garlic and tomato for the sauce and seasoned the soup",
    "Bake the cake in the oven, stir the batter, and taste the sauce",
]

SPACE = [
    "The telescope showed a galaxy of stars beyond our planet",
    "An astronaut in orbit watched the moon and stars from the rocket",
    "The planet and its moon orbit a distant star in the galaxy",
    "A rocket carried the telescope into orbit to study stars and planets",
]

THEMES = ["pets", "cooking", "space"]


def main():
    documents = PETS + COOKING + SPACE
    labels = ["pets"] * 4 + ["cooking"] * 4 + ["space"] * 4

    model = LSATopicModel(n_topics=3).fit(documents)

    print("LSA topics (top 5 words per topic):")
    for i, topic in enumerate(model.get_topics(n_words=5)):
        words = ", ".join(f"{term} ({weight:+.3f})" for term, weight in topic)
        print(f"  Topic {i}: {words}")

    print("\nExplained variance ratio:", np.round(model.explained_variance_ratio_, 3))
    print(f"Total explained variance: {model.explained_variance_ratio_.sum():.3f}")

    print("\nDominant topic per document:")
    dominant = model.dominant_topics(documents)
    for doc, label, topic in zip(documents, labels, dominant):
        print(f"  [{label:7s}] topic {topic}: {doc[:60]}...")

    # Theme check: each theme's docs should mostly share one dominant topic.
    for theme in THEMES:
        topics = [int(dominant[i]) for i, lab in enumerate(labels) if lab == theme]
        common = max(set(topics), key=topics.count)
        print(
            f"Theme '{theme}': dominant topics {topics} "
            f"-> {topics.count(common)}/4 docs on topic {common}"
        )


if __name__ == "__main__":
    main()
