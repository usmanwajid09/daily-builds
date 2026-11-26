"""Demo: skip-gram word embeddings on a tiny animals/food corpus."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word2vec_skipgram.model import SkipGramWord2Vec

# Animals share the same sentence frames ("eats food", "sleeps at home",
# ...) so their vectors should end up close together; foods share a
# different set of frames ("is sweet", "grows on a tree", ...). A few
# sentences pair dog and cat directly, and puppy/kitten mirror dog/cat,
# so the puppy - dog + cat analogy should land on kitten.
CORPUS = []
for animal in ["dog", "cat", "bird", "fish"]:
    CORPUS += [
        f"the {animal} eats food",
        f"the {animal} drinks water",
        f"the {animal} sleeps at home",
        f"the {animal} runs in the garden",
        f"the {animal} is a pet",
        f"the {animal} is happy",
        f"the {animal} likes to play",
        f"the {animal} lives in the house",
    ]
CORPUS += [
    "dog and cat are pets",
    "dog chases cat",
    "cat chases dog",
    "dog and cat play together",
    "dog and cat sleep together",
    "dog and cat eat together",
    "puppy is a young dog",
    "kitten is a young cat",
    "puppy plays with dog",
    "kitten plays with cat",
    "puppy eats food",
    "kitten eats food",
    "puppy sleeps at home",
    "kitten sleeps at home",
    "puppy is happy",
    "kitten is happy",
]
for food in ["apple", "banana"]:
    CORPUS += [
        f"I eat {food}",
        f"the {food} is sweet and tasty",
        f"the {food} grows on a tree",
        f"eat {food} for breakfast",
    ]
CORPUS += [
    "apple and banana are fruit",
    "apple and banana are sweet",
    "bread is fresh and tasty",
    "cheese is tasty and fresh",
    "milk is fresh and healthy",
    "eat bread for breakfast",
    "drink milk for breakfast",
    "bread and cheese are food",
]


def main():
    # window=1: the sentences are short (3-4 content words), so the
    # immediate neighbours already carry the meaning; a wider window
    # mostly adds noise on a corpus this small.
    model = SkipGramWord2Vec(
        embedding_dim=16,
        window=1,
        negative_samples=5,
        learning_rate=0.05,
        epochs=200,
        seed=42,
    ).fit(CORPUS)
    print(f"Sentences: {len(CORPUS)}, vocabulary: {len(model.vocab_)} words")
    print(f"Training pairs: {model.n_pairs_}, epochs: {model.epochs}")
    print(
        f"Loss: {model.loss_history_[0]:.3f} (first epoch) -> "
        f"{model.loss_history_[-1]:.3f} (last epoch)"
    )
    for word in ["dog", "cat", "apple", "puppy"]:
        neighbours = ", ".join(
            f"{w} ({s:.2f})" for w, s in model.most_similar(word, topn=5)
        )
        print(f"Nearest to '{word}': {neighbours}")
    print(
        f"similarity(dog, cat) = {model.similarity('dog', 'cat'):.3f}  vs  "
        f"similarity(dog, apple) = {model.similarity('dog', 'apple'):.3f}"
    )
    print(
        f"similarity(apple, banana) = {model.similarity('apple', 'banana'):.3f}"
    )
    analogy = model.analogy("puppy", "dog", "cat", topn=3)
    print(
        "Analogy: puppy - dog + cat = "
        + ", ".join(f"{w} ({s:.2f})" for w, s in analogy)
    )


if __name__ == "__main__":
    main()
