# Skip-Gram Word2Vec (from scratch)

Skip-gram word embeddings (word2vec) built from scratch with numpy only.
Each word gets a dense vector, trained to predict the words around it:
for every centre word, the model is pushed towards its real context
words and away from a few randomly drawn "negative" words (negative
sampling), updated with plain SGD. Words that appear in similar
contexts end up with similar vectors, so cosine similarity between
vectors measures how alike two words are. The demo also shows a word
analogy: `puppy - dog + cat` should land on `kitten`.

Tokenisation lowercases, strips punctuation, applies a tiny stemmer
("eats" -> "eat", so one word's contexts are not split across verb
forms), and drops stopwords ("the", "is", ...) before training, since
they sit next to almost every word and carry no distinguishing meaning.

## Usage

```python
from word2vec_skipgram.model import SkipGramWord2Vec

model = SkipGramWord2Vec(
    embedding_dim=16, window=1, negative_samples=5,
    learning_rate=0.05, epochs=200, seed=42,
).fit([
    "the dog eats food",
    "the cat eats food",
    "the apple is sweet",
    "the banana is sweet",
])

model.most_similar("dog")        # [('cat', 0.94), ...]
model.similarity("dog", "cat")   # 0.941
model.analogy("puppy", "dog", "cat")  # [('kitten', 0.95), ...]
model.get_vector("dog")          # numpy array of shape (16,)
```

Run the demo:

```bash
python word2vec_skipgram/demo.py
```

It trains on a 64-sentence toy corpus about animals and food (animals
share one set of sentence frames, foods another) in about 3 seconds.

Example output:

```
Sentences: 64, vocabulary: 36 words
Training pairs: 238, epochs: 200
Loss: 0.693 (first epoch) -> 0.253 (last epoch)
Nearest to 'dog': cat (0.94), chase (0.80), bird (0.60), fish (0.57), puppy (0.57)
Nearest to 'cat': dog (0.94), chase (0.75), bird (0.72), fish (0.69), water (0.65)
Nearest to 'apple': banana (0.91), fruit (0.83), tree (0.76), bread (0.73), food (0.66)
Nearest to 'puppy': kitten (1.00), together (0.83), home (0.63), cat (0.63), food (0.58)
similarity(dog, cat) = 0.941  vs  similarity(dog, apple) = 0.135
similarity(apple, banana) = 0.914
Analogy: puppy - dog + cat = kitten (0.95), together (0.88), home (0.66)
```

Animals cluster with animals and foods with foods: "dog" is closest to
"cat", "apple" is closest to "banana", and the puppy/dog/cat analogy
returns "kitten". (Exact decimals may vary slightly with the numpy
version; the neighbour order is stable for seed 42.)

## Run tests

```bash
python -m pytest tests/test_word2vec_skipgram.py -q
```
