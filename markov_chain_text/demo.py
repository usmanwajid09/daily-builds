"""Generate text with an order-2 Markov chain.

Trains on a short paragraph about machine learning, then generates a
30-word passage from the seed words "machine learning" with a fixed seed,
so the output is the same on every run.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markov_chain_text import MarkovChainText

CORPUS = (
    "machine learning models learn patterns from data "
    "machine learning engineers build models that predict outcomes "
    "data scientists collect data and train models on computers "
    "models improve when they see more data and better features "
    "machine learning is useful for prediction and classification"
)
SEED_WORDS = "machine learning"
LENGTH = 30
SEED = 42


def main():
    chain = MarkovChainText(order=2, random_state=SEED)
    chain.fit(CORPUS)
    text = chain.generate(length=LENGTH, seed_words=SEED_WORDS)
    print(f"Corpus: {len(CORPUS.split())} words, {chain.n_states} learned states")
    print(f'Seed words: "{SEED_WORDS}", length {LENGTH} words, seed {SEED}\n')
    print(f'Generated: "{text}"')
    print(f"\nWord count: {len(text.split())} (expected {LENGTH})")


if __name__ == "__main__":
    main()
