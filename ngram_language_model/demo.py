"""Demo: train a bigram language model on short cat/dog sentences,
compare perplexity of in-domain vs out-of-domain text, and generate
a few sentences greedily from a seed.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngram_language_model import NGramLanguageModel

CORPUS = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "the cat chases the dog",
    "the dog chases the cat home",
    "my cat likes milk and fish",
    "my dog likes bones and walks",
    "the cat sleeps all day",
    "the dog barks at the cat outside",
    "a happy dog wags its tail",
    "a sleepy cat naps in the sun",
]


def main():
    model = NGramLanguageModel(order=3, k=0.1)
    model.fit(CORPUS)

    print("Vocabulary size: %d" % model.vocab_size)
    print("Total tokens: %d" % model.total_tokens)

    in_domain = "the dog chases the cat"
    out_domain = "quantum banana spaceship zebra"
    print("Perplexity (in-domain '%s'): %.2f" % (in_domain, model.perplexity([in_domain])))
    print("Perplexity (out-of-domain '%s'): %.2f" % (out_domain, model.perplexity([out_domain])))

    for seed in ("my cat", "a happy", "the dog"):
        print("Generated from '%s': %s" % (seed, model.generate(seed, max_len=10)))


if __name__ == "__main__":
    main()
