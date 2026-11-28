"""Demo: learn BPE merges from a small corpus, then encode and decode."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer_bpe import BPETokenizer

# Repeated words (the, cat, learning, machine, ...) give the pair counts
# something real to merge; rare words stay mostly as characters.
CORPUS = [
    "the cat sat on the mat",
    "the cat likes the dog",
    "the dog chased the cat",
    "machine learning is fun",
    "learning machine learning",
    "the machine is learning",
    "deep learning learns from data",
    "the data teaches the machine",
    "learning from data is learning",
    "the cat and the dog play",
    "machine learning models learn",
    "the model learns the data",
    "cats and dogs are animals",
    "the learning machine never stops",
    "data science is machine learning",
]


def main():
    tokenizer = BPETokenizer(num_merges=50, min_frequency=2).fit(CORPUS)
    print(f"Merges learned: {len(tokenizer.merges)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("First 5 merges:")
    for left, right in tokenizer.merges[:5]:
        print(f"  {left!r} + {right!r} -> {left + right!r}")

    sentence = "the machine is learning"
    tokens = tokenizer.encode(sentence)
    print(f"\nSentence: {sentence!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens)!r}")
    print(f"Round-trip OK: {tokenizer.decode(tokens) == sentence}")

    word = "learning"
    word_tokens = tokenizer.encode(word)
    print(f"\nWord {word!r}: {len(word)} characters -> "
          f"{len(word_tokens)} tokens {word_tokens}")


if __name__ == "__main__":
    main()
