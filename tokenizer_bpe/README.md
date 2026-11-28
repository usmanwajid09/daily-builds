# BPE Tokenizer

A byte-pair encoding (BPE) tokenizer built from scratch with only the
Python standard library. It learns merge rules from a corpus, then uses
them to split text into subword tokens — the same idea behind the
tokenizers used by modern language models.

## How BPE works

**Learning merges (training).** Every word in the corpus starts as its
individual characters plus an end-of-word marker (`</w>`), e.g.
`learning` → `l e a r n i n g </w>`. Then, repeatedly:

1. Count every adjacent symbol pair across all words, weighted by how
   often each word occurs.
2. Take the most frequent pair (ties broken by first-seen order, so
   training is deterministic) and merge it everywhere, e.g.
   `i` + `n` → `in`.
3. Record the merge rule and repeat until `num_merges` rules are learned
   or no pair occurs at least `min_frequency` times.

Frequent words collapse into few (or one) tokens; rare words stay close
to individual characters.

**Applying merges (encoding).** To encode new text, each word is again
split into characters + `</w>`, and the learned merges are applied
greedily in the order they were learned. Unseen words simply fall back
to characters — encoding never crashes on out-of-vocabulary input.

**Decoding.** Tokens are concatenated and each `</w>` marker is replaced
by a space, reconstructing the normalised (lowercased, alphanumeric)
text.

## Usage

```python
from tokenizer_bpe import BPETokenizer

tok = BPETokenizer(num_merges=50, min_frequency=2).fit([
    "the cat sat on the mat",
    "machine learning is fun",
    "learning machine learning",
])

tokens = tok.encode("the machine is learning")
print(tokens)                    # ['the</w>', 'machine</w>', 'is</w>', 'learning</w>']
print(tok.decode(tokens))        # 'the machine is learning'
print(tok.vocab_size)            # number of distinct symbols learned
print(tok.merges[:3])            # first learned merge rules
```

## Run the demo

```bash
python tokenizer_bpe/demo.py
```

## Example output

```
Merges learned: 40
Vocab size: 61
First 5 merges:
  'e' + '</w>' -> 'e</w>'
  'i' + 'n' -> 'in'
  't' + 'h' -> 'th'
  'th' + 'e</w>' -> 'the</w>'
  'e' + 'a' -> 'ea'

Sentence: 'the machine is learning'
Tokens (4): ['the</w>', 'machine</w>', 'is</w>', 'learning</w>']
Decoded: 'the machine is learning'
Round-trip OK: True

Word 'learning': 8 characters -> 1 tokens ['learning</w>']
```

## Tests

```bash
pytest tests/test_tokenizer_bpe.py -q
```
