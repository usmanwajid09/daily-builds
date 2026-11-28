# stemmer_porter

A Porter stemmer (Porter, 1980) implemented from scratch in pure Python
(standard library only). It reduces English words to their stems, so
word variants such as `connect`, `connected`, `connecting` and
`connection` all collapse to `connect` -- the classic preprocessing step
for search and information retrieval.

## How the Porter algorithm works

A *consonant* is any letter other than A, E, I, O, U -- except Y, which
counts as a consonant at the start of a word or after a vowel, and as a
vowel after a consonant. Writing a word as `[C](VC)^m[V]` (optional
consonants, then `m` vowel-consonant sequences, then optional vowels),
`m` is the word's *measure*. Each rule replaces a suffix when the stem
before it satisfies a condition on `m` (usually `m > 0` or `m > 1`).
The algorithm runs five steps, applying at most one rule per step (the
longest matching suffix wins):

1. **Plurals and past participles** -- 1a: `sses -> ss`, `ies -> i`,
   `ss -> ss`, `s -> ''`; 1b: `(m>0) eed -> ee`, `(*v*) ed/ing` removed,
   then `-at/-bl/-iz` get their E back, double consonants shorten
   (except L, S, Z), and `(m=1 and *o)` adds E; 1c: `(*v*) y -> i`.
2. **Derivational suffixes** (`m > 0`): `ational -> ate`,
   `tional -> tion`, `enci -> ence`, `anci -> ance`, `izer -> ize`,
   `ization -> ize`, `ation -> ate`, `ator -> ate`, `alism -> al`,
   `iveness -> ive`, `fulness -> ful`, `ousness -> ous`,
   `aliti -> al`, `iviti -> ive`, `biliti -> ble`, and more.
3. **More derivational suffixes** (`m > 0`): `icate -> ic`,
   `ative -> ''`, `alize -> al`, `iciti -> ic`, `ical -> ic`,
   `ful -> ''`, `ness -> ''`.
4. **Remaining derivational suffixes** (`m > 1`): `al`, `ance`, `ence`,
   `er`, `ic`, `able`, `ible`, `ant`, `ement`, `ment`, `ent`, `ou`,
   `ism`, `ate`, `iti`, `ous`, `ive`, `ize` removed, and `ion` removed
   when the stem ends in S or T.
5. **Tidying up** -- 5a: a final E is dropped when `m > 1`, or when
   `m == 1` and the stem does not end CVC; 5b: a final double L is
   shortened to one L when `m > 1`.

Validated against the standard Porter test vocabulary (Porter 1980;
tartarus.org `voc.txt` / `output.txt`). One note: the widely copied
example list shows `valenci -> valence`, but that is the step-2
intermediate form -- step 5a then drops the final E, and the official
`output.txt` confirms it (`valence` stems to `valenc`). This
implementation produces the correct `valenc`.

## Usage

```python
from stemmer_porter import stem, stem_text, PorterStemmer

stem("caresses")        # 'caress'
stem("relational")      # 'relat'
stem("vietnamization")  # 'vietnam'

stem_text("connecting the connections")
# 'connect the connect'

stemmer = PorterStemmer()
stemmer.stem("happiness")  # 'happi'
```

`stem` lowercases its input, strips surrounding whitespace, raises
`TypeError` for non-strings and `ValueError` for empty input. Words of
length 1 or 2 pass through unchanged, as in Porter's reference
implementation. `stem_text` stems every whitespace-separated word in
order and joins the stems with single spaces.

## Example output

Running `python stemmer_porter/demo.py` from the repo root:

```
Porter stemmer demo
===================

Word -> stem:
  caresses         -> caress
  ponies           -> poni
  agreed           -> agre
  motoring         -> motor
  hopping          -> hop
  relational       -> relat
  conditional      -> condit
  vietnamization   -> vietnam
  hopefulness      -> hope
  communism        -> commun
  bowdlerize       -> bowdler
  happiness        -> happi

Sentence: 'Connecting the connections and the connected systems relationally'
Stemmed:  'connect the connect and the connect system relation'
Distinct raw words: 7 -> distinct stems: 5
```

## Tests

`pytest tests/test_stemmer_porter.py -q` -- 81 tests, covering the full
standard Porter test vocabulary (parametrized), each step separately,
idempotence on already-stemmed words, word-family collapse,
input validation, and `stem_text` behaviour.
