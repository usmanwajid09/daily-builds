# Language Detector

Character-trigram language identifier, from scratch. Each language
(English, French, Spanish, German) gets a profile of its 300 most
frequent character trigrams, built from small sample texts embedded
in the module. A new text is classified by the Cavnar & Trenchard
"out-of-place" distance: trigrams are ranked by frequency, and the
distance to a language is the sum of the absolute rank differences
(a trigram missing from a language profile costs the maximum rank).
The language with the smallest distance wins.

Accented characters are kept, so é/è (French), ñ/á (Spanish) and
ä/ö/ü/ß (German) act as strong signals. Text is lowercased and
padded with a space on each side so word-boundary trigrams such as
`" th"` are captured too.

## Behaviour

- `classify(text)` returns a language code: `"en"`, `"fr"`, `"es"`,
  `"de"`, or `"unknown"`.
- Empty or whitespace-only input raises `ValueError`; non-string
  input raises `TypeError`.
- Text with fewer than 10 alphabetic characters returns `"unknown"`
  — a couple of words carry too little trigram evidence to trust a
  guess (e.g. `classify("hello")` → `"unknown"`).
- `distances(text)` returns the out-of-place distance to each
  language (lower is closer); `scores(text)` returns per-language
  similarity scores that sum to 1 (higher is more likely).
- `LanguageDetector.fit({"en": [...], "fr": [...]})` rebuilds the
  profiles from your own training texts.

## Usage

```python
from language_detector.model import LanguageDetector

detector = LanguageDetector()
detector.classify("The sun rises early in the summer and the birds sing.")
# 'en'
detector.classify("Le soleil se lève tôt en été et les oiseaux chantent.")
# 'fr'
detector.distances("El sol sale temprano en verano y los pájaros cantan.")
# {'de': 12415, 'en': 11091, 'es': 7880, 'fr': 11108}
detector.scores("Die Sonne geht im Sommer früh auf und die Vögel singen.")
# {'de': 0.92, 'en': 0.03, 'es': 0.01, 'fr': 0.03}
detector.classify("hi")   # 'unknown'
detector.classify("")     # raises ValueError
```

## Demo

```bash
python language_detector/demo.py
```

Classifies one held-out sentence per language (none of them appear
in the training texts):

```
Languages: en (English), fr (French), es (Spanish), de (German)
Profile size: top 300 trigrams per language

[English] The sun rises early in the summer and the birds sing in the trees behind our house every morning.
  -> en (English) OK; closest: en=13035, de=17848, fr=18250, es=20462
     scores: en=0.99, de=0.01, fr=0.01, es=0.00

[French] Le soleil se lève tôt en été et les oiseaux chantent dans les arbres derrière notre maison chaque matin.
  -> fr (French) OK; closest: fr=15332, es=22085, de=22651, en=23128
     scores: fr=1.00, es=0.00, de=0.00, en=0.00

[Spanish] El sol sale temprano en verano y los pájaros cantan en los árboles detrás de nuestra casa cada mañana.
  -> es (Spanish) OK; closest: es=12991, fr=20002, en=22020, de=22551
     scores: es=1.00, fr=0.00, en=0.00, de=0.00

[German] Die Sonne geht im Sommer früh auf und die Vögel singen jeden Morgen in den Bäumen hinter unserem Haus.
  -> de (German) OK; closest: de=15443, fr=21542, en=22755, es=23594
     scores: de=1.00, fr=0.00, en=0.00, es=0.00

Demo accuracy: 4/4
Short text: 'hello' -> unknown (too little evidence)
Empty text raises ValueError: text must not be empty or whitespace-only
```

## Run tests

```bash
python -m pytest tests/test_language_detector.py -q
```

31 tests covering the four demo classifications, all training
sentences, validation errors (empty / non-string / bad `top_n` /
bad training data), short and digits-only text, and the
`distances()` / `scores()` behaviour.
