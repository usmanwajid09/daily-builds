# Sentiment Analyzer

Lexicon-based sentiment analysis for English text, with negation and
intensifier handling. Pure Python + numpy — no NLTK or spaCy.

## How it works

- A weighted word lexicon (~90 words) gives each sentiment word a score
  in [-1, 1]: "love" is +1.0, "terrible" is -1.0.
- A preceding negation ("not", "never", "no") flips the polarity of the
  next sentiment word, so "not good" scores negative.
- Intensifiers ("very", "extremely", "really") multiply the next
  sentiment word's weight, so "very good" scores higher than "good".
- Each sentence gets a score (mean of its sentiment words, clipped to
  [-1, 1]); the document score is the sentiment-word-weighted mean of
  its sentences. Scores above +0.1 are `positive`, below -0.1 are
  `negative`, otherwise `neutral`.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python sentiment_analyzer/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_sentiment_analyzer.py -q
```

## Example output

```
Text: I love this phone. The camera is amazing and the battery lasts all day.
  Score: 1.000  Label: positive  (sentiment words: 2)
Text: I do not like this at all. It is not good and very disappointing.
  Score: -0.837  Label: negative  (sentiment words: 3)
```
