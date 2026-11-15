# Naive Bayes Text Classifier

Multinomial Naive Bayes text classifier, from scratch. Documents are
tokenised into lowercase words, each class gets a prior and a
Laplace-smoothed word likelihood, and prediction uses log-probabilities
so long documents never underflow. Unknown words are simply ignored.

## Why

Naive Bayes is the classic baseline for text classification — spam
filters, sentiment analysis, topic routing — and it is small enough to
build by hand while still being genuinely useful on real text.

## Usage

```bash
pip install -r requirements.txt
python naive_bayes_text/demo.py
```

This trains on 10 toy sports/tech documents and classifies 4 held-out
ones.

Example output:

```
Vocabulary size: 51 words, classes: [sports, tech]
Training accuracy: 1.00
  'the goalkeeper saved the penalty' -> sports (true: sports, P=0.59)
  'the match was decided in extra time' -> sports (true: sports, P=0.87)
  'the new tablet has a bright display' -> tech (true: tech, P=0.81)
  'the update improves the wifi connection' -> tech (true: tech, P=0.59)
Test accuracy: 1.00
Unseen words handled: sports (no crash on unknown words)
```

## Run tests

```bash
python -m pytest tests/test_naive_bayes_text.py -q
```
