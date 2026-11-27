# Plagiarism Detector

Text-overlap detection between two documents using word n-gram
shingles. Pure Python + numpy.

## How it works

- Each text is turned into a set of word n-gram shingles (default
  trigrams, e.g. "the quick brown").
- **Jaccard similarity** = shared shingles / union of shingles.
- **Containment** = shared shingles / smaller shingle set — catches
  the case where a short text is copied into a long one.
- **LCS ratio** = longest-common-subsequence length (over words) /
  length of the shorter text — catches reordered or lightly edited
  copies.
- The **combined score** is the mean of the three; the verdict is
  "high overlap — likely copied" (≥ 0.5), "moderate overlap —
  possible paraphrase" (≥ 0.2), "low overlap" (≥ 0.05) or "no
  meaningful overlap". The report lists the shared shingles so the
  match can be inspected by eye.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python plagiarism_detector/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_plagiarism_detector.py -q
```

## Example output

```
Original vs near-copy:
  Jaccard: 0.9091, containment: 0.9524, LCS ratio: 0.9565
  Combined score: 0.9393 — high overlap — likely copied
  Shared shingles: 20
```
