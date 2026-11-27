# Named Entity Recognizer

Rule-based named entity recognition for PERSON, ORG, LOCATION and
DATE, with character offsets. Pure Python + regex — no NLTK or spaCy.

## How it works

- **DATE** is matched first, with patterns for "12 March 2026",
  "March 12, 2026", "2026-03-12" and "12/03/2026" (with month/day
  validity checks).
- **LOCATION** comes from gazetteers of countries and cities.
- **ORG** is detected by organisation suffixes ("Corporation",
  "University", "Labs", "Inc", "Bank", ...).
- **PERSON** is detected from a first-name gazetteer, or from a
  capitalised multi-word sequence that is not a sentence starter.
- Single capitalised words at the start of a sentence are ignored
  unless they are in a gazetteer.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python named_entity_recognizer/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_named_entity_recognizer.py -q
```

## Example output

```
Entities found:
  [  PERSON] 'Usman Wajid' (chars 0-11)
  [     ORG] 'FAST-NUCES' (chars 26-36)
  [LOCATION] 'Lahore' (chars 40-46)
  [     ORG] 'Google' (chars 73-79)
  [    DATE] '12 March 2026' (chars 83-96)
```
