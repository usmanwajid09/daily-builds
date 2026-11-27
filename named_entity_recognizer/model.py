"""Rule-based named entity recognizer.

Finds PERSON, ORG, LOCATION and DATE entities using capitalised-sequence
heuristics, small gazetteers (first names, org suffixes, countries,
cities) and date patterns (e.g. "12 March 2026", "2026-03-12",
"March 12, 2026"). Every entity carries its character offsets.
Pure Python + regex; no external NLP libraries.
"""

import re

PERSON = "PERSON"
ORG = "ORG"
LOCATION = "LOCATION"
DATE = "DATE"

FIRST_NAMES = {
    "ahmed", "ali", "usman", "fatima", "ayesha", "hassan", "hussain",
    "maryam", "bilal", "zainab", "omar", "sara", "john", "mary", "james",
    "robert", "michael", "william", "david", "richard", "charles",
    "joseph", "thomas", "daniel", "matthew", "emma", "olivia", "sophia",
    "ava", "mia", "isabella", "amelia", "harper", "evelyn", "elizabeth",
    "susan", "jessica", "sarah", "karen", "nancy", "lisa", "betty",
    "george", "edward", "ronald", "timothy", "jason", "jeffrey", "ryan",
    "jacob", "gary", "nicholas", "eric", "jonathan", "stephen", "larry",
    "justin", "scott", "brandon", "benjamin", "samuel", "gregory",
    "alexander", "patrick", "frank", "raymond", "jack", "dennis",
    "jerry", "tyler", "aaron", "jose", "nathan", "adam", "henry",
    "douglas", "zachary", "kyle", "noah", "ethan", "jeremy", "harold",
    "kevin", "brian", "george", "edward", "ronald", "anthony", "kevin",
    "asim", "hamza", "hassan", "iqra", "mehar", "sikandar", "faisal",
    "lehmia", "saif", "ullah", "nawaz", "sharif", "khan", "malik",
}

KNOWN_ORGS = {
    "google", "microsoft", "apple", "amazon", "facebook", "meta",
    "netflix", "tesla", "ibm", "intel", "samsung", "oracle", "adobe",
    "spotify", "twitter", "openai", "nasa", "spacex", "uber", "airbnb",
}

ORG_SUFFIXES = {
    "inc", "ltd", "llc", "corp", "corporation", "company", "co",
    "group", "labs", "technologies", "technology", "systems",
    "solutions", "university", "college", "institute", "school",
    "bank", "airlines", "studios", "media", "holdings", "partners",
    "industries", "enterprises", "foundation", "association",
}

COUNTRIES = {
    "pakistan", "india", "china", "japan", "germany", "france", "italy",
    "spain", "canada", "australia", "brazil", "russia", "turkey",
    "egypt", "nigeria", "kenya", "bangladesh", "nepal", "sri lanka",
    "thailand", "vietnam", "indonesia", "malaysia", "singapore",
    "korea", "south korea", "united states", "united kingdom",
    "united arab emirates", "saudi arabia", "netherlands", "belgium",
    "switzerland", "austria", "poland", "greece", "portugal", "ireland",
    "norway", "sweden", "denmark", "finland", "mexico", "argentina",
    "chile", "peru", "colombia", "iran", "iraq", "afghanistan",
}

CITIES = {
    "lahore", "karachi", "islamabad", "rawalpindi", "faisalabad",
    "multan", "peshawar", "quetta", "vehari", "burewala", "london",
    "paris", "berlin", "tokyo", "beijing", "delhi", "mumbai", "dubai",
    "new york", "los angeles", "chicago", "san francisco", "boston",
    "toronto", "sydney", "melbourne", "singapore", "kuala lumpur",
    "bangkok", "jakarta", "seoul", "shanghai", "hong kong", "cairo",
    "nairobi", "lagos", "madrid", "rome", "amsterdam", "vienna",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERNS = [
    # 12 March 2026 / 12th March 2026 / 12 Mar 26
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4}|\d{2})\b"),
    # March 12, 2026 / Mar 12 2026
    re.compile(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"),
    # 2026-03-12 / 2026/03/12
    re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"),
    # 12/03/2026 (day/month/year)
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
]

# Capitalised word sequences, e.g. "Usman Wajid", "FAST-NUCES" or
# "Microsoft Corporation". Each segment may be Titlecase or ALL-CAPS.
_CAP_SEQ_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:[-\s](?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
)

# Words that commonly start a sentence but are not entities.
_SENTENCE_STARTERS = {
    "the", "a", "an", "in", "on", "at", "for", "with", "from", "by",
    "and", "but", "or", "so", "if", "when", "while", "after", "before",
    "during", "this", "that", "these", "those", "it", "its", "he", "she",
    "we", "they", "you", "i", "his", "her", "their", "our", "my",
    "however", "therefore", "meanwhile", "finally", "first", "second",
    "today", "yesterday", "tomorrow", "now", "then", "here", "there",
}


def _find_dates(text):
    """Find DATE entities; returns list of entity dicts."""
    entities = []
    seen = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if (start, end) in seen:
                continue
            groups = match.groups()
            valid = True
            if pattern is _DATE_PATTERNS[0]:
                day, month_name = int(groups[0]), groups[1].lower()
                valid = month_name in MONTHS and 1 <= day <= 31
            elif pattern is _DATE_PATTERNS[1]:
                month_name, day = groups[0].lower(), int(groups[1])
                valid = month_name in MONTHS and 1 <= day <= 31
            elif pattern is _DATE_PATTERNS[2]:
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                valid = 1 <= month <= 12 and 1 <= day <= 31
            elif pattern is _DATE_PATTERNS[3]:
                day, month = int(groups[0]), int(groups[1])
                valid = 1 <= month <= 12 and 1 <= day <= 31
            if not valid:
                continue
            seen.add((start, end))
            entities.append({
                "text": match.group(0), "label": DATE,
                "start": start, "end": end,
            })
    return entities


def _classify_capitalised(text, start, end, date_spans):
    """Classify one capitalised sequence; returns a label or None."""
    if any(start >= ds and end <= de for ds, de in date_spans):
        return None
    phrase = text[start:end]
    words = re.split(r"[-\s]+", phrase)
    lowered = phrase.lower()
    last = words[-1].lower().rstrip(".")
    first = words[0].lower()
    if lowered in COUNTRIES or lowered in CITIES:
        return LOCATION
    if last in ORG_SUFFIXES:
        return ORG
    if first in FIRST_NAMES:
        return PERSON
    if re.sub(r"[-\s]", "", phrase).isupper():
        # All-caps acronym (e.g. FAST-NUCES, NASA): an organisation.
        return ORG
    if len(words) >= 2 and first not in _SENTENCE_STARTERS:
        # Two or more capitalised words, not a sentence starter:
        # most likely a person or organisation name.
        if any(w.lower() in ORG_SUFFIXES for w in words):
            return ORG
        return PERSON
    if len(words) == 1:
        if lowered in CITIES or lowered in COUNTRIES:
            return LOCATION
        if lowered in KNOWN_ORGS:
            return ORG
        if first in FIRST_NAMES:
            return PERSON
    return None


def extract_entities(text):
    """Extract named entities from ``text``.

    Returns a list of ``{"text", "label", "start", "end"}`` dicts,
    ordered by position in the text.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    entities = _find_dates(text)
    date_spans = [(e["start"], e["end"]) for e in entities]
    for match in _CAP_SEQ_RE.finditer(text):
        start, end = match.span()
        phrase = match.group(0)
        # Skip single capitalised words at the start of a sentence that
        # are ordinary sentence starters.
        if start == 0 or text[start - 2:start] in (". ", "! ", "? "):
            if len(phrase.split()) == 1 and phrase.lower() in _SENTENCE_STARTERS:
                continue
        label = _classify_capitalised(text, start, end, date_spans)
        if label is None:
            continue
        # Avoid overlapping a longer already-found entity.
        if any(start >= e["start"] and end <= e["end"] for e in entities):
            continue
        entities.append({
            "text": phrase, "label": label, "start": start, "end": end,
        })
    entities.sort(key=lambda e: (e["start"], e["end"]))
    return entities


def entities_by_label(text):
    """Group extracted entities by label: {label: [entity texts]}."""
    grouped = {PERSON: [], ORG: [], LOCATION: [], DATE: []}
    for entity in extract_entities(text):
        grouped[entity["label"]].append(entity["text"])
    return grouped


class NamedEntityRecognizer:
    """Rule-based NER for PERSON / ORG / LOCATION / DATE."""

    def extract(self, text):
        """List of entity dicts with text, label and character offsets."""
        return extract_entities(text)

    def by_label(self, text):
        """Entities grouped by label."""
        return entities_by_label(text)
