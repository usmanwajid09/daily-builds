"""Demo: extract named entities from a short news-style paragraph."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from named_entity_recognizer.model import NamedEntityRecognizer, entities_by_label

TEXT = (
    "Usman Wajid, a student at FAST-NUCES in Lahore, presented his "
    "project to Google on 12 March 2026. "
    "The team from NovaStack travelled to Karachi, Pakistan, to meet "
    "Ahmed Khan from Microsoft Corporation on 2026-04-01."
)


def main():
    recognizer = NamedEntityRecognizer()
    print("Entities found:")
    for entity in recognizer.extract(TEXT):
        print(f"  [{entity['label']:>8}] {entity['text']!r} "
              f"(chars {entity['start']}-{entity['end']})")
    print()
    print("Grouped by label:")
    for label, items in entities_by_label(TEXT).items():
        print(f"  {label}: {items}")


if __name__ == "__main__":
    main()
