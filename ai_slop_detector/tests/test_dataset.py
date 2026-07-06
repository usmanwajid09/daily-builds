import csv
from pathlib import Path

from ai_slop_detector.data.build_dataset import build_samples, TOPICS

DATA_PATH = Path(__file__).parent.parent / "data" / "samples.csv"


def test_committed_csv_exists_and_is_well_formed():
    assert DATA_PATH.exists(), "samples.csv must be generated and committed"
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for row in rows:
        assert row["label"] in {"human", "ai"}
        assert row["text"].strip() != ""


def test_dataset_is_balanced():
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_human = sum(1 for r in rows if r["label"] == "human")
    n_ai = sum(1 for r in rows if r["label"] == "ai")
    assert n_human == n_ai, f"expected balanced classes, got human={n_human} ai={n_ai}"


def test_no_duplicate_texts_across_dataset():
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    assert len(texts) == len(set(texts)), "dataset contains duplicate text rows"


def test_build_samples_is_deterministic_given_seed():
    a = build_samples(seed=123)
    b = build_samples(seed=123)
    assert a == b


def test_build_samples_covers_every_topic_for_both_labels():
    samples = build_samples()
    for topic in TOPICS:
        human_hits = [s for s, label in samples if label == "human" and topic in s]
        ai_hits = [s for s, label in samples if label == "ai" and topic in s]
        assert human_hits, f"no human sample mentions topic: {topic}"
        assert ai_hits, f"no ai sample mentions topic: {topic}"
