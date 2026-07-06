"""Interactive/CLI demo: classify a piece of text as human- or ai-written,
and show which stylometric features drove the prediction.

Usage:
    python -m ai_slop_detector.demo "Some text to classify."
    python -m ai_slop_detector.demo --file notes.txt
    python -m ai_slop_detector.demo               # interactive REPL, Ctrl-D/Ctrl-C to quit

Loads trained_model.json (produced by `python -m ai_slop_detector.train`);
if it's missing, trains it first with default settings so the demo works
out of the box.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ai_slop_detector.features import FEATURE_NAMES, extract_features

MODEL_PATH = Path(__file__).parent / "trained_model.json"
LABEL_NAMES = {0: "human", 1: "ai"}


class LoadedModel:
    def __init__(self, payload: dict):
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                "trained_model.json's feature_names no longer match "
                "features.FEATURE_NAMES -- retrain with `python -m "
                "ai_slop_detector.train` before running the demo."
            )
        self.weights = np.array(payload["weights"])
        self.bias = payload["bias"]
        self.mean = np.array(payload["scaler_mean"])
        self.std = np.array(payload["scaler_std"])

    def predict(self, text: str) -> dict:
        fv = extract_features(text)
        raw = fv.to_array()
        scaled = (raw - self.mean) / self.std
        contributions = scaled * self.weights
        z = float(contributions.sum() + self.bias)
        p_ai = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        label = "ai" if p_ai >= 0.5 else "human"

        ranked = sorted(
            zip(FEATURE_NAMES, raw, contributions),
            key=lambda t: -abs(t[2]),
        )
        return {
            "label": label,
            "p_ai": p_ai,
            "contributions": ranked,
        }


def load_model(path: Path = MODEL_PATH) -> LoadedModel:
    if not path.exists():
        print(f"No trained model found at {path} -- training one now with default settings...")
        subprocess.run([sys.executable, "-m", "ai_slop_detector.train"], check=True)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return LoadedModel(payload)


def format_prediction(text: str, result: dict, top_n: int = 5) -> str:
    lines = [
        f'Text: "{text}"' if len(text) <= 80 else f'Text: "{text[:77]}..."',
        f"Prediction: {result['label']}  (p_ai={result['p_ai']:.3f})",
        f"Top {top_n} contributing features (positive pushes toward 'ai', negative toward 'human'):",
    ]
    for name, raw_value, contribution in result["contributions"][:top_n]:
        lines.append(f"  {name:<38s} value={raw_value:8.3f}  contribution={contribution:+.3f}")
    return "\n".join(lines)


def classify_one(model: LoadedModel, text: str) -> None:
    text = text.strip()
    if not text:
        print("(empty input, skipping)")
        return
    try:
        result = model.predict(text)
    except ValueError as e:
        print(f"Could not classify: {e}")
        return
    print(format_prediction(text, result))
    print()


def run_repl(model: LoadedModel) -> None:
    print("ai-slop-detector interactive demo. Enter text and press Enter (Ctrl-D to quit).\n")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        classify_one(model, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="Text to classify (skip for interactive mode)")
    parser.add_argument("--file", type=Path, help="Classify each non-empty line of a file")
    args = parser.parse_args()

    model = load_model()

    if args.file:
        lines = [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines()]
        for line in lines:
            if line:
                classify_one(model, line)
    elif args.text:
        classify_one(model, args.text)
    else:
        run_repl(model)


if __name__ == "__main__":
    main()
