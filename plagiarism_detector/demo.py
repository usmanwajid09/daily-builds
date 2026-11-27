"""Demo: compare an original paragraph with a near-copy and an original."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plagiarism_detector.model import PlagiarismDetector

ORIGINAL = (
    "The quick brown fox jumps over the lazy dog near the river bank "
    "on a bright sunny morning in the month of June."
)
NEAR_COPY = (
    "The quick brown fox jumps over the lazy dog near the river bank "
    "on a bright sunny morning in the month of July."
)
UNRELATED = (
    "Quantum computers use qubits and superposition to perform certain "
    "calculations far faster than classical machines ever could."
)


def _show(title, a, b, detector):
    report = detector.compare(a, b)
    print(f"{title}:")
    print(f"  Jaccard: {report['jaccard']}, containment: {report['containment']}, "
          f"LCS ratio: {report['lcs_ratio']}")
    print(f"  Combined score: {report['combined_score']} — {report['verdict']}")
    print(f"  Shared shingles: {report['shared_shingles']}")
    for shingle in report["top_shingles"][:5]:
        print(f"    ... {shingle} ...")
    print()


def main():
    detector = PlagiarismDetector(n=3)
    _show("Original vs near-copy", ORIGINAL, NEAR_COPY, detector)
    _show("Original vs unrelated", ORIGINAL, UNRELATED, detector)
    _show("Original vs itself", ORIGINAL, ORIGINAL, detector)


if __name__ == "__main__":
    main()
