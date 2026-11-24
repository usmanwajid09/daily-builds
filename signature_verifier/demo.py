"""Demo: enroll a reference signature, then verify genuine vs forged samples.

Enrolls 3 genuine references for "usman", verifies 5 fresh genuine
samples and 5 forgeries, and prints each DTW distance, the decision,
and the overall accuracy.
"""

import os
import sys

# Allow running as a plain script (`python signature_verifier/demo.py`):
# make the repo root importable so `signature_verifier` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_verifier.verifier import (
    DEFAULT_THRESHOLD,
    SignatureVerifier,
    extract_features,
    make_signature,
)

PERSON = "usman"


def main():
    verifier = SignatureVerifier()
    references = [make_signature(PERSON, genuine=True, seed=100 + i) for i in range(3)]
    verifier.enroll(PERSON, references)

    ref_features = extract_features(references[0])
    print(f"enrolled {PERSON!r} with 3 references (threshold {DEFAULT_THRESHOLD})")
    print(
        "reference features: aspect_ratio="
        f"{ref_features['aspect_ratio']:.2f}, path_length="
        f"{ref_features['total_path_length']:.2f}, mean_curvature="
        f"{ref_features['mean_curvature']:.3f}, mean_speed="
        f"{ref_features['mean_speed']:.3f}"
    )
    print()

    genuine = [make_signature(PERSON, genuine=True, seed=200 + i) for i in range(5)]
    forgeries = [make_signature(PERSON, genuine=False, seed=300 + i) for i in range(5)]

    correct = 0
    total = 0
    print("genuine samples:")
    for i, sample in enumerate(genuine):
        result = verifier.verify(PERSON, sample)
        ok = result["decision"] == "genuine"
        correct += ok
        total += 1
        print(
            f"  genuine {i + 1}: distance={result['distance']:.4f} "
            f"score={result['score']:.2f} decision={result['decision']} "
            f"{'OK' if ok else 'WRONG'}"
        )
    print("forgery samples:")
    for i, sample in enumerate(forgeries):
        result = verifier.verify(PERSON, sample)
        ok = result["decision"] == "forgery"
        correct += ok
        total += 1
        print(
            f"  forgery {i + 1}: distance={result['distance']:.4f} "
            f"score={result['score']:.2f} decision={result['decision']} "
            f"{'OK' if ok else 'WRONG'}"
        )
    print()
    print(f"accuracy: {correct}/{total} = {100.0 * correct / total:.0f}%")


if __name__ == "__main__":
    main()
