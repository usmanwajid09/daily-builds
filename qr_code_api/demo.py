"""Demo: encode text to a bit grid, print it, and decode it back."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_code_api.app import create_app


def main():
    client = create_app().test_client()

    encoded = client.post("/encode", json={"text": "Hi!"}).get_json()
    print(f"Encoded 'Hi!' -> {encoded['rows']}x{encoded['cols']} grid")
    print(encoded["ascii"])

    decoded = client.post("/decode", json={"grid": encoded["grid"]}).get_json()
    print(f"Decoded back: {decoded['text']!r}")
    assert decoded["text"] == "Hi!", "round-trip failed"

    bad = client.post("/encode", json={"text": ""})
    print(f"Empty text rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
