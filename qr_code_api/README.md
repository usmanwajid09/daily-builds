# Barcode-Code API

A tiny Flask API that encodes text into a deterministic 2D grid of
0/1 bits — like a barcode you can print as text — and decodes such a
grid back to the original text. It uses no imaging library: the grid
is plain JSON plus a `#`/space ASCII picture.

## Why

Real QR codes need an imaging library, but the core idea — text in,
a grid of bits out, the same text back — is easy to show with plain
Python. This app demonstrates a reversible encoding, strict input
validation, and a round-trip property tests can check.

## How the encoding works

The UTF-8 bytes of the text are written most-significant bit first,
row by row, into a grid 16 bits wide, zero-padded at the end. Decoding
reads the bits back into bytes, drops trailing zero padding, and
decodes UTF-8. The same text always produces the same grid.

## Usage

```bash
pip install -r requirements.txt
python -m qr_code_api.demo
```

The demo encodes `"Hi!"`, prints its ASCII picture, and decodes it
back.

To run the app for real:

```bash
flask --app "qr_code_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/encode` | `{"text": str}` (max 256 chars) -> `{"text", "rows", "cols", "grid", "ascii"}` |
| POST | `/decode` | `{"grid": [[0\|1, ...], ...]}` -> `{"text": str}` |

Validation errors return `400` with `{"error": "..."}`; unknown routes
return `404` with `{"error": "..."}`. A grid with ragged rows, non-0/1
cells, or bytes that are not valid UTF-8 returns `400`.

## Run tests

```bash
python -m pytest tests/test_qr_code_api.py -v
```
