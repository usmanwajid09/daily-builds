# Template-Matching OCR (5x7 bitmap font)

Recognise printed digits (0-9) and uppercase letters (A-Z) with
template matching and no external fonts or image files. A built-in
5x7 bitmap font defines one template per character; text is rendered
into a binary image, split into fixed-pitch character cells (glyph
width 5, spacing 1, a blank cell meaning a word space), each cell is
resampled to 5x7 and matched against the templates by Hamming
distance, and the closest template wins.

## Usage

```python
from ocr_template import TemplateOCR

ocr = TemplateOCR()
image = ocr.render("HELLO 123", scale=2)
print(ocr.recognize(image))  # 'HELLO 123'
print(ocr.recognize_with_scores(image)[1])  # [(char, hamming dist), ...]
```

## Run demo / tests

```bash
python ocr_template/demo.py
python -m pytest tests/test_ocr_template.py -v
```

The demo renders "HELLO 123", recognises it back, and repeats the
test after flipping 12 pixels to show the matcher tolerates light
noise.
