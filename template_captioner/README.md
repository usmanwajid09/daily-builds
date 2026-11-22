# template_captioner

Template-based image captioning from scratch with numpy only — no learned
model, no vision-language network. The captioner measures exact scene
attributes and composes a sentence from templates:

- **Dominant colours** — each pixel is matched to the nearest named colour
  in a small palette; colours covering ≥5% of the image are reported.
- **Shape count** — connected foreground regions (pixels differing from the
  median background) are counted, largest first.
- **Brightness / contrast / saturation** — mean and standard deviation of
  luminance, plus mean channel spread.
- **Layout** — the quadrant (upper/lower left/right, or centre) holding the
  largest shape's centroid.

`analyze_image()` returns all attributes as a dict; `caption()` turns them
into a sentence like *"A dark image with 1 shape on a dark background.
Dominant colours: red, black. The largest shape is in the centre of the
image."*

## Run

```bash
python template_captioner/demo.py
```

## Demo output

Captions three synthetic scenes: a red circle on black, a blue square +
yellow circle on white, and a blank black image (captioned as plain, with
no shapes).

## Tests

```bash
python -m pytest template_captioner/tests/test_template_captioner.py -v
```
