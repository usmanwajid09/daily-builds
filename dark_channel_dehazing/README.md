# dark_channel_dehazing

Single-image dehazing from scratch with numpy only, implementing the dark
channel prior of He, Sun & Tang (CVPR 2009):

1. **Dark channel** — per-pixel minimum over the RGB channels, then a
   minimum filter over a local patch. Haze-free images have a dark channel
   near zero almost everywhere; haze lifts it.
2. **Atmospheric light** — the brightest pixels of the hazy image among the
   top 0.1% of dark-channel values.
3. **Transmission map** — `t = 1 - omega * dark_channel(I / A)`, lightly
   smoothed with a box filter. `t = 1` means no haze.
4. **Recovery** — `J = (I - A) / max(t, t0) + A`, clipped to [0, 1].

## Run

```bash
python dark_channel_dehazing/demo.py
```

## Demo output

A synthetic 64×64 scene is hazed with transmission 0.55 and atmospheric
light (0.9, 0.9, 0.9), then dehazed. Expected: the estimated atmospheric
light is close to 0.9, mean transmission is near 0.55, and the recovered
image has a clearly lower MSE against the clean scene than the hazy image
does.

## Tests

```bash
python -m pytest dark_channel_dehazing/tests/test_dark_channel_dehazing.py -v
```
