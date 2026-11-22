# cnn_from_scratch

A tiny convolutional neural network built from scratch with numpy only —
no torch, no TensorFlow. It implements the classic pieces by hand:

- `Conv2D` — single-channel 2-D convolution (stride 1, no padding), with
  forward and backward passes and SGD updates for filters and biases.
- `ReLU` — element-wise activation with a masked backward pass.
- `MaxPool2D` — 2×2 max-pooling (stride 2) that remembers argmax positions
  so gradients route back to the winning pixel.
- `Dense` — fully connected layer.
- `TinyCNN` — `Conv -> ReLU -> MaxPool -> Dense -> softmax`, trained with
  SGD on softmax cross-entropy loss.

A synthetic data generator (`make_shape_dataset`) draws noisy circles
(label 0) and squares (label 1) on 12×12 images so the network has a real
task to learn without any external dataset.

## Run

```bash
python cnn_from_scratch/demo.py
```

## Demo output

Trains for 30 epochs on 80 synthetic images and evaluates on 20 held-out
images. Expected output: loss falls from ~0.7 to well under 0.2 and test
accuracy is above 0.9.

## Tests

```bash
python -m pytest cnn_from_scratch/tests/test_cnn_from_scratch.py -v
```
