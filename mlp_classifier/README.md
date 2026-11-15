# MLP classifier, from scratch

A small multi-layer perceptron classifier — no frameworks, just numpy.
It is a feed-forward network with one hidden layer: an affine
transform, ReLU activation, a second affine transform, and a softmax
output trained with cross-entropy loss by mini-batch stochastic
gradient descent. Weights use He initialisation (scaled for ReLU),
inputs are standardised before training, and the trained mean/std are
stored so predictions use the same scaling. Training shuffles the data
each epoch with a seeded RNG, so results are reproducible for a given
`random_state`. The network is deliberately small — it exists to show
the full forward/backward pass in readable numpy, not to compete with
a framework — but it genuinely learns non-linear boundaries such as
two moons.

## Why

Every deep-learning framework hides this exact computation behind an
autograd engine. Writing the forward and backward pass by hand, in
about 40 lines, is the fastest way to actually understand what a
neural network is doing when it "learns".

## Usage

```bash
pip install -r requirements.txt
python mlp_classifier/demo.py
```

This generates two interleaving moons (a non-linear boundary), trains
a 16-unit MLP for 300 epochs on 80% of the data, and prints test
accuracy and final training loss on the remaining 20%.

Example result:

```
Test accuracy: 0.975
Final training loss: 0.0891
```

## Run tests

```bash
python -m pytest tests/test_mlp_classifier.py -v
```
