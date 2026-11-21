# Gradient Descent Variants, from scratch

SGD, SGD with momentum, RMSprop and Adam — implemented with numpy only,
sharing one interface (`step(params, grads)` returns new parameters and
never mutates its inputs) — compared on an ill-conditioned quadratic and
on binary logistic regression.

## What it does

- **`SGD(lr)`** — plain gradient descent: `x -= lr * grad`.
- **`Momentum(lr, momentum)`** — velocity `v = momentum * v + grad`,
  then `x -= lr * v`. Sustained gradients accelerate; oscillations cancel.
- **`RMSprop(lr, decay, eps)`** — per-coordinate adaptive rates from a
  decaying average of squared gradients. Steep coordinates get smaller
  steps, flat ones bigger steps.
- **`Adam(lr, beta1, beta2, eps)`** — bias-corrected first and second
  moment estimates; momentum plus RMSprop-style scaling. The first step
  is roughly `lr * sign(grad)`.
- **`minimize(objective, x0, optimizer, n_steps)`** — runs any optimizer
  on any `f(x) -> (loss, grad)` objective; returns `(x_final, loss_history)`.
- **`compare_optimizers(objective, x0, n_steps, optimizers=None)`** — races
  all four (or a custom dict/list of optimizers) from the same start;
  returns `{name: {"x", "loss", "history"}}`.
- **`quadratic_loss(A, b)`** — `f(x) = 0.5 x^T A x - b^T x`, optimum at
  `x* = A^-1 b`. **`logistic_loss(X, y)`** — mean binary logistic loss
  with a numerically stable sigmoid (append a ones column to `X` for an
  intercept).

All constructors validate their hyperparameters (`lr > 0`, `momentum` in
`[0, 1)`, `decay`/`beta` in `(0, 1)`, `eps > 0`) and raise `ValueError`
otherwise; `minimize` validates `n_steps >= 1`, a non-empty `x0`, and
that the objective returns a `(loss, grad)` tuple of matching shape.

## Usage

```bash
pip install -r requirements.txt
python gradient_descent_variants/demo.py
```

## Example output

```
Quadratic f(x) = 0.5 x^T A x - b^T x, A = diag([1, 100])
start x0 = [5. 5.], optimum x* = [1.   0.01], optimum loss = -0.505000

200 steps from x0 = [5, 5]:
optimizer    final loss    loss drop  dist to optimum
-----------------------------------------------------
SGD           -0.358480  1252.858480         0.535919
Momentum      -0.505000  1253.005000         0.000088
RMSprop       -0.505000  1253.005000         0.000000
Adam          -0.488806  1252.988806         0.016789

Logistic regression, 300 steps from w0 = 0:
optimizer    final loss    loss drop
------------------------------------
SGD            0.384247     0.308900
Momentum       0.384246     0.308901
RMSprop        0.384371     0.308776
Adam           0.384246     0.308901
  SGD        train accuracy: 0.835
  Momentum   train accuracy: 0.835
  RMSprop    train accuracy: 0.835
  Adam       train accuracy: 0.835
```

The point of the quadratic: with condition number 100, plain SGD's one
learning rate is throttled by the steep direction (lr must stay below
2/100), so after 200 steps it is still 0.54 from the optimum, while
momentum, RMSprop and Adam all get there. The logistic part shows the
flip side — on a well-scaled problem all four land within a whisker of
each other; adaptivity helps conditioning, it is not magic.

One honest detail in the demo: Adam runs with `beta2=0.99` rather than
the usual `0.999`. With `0.999` the second moment remembers the huge
initial gradients for ~1000 steps and Adam crawls on this problem (it
does converge — it just needs ~500 steps); `0.99` adapts in ~100 steps.

## Run tests

```bash
python -m pytest tests/test_gradient_descent_variants.py -v
```

38 tests: hand-computed single steps for all four optimizers (SGD exact,
momentum velocity over two steps, RMSprop first step, Adam's first step
≈ `lr * sign(grad)`), no-mutation and independent-state checks, convergence
on a quadratic, all-four comparison, the quadratic optimum at `A^-1 b`,
a finite-difference check of the logistic gradient, and `ValueError`
validation for bad hyperparameters, bad shapes, and bad objectives.
