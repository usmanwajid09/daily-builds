"""Generates synthetic data, fits LinearRegressionGD, and saves a plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linreg_gd.model import LinearRegressionGD


def main():
    rng = np.random.default_rng(seed=42)
    true_w, true_b = 2.5, -1.0

    x = rng.uniform(0, 10, size=80)
    noise = rng.normal(0, 1.5, size=80)
    y = true_w * x + true_b + noise

    model = LinearRegressionGD(learning_rate=0.02, n_iters=1500).fit(x, y)

    print(f"True:    w={true_w:.3f}, b={true_b:.3f}")
    print(f"Learned: w={model.w_:.3f}, b={model.b_:.3f}")
    print(f"Final training loss (MSE): {model.loss_history_[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x, y, alpha=0.6, label="data")
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, model.predict(x_line), color="red", label="fitted line")
    axes[0].set_title("Fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(model.loss_history_)
    axes[1].set_title("Training loss (MSE)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")

    fig.tight_layout()
    fig.savefig("fit_plot.png", dpi=120)
    print("Saved fit_plot.png")


if __name__ == "__main__":
    main()
