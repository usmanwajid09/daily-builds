"""Demo: caption three synthetic scenes."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from template_captioner.captioner import analyze_image, caption, make_test_image


def show(name, img):
    info = analyze_image(img)
    print(f"--- {name} ---")
    print(f"Caption: {caption(img)}")
    print(f"Brightness: {info['brightness']:.3f} ({info['brightness_word']}), "
          f"contrast: {info['contrast']:.3f}, saturation: {info['saturation']:.3f}")
    print(f"Shapes: {info['n_shapes']} (sizes {info['shape_sizes']}), "
          f"largest in {info['largest_shape_quadrant']}")
    print(f"Dominant colours: {info['dominant_colours']}")


def main():
    red_circle = make_test_image(32, shapes=[
        ("circle", (1.0, 0.0, 0.0), (16, 16), 6)], background=0.0)
    show("Red circle on black", red_circle)

    two_shapes = make_test_image(32, shapes=[
        ("square", (0.0, 0.0, 1.0), (8, 8), 4),
        ("circle", (1.0, 1.0, 0.0), (24, 24), 5)], background=1.0)
    show("Blue square + yellow circle on white", two_shapes)

    blank = np.zeros((32, 32, 3))
    show("Blank black image", blank)


if __name__ == "__main__":
    main()
