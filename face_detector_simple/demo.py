"""Demo: find the one face in a synthetic scene.

Builds an 80x100 RGB image — a skin-coloured ellipse (the face) on a
blue background, plus a small skin-coloured square and a large
skin-coloured rectangle as distractors — and reports the detected
face bounding boxes.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from face_detector_simple.detection import detect_faces, skin_mask


def make_test_image(height=80, width=100):
    """Return a synthetic scene with one face ellipse and two distractors."""
    image = np.zeros((height, width, 3))
    image[:] = (40.0, 80.0, 180.0)          # blue background
    skin = np.array([210.0, 160.0, 130.0])
    yy, xx = np.mgrid[0:height, 0:width]
    face = ((yy - 40) / 22.0) ** 2 + ((xx - 50) / 16.0) ** 2 <= 1.0
    image[face] = skin                      # the face (ellipse)
    image[5:12, 5:12] = skin                # small skin square (too small)
    image[60:78, 70:98] = skin              # large skin rectangle (not a face)
    return image, (18, 34, 63, 67)


def main():
    image, true_box = make_test_image()
    mask = skin_mask(image)
    faces = detect_faces(image)
    print(f"image: {image.shape[1]}x{image.shape[0]} RGB")
    print(f"skin pixels: {int(mask.sum())}")
    print(f"faces detected: {len(faces)}")
    for box in faces:
        print(f"  face box (top, left, bottom, right): {box}")
    print(f"true face box: {true_box}")
    return faces


if __name__ == "__main__":
    main()
