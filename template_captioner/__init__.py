"""Template-based image captioning from scratch (numpy only)."""

from template_captioner.captioner import (
    analyze_image,
    caption,
    count_shapes,
    dominant_colours,
    largest_shape_quadrant,
    make_test_image,
)

__all__ = [
    "analyze_image",
    "caption",
    "count_shapes",
    "dominant_colours",
    "largest_shape_quadrant",
    "make_test_image",
]
