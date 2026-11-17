"""Running-average background subtraction from scratch (numpy only)."""

from background_subtraction.background_subtraction import (
    BackgroundSubtractor,
    foreground_mask,
    running_average_update,
)

__all__ = [
    "BackgroundSubtractor",
    "foreground_mask",
    "running_average_update",
]
