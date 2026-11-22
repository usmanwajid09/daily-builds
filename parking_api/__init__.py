"""Flask parking API: spots, check-in/check-out, fees and occupancy summary."""
from .app import create_app

__all__ = ["create_app"]
