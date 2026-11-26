"""Character-trigram language identifier (English, French, Spanish, German)."""

from language_detector.model import (
    LANGUAGES,
    SAMPLE_TEXTS,
    LanguageDetector,
    detect_language,
    normalize,
    profile_distance,
    trigram_profile,
    trigrams,
)

__all__ = [
    "LANGUAGES",
    "SAMPLE_TEXTS",
    "LanguageDetector",
    "detect_language",
    "normalize",
    "profile_distance",
    "trigram_profile",
    "trigrams",
]
