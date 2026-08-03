"""Text-translation eligibility rules independent of OCR box processing."""

from __future__ import annotations

import re
from functools import lru_cache

from src.image_translation.config import get_settings


@lru_cache(maxsize=1)
def _no_translate_patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile the configured no-translation terms once per process."""
    terms = get_settings().text_translation.no_translate_terms
    escaped_terms = "|".join(re.escape(term) for term in terms)
    return (
        re.compile(r"(?<![a-zA-Z])(" + escaped_terms + r")\b", flags=re.IGNORECASE),
        re.compile(
            r"^\s*[-.0-9,]+\s*(?:" + escaped_terms + r")\s*$",
            flags=re.IGNORECASE,
        ),
    )


def should_translate(text: str | None, language: str = "en_zh") -> bool:
    """Return whether text should be sent to the translation model.

    The configured terms first exclude standalone units and fixed technical
    terms. Language-specific rules are then applied to the remaining text.
    """
    if not text or not text.strip():
        return False

    remove_units_regex, full_match_unit_pattern = _no_translate_patterns()
    if full_match_unit_pattern.fullmatch(text.strip()):
        return False

    cleaned_text = remove_units_regex.sub("", text)
    if not any(char.isalpha() for char in cleaned_text.strip()):
        return False

    if language == "en_zh":
        english_letter_count = sum(1 for char in text if char.isascii() and char.isalpha())
        return english_letter_count >= 2

    if language == "zh_en":
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    if language == "any_zh":
        alphabetic_chars = [char for char in text if char.isalpha()]
        return bool(alphabetic_chars) and not all(
            "\u4e00" <= char <= "\u9fff" for char in alphabetic_chars
        )

    return any(char.isalpha() for char in text)
