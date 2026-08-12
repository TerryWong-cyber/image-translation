"""Supported translation targets and their prompt/font configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationTarget:
    code: str
    english_name: str
    font_setting: str


AUTO_TRANSLATION_TARGETS = {
    "zh": TranslationTarget("zh", "Simplified Chinese (compatibility alias)", "font_zh_cn_file"),
    "zh_cn": TranslationTarget("zh_cn", "Simplified Chinese", "font_zh_cn_file"),
    "zh_tw": TranslationTarget("zh_tw", "Traditional Chinese (Taiwan)", "font_zh_tw_file"),
    "en": TranslationTarget("en", "English", "font_latin_file"),
    "ja": TranslationTarget("ja", "Japanese", "font_ja_file"),
    "ko": TranslationTarget("ko", "Korean", "font_ko_file"),
    "fr": TranslationTarget("fr", "French", "font_latin_file"),
    "ar": TranslationTarget("ar", "Arabic", "font_arabic_file"),
    "de": TranslationTarget("de", "German", "font_latin_file"),
    "ru": TranslationTarget("ru", "Russian", "font_cyrillic_file"),
    "nl": TranslationTarget("nl", "Dutch", "font_latin_file"),
    "pt": TranslationTarget("pt", "Portuguese", "font_latin_file"),
    "th": TranslationTarget("th", "Thai", "font_thai_file"),
    "es": TranslationTarget("es", "Spanish", "font_latin_file"),
    "it": TranslationTarget("it", "Italian", "font_latin_file"),
    "vi": TranslationTarget("vi", "Vietnamese", "font_vietnamese_file"),
    "id": TranslationTarget("id", "Indonesian", "font_latin_file"),
}

LEGACY_TRANSLATION_TARGETS = {
    "en_zh": AUTO_TRANSLATION_TARGETS["zh"],
    "zh_en": AUTO_TRANSLATION_TARGETS["en"],
}


def translation_target(language: str) -> TranslationTarget:
    """Return target-language metadata for an accepted translation direction."""
    if language in LEGACY_TRANSLATION_TARGETS:
        return LEGACY_TRANSLATION_TARGETS[language]
    if language.startswith("any_"):
        target = AUTO_TRANSLATION_TARGETS.get(language.removeprefix("any_"))
        if target is not None:
            return target
    raise ValueError(f"Unsupported translation language: {language}")


def translation_prompt(prompts, language: str) -> str:
    """Resolve a validated translation direction to its configured prompt."""
    try:
        translation_target(language)
        return getattr(prompts, f"translate_{language}")
    except AttributeError as exc:
        raise ValueError(f"Unsupported translation language: {language}") from exc
