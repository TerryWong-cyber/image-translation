"""Protocol-neutral contracts for an image translation request."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


TranslationLanguage = Literal[
    "en_zh",
    "zh_en",
    "any_zh",
    "any_zh_cn",
    "any_zh_tw",
    "any_en",
    "any_ko",
    "any_ja",
    "any_th",
    "any_fr",
    "any_ar",
    "any_de",
    "any_ru",
    "any_nl",
    "any_pt",
    "any_es",
    "any_it",
    "any_vi",
    "any_id",
]


class TranslationItem(BaseModel):
    """One translated text region in the image."""

    coordinate: list[list[float]]
    source_text: str
    translated_text: str


class TranslationResult(BaseModel):
    """Structured result returned by the shared application service."""

    status: Literal["success"] = "success"
    request_id: str
    source_url: str
    translated_url: str
    data: list[TranslationItem]
    duration_ms: int = Field(ge=0)


class UrlTranslationRequest(BaseModel):
    """Shared REST/MCP input; storage details belong to server configuration."""

    model_config = ConfigDict(extra="forbid")
    image_url: HttpUrl = Field(max_length=4096)
    language: TranslationLanguage = "en_zh"
    segment: bool = False

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.fragment:
            raise ValueError("Image URL cannot contain credentials or a fragment")
        return value
