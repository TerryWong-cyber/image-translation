"""Protocol-neutral contracts for an image translation request."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


def _validate_bucket(value: str) -> str:
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("bucket must be a single object-storage bucket name")
    return value


def _validate_object_key(value: str) -> str:
    if value.startswith(("/", "\\")) or "\\" in value or "\x00" in value:
        raise ValueError("object key must be a relative POSIX path")
    if ".." in PurePosixPath(value).parts:
        raise ValueError("object key cannot contain '..' path segments")
    return value


class TranslationCommand(BaseModel):
    """Application command accepted by both REST and MCP."""

    model_config = ConfigDict(str_strip_whitespace=True)

    bucket: str = Field(min_length=1, max_length=255)
    image_key: str = Field(min_length=1, max_length=2048)
    language: TranslationLanguage = "en_zh"
    save_bucket: str | None = Field(default=None, min_length=1, max_length=255)
    output_key: str | None = Field(default=None, min_length=1, max_length=2048)
    segment: bool = False

    @field_validator("bucket", "save_bucket")
    @classmethod
    def validate_bucket(cls, value: str | None) -> str | None:
        return _validate_bucket(value) if value is not None else None

    @field_validator("image_key", "output_key")
    @classmethod
    def validate_object_key(cls, value: str | None) -> str | None:
        return _validate_object_key(value) if value is not None else None


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

    def to_rest_payload(self) -> dict[str, object]:
        """Keep the legacy REST response shape stable."""
        return {
            "status": self.status,
            "source_url": self.source_url,
            "translated_url": self.translated_url,
            "data": [item.model_dump(mode="json") for item in self.data],
        }
