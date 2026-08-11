"""Typed runtime configuration for the image-translation service.

Only this module reads environment variables.  Application code consumes the
grouped settings below, which keeps deployment details out of business logic.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping, cast
from urllib.parse import urljoin, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse the small, portable subset of dotenv syntax used by this project."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"Invalid dotenv entry at {path}:{line_number}")

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ConfigurationError(f"Invalid environment variable name at {path}:{line_number}")

        raw_value = raw_value.strip()
        if not raw_value:
            values[key] = ""
            continue
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ConfigurationError(f"Invalid dotenv value at {path}:{line_number}: {exc}") from exc
        values[key] = " ".join(parsed) if parsed else ""
    return values


def _environment() -> dict[str, str]:
    env_file = Path(os.environ.get("IMAGE_TRANSLATION_ENV_FILE", PROJECT_ROOT / ".env")).expanduser()
    values = _parse_env_file(env_file)
    values.update(os.environ)
    return values


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"Missing required setting {name}. Copy .env.example to .env and configure it."
        )
    return value


def _int(env: Mapping[str, str], name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = _required(env, name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, *, minimum: float = 0.0) -> float:
    raw = _required(env, name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str) -> bool:
    raw = _required(env, name).lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {raw!r}")


def _optional_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    if not env.get(name, "").strip():
        return default
    return _bool(env, name)


def _optional_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if not env.get(name, "").strip():
        return default
    return _int(env, name, minimum=minimum, maximum=maximum)


OcrDevice = Literal["cpu", "gpu"]
OcrVersion = Literal["PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"]


def _ocr_device(env: Mapping[str, str]) -> OcrDevice:
    value = env.get("OCR_DEVICE", "cpu").strip().lower() or "cpu"
    if value not in {"cpu", "gpu"}:
        raise ConfigurationError(f"OCR_DEVICE must be one of cpu, gpu, got {value!r}")
    return cast(OcrDevice, value)


def _ocr_version(env: Mapping[str, str]) -> OcrVersion:
    value = env.get("OCR_VERSION", "PP-OCRv4").strip() or "PP-OCRv4"
    choices = {"PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"}
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ConfigurationError(f"OCR_VERSION must be one of {expected}, got {value!r}")
    return cast(OcrVersion, value)


def _csv(env: Mapping[str, str], name: str, default: str = "") -> tuple[str, ...]:
    raw = env.get(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _url(base_url: str, endpoint: str) -> str:
    return urljoin(f"{base_url.rstrip('/')}/", endpoint.lstrip("/"))


def _absolute_url(env: Mapping[str, str], name: str) -> str:
    value = _required(env, name).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute HTTP(S) URL, got {value!r}")
    return value


def _route(env: Mapping[str, str], name: str) -> str:
    value = _required(env, name)
    if not value.startswith("/"):
        raise ConfigurationError(f"{name} must start with '/', got {value!r}")
    return value


def _oss_file_route(env: Mapping[str, str], name: str) -> str:
    value = _route(env, name)
    missing = {placeholder for placeholder in ("{bucket}", "{key}") if placeholder not in value}
    if missing:
        raise ConfigurationError(f"{name} must contain placeholders: {', '.join(sorted(missing))}")
    return value


@dataclass(frozen=True)
class ServerSettings:
    app: str
    host: str
    port: int
    reload: bool
    reload_dirs: tuple[str, ...]


@dataclass(frozen=True)
class ApiSettings:
    image_translate_path: str


@dataclass(frozen=True)
class McpSettings:
    enabled: bool
    path: str
    stateless_http: bool
    json_response: bool
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    max_request_body_size: int
    max_concurrency: int


@dataclass(frozen=True)
class HttpSettings:
    request_timeout_seconds: float
    max_connections: int
    max_keepalive_connections: int
    keepalive_expiry_seconds: float


@dataclass(frozen=True)
class LlmSettings:
    api_base_url: str
    legacy_api_base_url: str
    translation_endpoint: str
    legacy_inference_endpoint: str
    recognition_endpoint: str
    recognition_batch_endpoint: str
    translation_timeout_seconds: float
    recognition_timeout_seconds: float
    max_new_tokens: int
    batch_size: int

    @property
    def translation_url(self) -> str:
        return _url(self.api_base_url, self.translation_endpoint)

    @property
    def legacy_inference_url(self) -> str:
        return _url(self.legacy_api_base_url, self.legacy_inference_endpoint)

    @property
    def recognition_url(self) -> str:
        return _url(self.api_base_url, self.recognition_endpoint)

    @property
    def recognition_batch_url(self) -> str:
        return _url(self.api_base_url, self.recognition_batch_endpoint)


@dataclass(frozen=True)
class OssSettings:
    base_url: str
    upload_endpoint: str
    file_endpoint: str

    def upload_url(self) -> str:
        return _url(self.base_url, self.upload_endpoint)

    def file_url(self, bucket_name: str, file_key: str) -> str:
        endpoint = self.file_endpoint.format(bucket=bucket_name, key=file_key)
        return _url(self.base_url, endpoint)


@dataclass(frozen=True)
class OcrSettings:
    device: OcrDevice
    gpu_id: int
    language: str
    version: OcrVersion
    use_angle_classifier: bool
    unclip_ratio: float
    task_timeout_seconds: float
    worker_shutdown_timeout_seconds: float
    box_height_tolerance_ratio: float
    require_local_models: bool
    textline_orientation_model_dir: Path | None
    detection_model_dir: Path | None
    recognition_model_dir: Path | None


@dataclass(frozen=True)
class TextTranslationSettings:
    no_translate_terms_file: Path
    no_translate_terms: tuple[str, ...]


@dataclass(frozen=True)
class PromptSettings:
    directory: Path
    vision_file: Path
    translations_file: Path
    recognition: str
    language_detection: str
    translate_en_zh: str
    translate_zh_en: str
    translate_any_zh: str


@dataclass(frozen=True)
class PathSettings:
    font_file: Path
    test_input_dir: Path
    test_output_dir: Path
    test_translation_output_dir: Path


@dataclass(frozen=True)
class Settings:
    server: ServerSettings
    api: ApiSettings
    mcp: McpSettings
    http: HttpSettings
    llm: LlmSettings
    oss: OssSettings
    ocr: OcrSettings
    text_translation: TextTranslationSettings
    prompts: PromptSettings
    paths: PathSettings


def _path(env: Mapping[str, str], name: str) -> Path:
    path = Path(_required(env, name)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_directory(env: Mapping[str, str], name: str) -> Path | None:
    raw = env.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.is_dir():
        raise ConfigurationError(f"{name} directory does not exist: {path}")
    return path


def _mcp_path(env: Mapping[str, str]) -> str:
    value = env.get("MCP_PATH", "/mcp").strip() or "/mcp"
    if not value.startswith("/"):
        raise ConfigurationError(f"MCP_PATH must start with '/', got {value!r}")
    normalized = value.rstrip("/") or "/"
    if normalized == "/":
        raise ConfigurationError("MCP_PATH cannot be '/', because it would shadow the REST API")
    return normalized


def _directory(env: Mapping[str, str], name: str) -> Path:
    path = _path(env, name)
    if not path.is_dir():
        raise ConfigurationError(f"{name} is not a directory: {path}")
    return path


def _no_translate_terms(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise ConfigurationError(f"NO_TRANSLATE_TERMS_FILE does not exist: {path}")
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in NO_TRANSLATE_TERMS_FILE {path}: {exc.msg}") from exc

    categories = raw_data.get("categories") if isinstance(raw_data, dict) else None
    if not isinstance(categories, dict):
        raise ConfigurationError(f"NO_TRANSLATE_TERMS_FILE must contain a 'categories' object: {path}")

    terms: list[str] = []
    for category, category_terms in categories.items():
        if not isinstance(category, str) or not category.strip():
            raise ConfigurationError(f"NO_TRANSLATE_TERMS_FILE has an invalid category name: {path}")
        if not isinstance(category_terms, list):
            raise ConfigurationError(f"Category {category!r} must be a list in {path}")
        for term in category_terms:
            if not isinstance(term, str) or not term.strip():
                raise ConfigurationError(f"Category {category!r} contains an invalid term in {path}")
            if term not in terms:
                terms.append(term)

    if not terms:
        raise ConfigurationError(f"NO_TRANSLATE_TERMS_FILE contains no terms: {path}")
    return tuple(terms)


def _prompt_document(directory: Path, filename: str) -> tuple[Path, dict[str, object]]:
    path = directory / filename
    if not path.is_file():
        raise ConfigurationError(f"Prompt file does not exist: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid prompt JSON in {path}: {exc.msg}") from exc
    if not isinstance(content, dict):
        raise ConfigurationError(f"Prompt JSON must be an object: {path}")
    return path, content


def _prompt_value(document: dict[str, object], key: str, path: Path) -> str:
    entry = document.get(key)
    if not isinstance(entry, dict):
        raise ConfigurationError(f"Prompt JSON is missing object {key!r}: {path}")
    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError(f"Prompt {key!r} must have a non-empty 'prompt' string: {path}")
    return prompt


def _prompt_settings(env: Mapping[str, str]) -> PromptSettings:
    directory = _directory(env, "PROMPTS_DIR")
    vision_file, vision_document = _prompt_document(directory, "vision.json")
    translations_file, translations_document = _prompt_document(directory, "translations.json")
    return PromptSettings(
        directory=directory,
        vision_file=vision_file,
        translations_file=translations_file,
        recognition=_prompt_value(vision_document, "recognition", vision_file),
        language_detection=_prompt_value(vision_document, "language_detection", vision_file),
        translate_en_zh=_prompt_value(translations_document, "translate_en_zh", translations_file),
        translate_zh_en=_prompt_value(translations_document, "translate_zh_en", translations_file),
        translate_any_zh=_prompt_value(translations_document, "translate_any_zh", translations_file),
    )


def _ocr_settings(env: Mapping[str, str]) -> OcrSettings:
    use_angle_classifier = _bool(env, "OCR_USE_ANGLE_CLASSIFIER")
    require_local_models = _optional_bool(env, "OCR_REQUIRE_LOCAL_MODELS", False)
    orientation_dir = _optional_directory(env, "OCR_TEXTLINE_MODEL_DIR")
    detection_dir = _optional_directory(env, "OCR_DETECTION_MODEL_DIR")
    recognition_dir = _optional_directory(env, "OCR_RECOGNITION_MODEL_DIR")

    if (detection_dir is None) != (recognition_dir is None):
        raise ConfigurationError(
            "OCR_DETECTION_MODEL_DIR and OCR_RECOGNITION_MODEL_DIR must be configured together"
        )
    if require_local_models:
        missing = []
        if detection_dir is None:
            missing.append("OCR_DETECTION_MODEL_DIR")
        if recognition_dir is None:
            missing.append("OCR_RECOGNITION_MODEL_DIR")
        if use_angle_classifier and orientation_dir is None:
            missing.append("OCR_TEXTLINE_MODEL_DIR")
        if missing:
            raise ConfigurationError(
                "OCR_REQUIRE_LOCAL_MODELS=true requires: " + ", ".join(missing)
            )

    return OcrSettings(
        device=_ocr_device(env),
        gpu_id=_optional_int(env, "OCR_GPU_ID", 0, minimum=0),
        language=_required(env, "OCR_LANGUAGE"),
        version=_ocr_version(env),
        use_angle_classifier=use_angle_classifier,
        unclip_ratio=_float(env, "OCR_DET_DB_UNCLIP_RATIO"),
        task_timeout_seconds=_float(env, "OCR_TASK_TIMEOUT_SECONDS"),
        worker_shutdown_timeout_seconds=_float(env, "OCR_WORKER_SHUTDOWN_TIMEOUT_SECONDS"),
        box_height_tolerance_ratio=_float(env, "OCR_BOX_HEIGHT_TOLERANCE_RATIO"),
        require_local_models=require_local_models,
        textline_orientation_model_dir=orientation_dir,
        detection_model_dir=detection_dir,
        recognition_model_dir=recognition_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = _environment()
    return Settings(
        server=ServerSettings(
            app=_required(env, "SERVER_APP"),
            host=_required(env, "SERVER_HOST"),
            port=_int(env, "SERVER_PORT", minimum=1, maximum=65535),
            reload=_bool(env, "SERVER_RELOAD"),
            reload_dirs=tuple(
                item.strip() for item in _required(env, "SERVER_RELOAD_DIRS").split(",") if item.strip()
            ),
        ),
        api=ApiSettings(
            image_translate_path=_route(env, "API_IMAGE_TRANSLATE_PATH"),
        ),
        mcp=McpSettings(
            enabled=_optional_bool(env, "MCP_ENABLED", True),
            path=_mcp_path(env),
            stateless_http=_optional_bool(env, "MCP_STATELESS_HTTP", True),
            json_response=_optional_bool(env, "MCP_JSON_RESPONSE", True),
            allowed_hosts=_csv(
                env,
                "MCP_ALLOWED_HOSTS",
                "127.0.0.1,127.0.0.1:*,localhost,localhost:*,[::1],[::1]:*",
            ),
            allowed_origins=_csv(
                env,
                "MCP_ALLOWED_ORIGINS",
                (
                    "http://127.0.0.1,http://127.0.0.1:*,"
                    "http://localhost,http://localhost:*,"
                    "http://[::1],http://[::1]:*"
                ),
            ),
            max_request_body_size=_optional_int(
                env,
                "MCP_MAX_REQUEST_BODY_SIZE",
                1024 * 1024,
                minimum=1024,
            ),
            max_concurrency=_optional_int(
                env,
                "MCP_MAX_CONCURRENCY",
                2,
                minimum=1,
            ),
        ),
        http=HttpSettings(
            request_timeout_seconds=_float(env, "HTTP_REQUEST_TIMEOUT_SECONDS"),
            max_connections=_int(env, "HTTP_MAX_CONNECTIONS", minimum=1),
            max_keepalive_connections=_int(env, "HTTP_MAX_KEEPALIVE_CONNECTIONS", minimum=0),
            keepalive_expiry_seconds=_float(env, "HTTP_KEEPALIVE_EXPIRY_SECONDS"),
        ),
        llm=LlmSettings(
            api_base_url=_absolute_url(env, "LLM_API_BASE_URL"),
            legacy_api_base_url=_absolute_url(env, "LLM_LEGACY_API_BASE_URL"),
            translation_endpoint=_route(env, "LLM_TRANSLATION_ENDPOINT"),
            legacy_inference_endpoint=_route(env, "LLM_LEGACY_INFERENCE_ENDPOINT"),
            recognition_endpoint=_route(env, "LLM_RECOGNITION_ENDPOINT"),
            recognition_batch_endpoint=_route(env, "LLM_RECOGNITION_BATCH_ENDPOINT"),
            translation_timeout_seconds=_float(env, "LLM_TRANSLATION_TIMEOUT_SECONDS"),
            recognition_timeout_seconds=_float(env, "LLM_RECOGNITION_TIMEOUT_SECONDS"),
            max_new_tokens=_int(env, "LLM_MAX_NEW_TOKENS", minimum=1),
            batch_size=_int(env, "LLM_BATCH_SIZE", minimum=1),
        ),
        oss=OssSettings(
            base_url=_absolute_url(env, "OSS_BASE_URL"),
            upload_endpoint=_route(env, "OSS_UPLOAD_ENDPOINT"),
            file_endpoint=_oss_file_route(env, "OSS_FILE_ENDPOINT"),
        ),
        ocr=_ocr_settings(env),
        text_translation=TextTranslationSettings(
            no_translate_terms_file=_path(env, "NO_TRANSLATE_TERMS_FILE"),
            no_translate_terms=_no_translate_terms(_path(env, "NO_TRANSLATE_TERMS_FILE")),
        ),
        prompts=_prompt_settings(env),
        paths=PathSettings(
            font_file=_path(env, "FONT_FILE"),
            test_input_dir=_path(env, "TEST_INPUT_DIR"),
            test_output_dir=_path(env, "TEST_OUTPUT_DIR"),
            test_translation_output_dir=_path(env, "TEST_TRANSLATION_OUTPUT_DIR"),
        ),
    )
