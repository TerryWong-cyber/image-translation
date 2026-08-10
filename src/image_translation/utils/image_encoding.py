from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ImageEncodingSpec:
    format_name: str
    codec_extension: str
    output_extension: str
    content_type: str
    parameters: tuple[tuple[str, int], ...]


_FORMAT_CONFIG = {
    "jpeg": {
        "codec_extension": ".jpg",
        "extensions": (".jpg", ".jpeg"),
        "content_type": "image/jpeg",
    },
    "png": {
        "codec_extension": ".png",
        "extensions": (".png",),
        "content_type": "image/png",
    },
    "webp": {
        "codec_extension": ".webp",
        "extensions": (".webp",),
        "content_type": "image/webp",
    },
    "bmp": {
        "codec_extension": ".bmp",
        "extensions": (".bmp",),
        "content_type": "image/bmp",
    },
    "tiff": {
        "codec_extension": ".tiff",
        "extensions": (".tif", ".tiff"),
        "content_type": "image/tiff",
    },
}

_CONTENT_TYPE_FORMATS = {
    config["content_type"]: format_name
    for format_name, config in _FORMAT_CONFIG.items()
}
_CONTENT_TYPE_FORMATS["image/jpg"] = "jpeg"

_EXTENSION_FORMATS = {
    extension: format_name
    for format_name, config in _FORMAT_CONFIG.items()
    for extension in config["extensions"]
}


def _detect_format_from_bytes(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes.startswith(b"BM"):
        return "bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    return None


def _encoding_parameters(format_name: str) -> tuple[tuple[str, int], ...]:
    if format_name == "jpeg":
        return (("IMWRITE_JPEG_QUALITY", 95),)
    if format_name == "png":
        return (("IMWRITE_PNG_COMPRESSION", 3),)
    if format_name == "webp":
        return (("IMWRITE_WEBP_QUALITY", 95),)
    return ()


def resolve_image_encoding(
    image_bytes: bytes,
    file_name: str,
    content_type: str | None = None,
) -> ImageEncodingSpec:
    """Resolve a safe output codec, preferring the image's actual byte format."""
    source_extension = PurePosixPath(file_name).suffix.lower()
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()

    format_name = (
        _detect_format_from_bytes(image_bytes)
        or _CONTENT_TYPE_FORMATS.get(normalized_content_type)
        or _EXTENSION_FORMATS.get(source_extension)
        or "jpeg"
    )
    config = _FORMAT_CONFIG[format_name]
    output_extension = (
        source_extension
        if source_extension in config["extensions"]
        else config["codec_extension"]
    )
    return ImageEncodingSpec(
        format_name=format_name,
        codec_extension=config["codec_extension"],
        output_extension=output_extension,
        content_type=config["content_type"],
        parameters=_encoding_parameters(format_name),
    )


def encode_image(image, spec: ImageEncodingSpec) -> bytes:
    import cv2

    encoding_parameters = [
        value
        for constant_name, parameter_value in spec.parameters
        for value in (getattr(cv2, constant_name), parameter_value)
    ]
    success, buffer = cv2.imencode(
        spec.codec_extension,
        image,
        encoding_parameters,
    )
    if not success:
        raise ValueError(f"Failed to encode image as {spec.format_name}")
    return buffer.tobytes()


def replace_image_extension(file_name: str, extension: str) -> str:
    return str(PurePosixPath(file_name).with_suffix(extension))
