"""Public application errors shared by protocol adapters."""


class ImageTranslationError(RuntimeError):
    """Base class for expected image-translation failures."""


class InvalidImageError(ImageTranslationError):
    """Raised when downloaded bytes are not a supported image."""


class OcrTimeoutError(ImageTranslationError):
    """Raised when OCR exceeds the configured task timeout."""


class OcrProcessingError(ImageTranslationError):
    """Raised when the OCR worker cannot process a request."""


class ImageTooLargeError(InvalidImageError):
    """Input exceeds the configured byte or pixel limit."""


class InvalidImageUrlError(InvalidImageError):
    """URL is unsafe or unsupported."""


class ImageDownloadError(ImageTranslationError):
    """Source server cannot supply the image."""


class ModelUpstreamError(ImageTranslationError):
    """Model request failed or returned unusable output."""


class ModelTimeoutError(ModelUpstreamError):
    """Model request exceeded its configured timeout."""
