"""Public application errors shared by protocol adapters."""


class ImageTranslationError(RuntimeError):
    """Base class for expected image-translation failures."""


class InvalidImageError(ImageTranslationError):
    """Raised when downloaded bytes are not a supported image."""


class OcrTimeoutError(ImageTranslationError):
    """Raised when OCR exceeds the configured task timeout."""


class OcrProcessingError(ImageTranslationError):
    """Raised when the OCR worker cannot process a request."""
