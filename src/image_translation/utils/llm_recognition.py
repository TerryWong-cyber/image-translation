import re

from image_translation.clients.llm_client import request_chat_batch
from image_translation.config import get_settings


def call_llm_recognition_api(
    image_base64, prompt=None, max_new_tokens=None, batch_size=None, timeout=None, url=None
):
    return call_llm_recognition_api_batch(
        [image_base64], prompt=prompt, max_new_tokens=max_new_tokens, timeout=timeout, url=url,
        batch_size=batch_size,
    )[0]


def call_llm_recognition_api_batch(
    base64_images, prompt=None, max_new_tokens=None, timeout=None, url=None, batch_size=None
):
    settings = get_settings()
    content = [[
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," if image.startswith("iVBOR") else "data:image/jpeg;base64,") + image}},
        {"type": "text", "text": prompt or settings.prompts.recognition},
    ] for image in base64_images]
    return request_chat_batch(
        url or settings.llm.recognition_batch_url, content,
        timeout or settings.llm.recognition_timeout_seconds, model=settings.llm.model,
        api_key=settings.llm.api_key, max_tokens=max_new_tokens or settings.llm.max_new_tokens,
        batch_size=batch_size or settings.llm.batch_size,
    )


def normalize_text(text: str) -> str:
    """一个辅助函数，用于清理和标准化文本。"""
    if not isinstance(text, str) or not text:
        return ""

    # 1. 转为小写
    text = text.lower()

    # 2. 移除所有不是字母或数字的字符。空格、换行、标点符号等，会保留所有语言的字母（包括中文、日文等）和数字。
    normalized = re.sub(r'[^\w]', '', text, flags=re.UNICODE)

    return normalized
