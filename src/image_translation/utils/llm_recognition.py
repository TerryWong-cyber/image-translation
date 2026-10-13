import re

from image_translation.clients.llm_client import request_model

from image_translation.config import get_settings


# deprecated
def call_llm_recognition_api(
        image_base64, prompt=None, max_new_tokens=None, batch_size=None, timeout=None, url=None
):
    settings = get_settings()
    prompt = prompt or settings.prompts.recognition
    max_new_tokens = max_new_tokens or settings.llm.max_new_tokens
    batch_size = batch_size or settings.llm.batch_size
    timeout = timeout or settings.llm.recognition_timeout_seconds
    url = url or settings.llm.recognition_url
    data = {
        "system_prompt": "",
        "user_prompt": prompt,
        "image_base64": image_base64
    }
    return request_model(url, data, timeout, 1, response_field="response")[0]


# 新增一个用于批量调用的API函数
def call_llm_recognition_api_batch(base64_images, prompt=None, max_new_tokens=None, timeout=None, url=None):
    settings = get_settings()
    prompt = prompt or settings.prompts.recognition
    max_new_tokens = max_new_tokens or settings.llm.max_new_tokens
    timeout = timeout or settings.llm.recognition_timeout_seconds
    url = url or settings.llm.recognition_batch_url
    data = {
        "user_prompt": prompt,
        "image_base64_list": base64_images,
        "max_new_tokens": max_new_tokens
    }
    return request_model(url, data, timeout, len(base64_images))


def normalize_text(text: str) -> str:
    """一个辅助函数，用于清理和标准化文本。"""
    if not isinstance(text, str) or not text:
        return ""

    # 1. 转为小写
    text = text.lower()

    # 2. 移除所有不是字母或数字的字符。空格、换行、标点符号等，会保留所有语言的字母（包括中文、日文等）和数字。
    normalized = re.sub(r'[^\w]', '', text, flags=re.UNICODE)

    return normalized
