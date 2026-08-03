import json
import re

import requests

from src.image_translation.config import get_settings


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
    headers = {"Content-Type": "application/json"}
    data = {
        "system_prompt": "",
        "user_prompt": prompt,
        "image_base64": image_base64
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print("call_inference_api response:", result)
        if result.get("status") == "success":
            return result.get("response", "")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# 新增一个用于批量调用的API函数
def call_llm_recognition_api_batch(base64_images, prompt=None, max_new_tokens=None, timeout=None, url=None):
    settings = get_settings()
    prompt = prompt or settings.prompts.recognition
    max_new_tokens = max_new_tokens or settings.llm.max_new_tokens
    timeout = timeout or settings.llm.recognition_timeout_seconds
    url = url or settings.llm.recognition_batch_url
    headers = {"Content-Type": "application/json"}
    data = {
        "user_prompt": prompt,
        "image_base64_list": base64_images,
        "max_new_tokens": max_new_tokens
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print(f"call_inference_api_batch response (total {len(base64_images)} images):", result)
        if result.get("status") == "success":
            return result.get("responses", [])  # 返回识别结果列表
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def normalize_text(text: str) -> str:
    """一个辅助函数，用于清理和标准化文本。"""
    if not isinstance(text, str) or not text:
        return ""

    # 1. 转为小写
    text = text.lower()

    # 2. 移除所有不是字母或数字的字符。空格、换行、标点符号等，会保留所有语言的字母（包括中文、日文等）和数字。
    normalized = re.sub(r'[^\w]', '', text, flags=re.UNICODE)

    return normalized
