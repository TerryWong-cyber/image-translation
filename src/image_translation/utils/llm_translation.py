import json
import re

import requests

from image_translation.components.text_box.text_format import cal_sentence_char_len, split_translated_texts
from image_translation.config import get_settings
from image_translation.utils.translation_prompts import translation_prompt


def call_inference_api_v2(
    prompt, texts_to_translate, max_new_tokens=None, batch_size=None, timeout=None, url=None
):
    settings = get_settings().llm
    max_new_tokens = max_new_tokens or settings.max_new_tokens
    batch_size = batch_size or settings.batch_size
    timeout = timeout or settings.translation_timeout_seconds
    url = url or settings.translation_url
    headers = {"Content-Type": "application/json"}
    data = {
        "texts_to_translate": texts_to_translate,
        "translate_prompt": prompt,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print("call_inference_api response:", result)
        if result.get("status") == "success":
            return result.get("responses", [])
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def call_inference_api(
    prompts, max_new_tokens=None, top_k=50, temperature=0, num_beams=1, diversity_penalty=0.0,
    batch_size=None, timeout=None, url=None
):
    settings = get_settings().llm
    max_new_tokens = max_new_tokens or settings.max_new_tokens
    batch_size = batch_size or settings.batch_size
    timeout = timeout or settings.translation_timeout_seconds
    url = url or settings.legacy_inference_url
    headers = {"Content-Type": "application/json"}
    data = {
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "temperature": temperature,
        "num_beams": num_beams,
        "diversity_penalty": diversity_penalty,
        "batch_size": batch_size
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        print("call_inference_api response:", result)
        if result.get("status") == "success":
            return result.get("responses", [])
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


def filter_multiple_translations(original_text, translated_text):
    """
    取多重释义的第一个翻译结果
    判断条件：译文长度 > 原文长度 * 2 且译文中包含中文分号；
    处理：根据分号切分，返回第一部分
    """
    # 判断是否为多重释义
    is_multiple = (
            cal_sentence_char_len(translated_text) > cal_sentence_char_len(original_text) * 2 and
            "；" in translated_text
    )

    if is_multiple:
        # 根据分号切分，取第一部分
        filtered_translated_text = translated_text.split("；")[0]
        return filtered_translated_text
    else:
        return translated_text


def llm_translate(batch_agg_box_ids, batch_aggregated_texts, context, batch_size, language="en_zh"):
    settings = get_settings()
    prompt = translation_prompt(settings.prompts, language)

    max_new_tokens = settings.llm.max_new_tokens
    # prompts = [prompt + text for text in batch_aggregated_texts]

    # llm_res = call_inference_api(prompts=prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    llm_res = call_inference_api_v2(prompt=prompt, texts_to_translate=batch_aggregated_texts,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)

    if llm_res is None:
        print("Error: llm_res is None!")
        return {}, [], {}

    unchanged_agg_box_ids = []
    translated_results_for_boxes = {}

    if len(llm_res) != len(batch_aggregated_texts):
        print(f"Error: len(llm_res) {len(llm_res)} not equal len(batch_aggregated_texts) {len(batch_aggregated_texts)}")

    agg_translated_map = {}
    for i, res in enumerate(llm_res):
        res = res.replace("\\n", " ")
        agg_text = batch_aggregated_texts[i]
        translated_text = res if res is not None else agg_text
        agg_box_id = batch_agg_box_ids[i]

        print(f"Aggregated Text: {agg_text}")
        print(f"Translated Text: {translated_text}")

        if normalize_text(agg_text) == normalize_text(translated_text):
            unchanged_agg_box_ids.append(agg_box_id)
            continue

        # 处理多重释义
        # translated_text = filter_multiple_translations(agg_text, translated_text)

        agg_translated_map[agg_box_id] = (agg_text, translated_text)
        # agg_translation_result.append({"box_id": agg_box_id, "source_text": agg_text, "translated_text": translated_text})

        # 从上下文中获取所需数据
        split_texts = split_translated_texts(
            agg_box_id, translated_text, context.agg_box_map_origin, context.ocr_box_map_width_height
        )

        origin_box_ids = context.agg_box_map_origin.get(agg_box_id)
        if origin_box_ids:
            for j, o_box_id in enumerate(origin_box_ids):
                translated_results_for_boxes[o_box_id] = split_texts[j] if j < len(split_texts) else ""
        else:
            print(f"Error: origin_box_ids not found for aggregated box {agg_box_id}")

    return translated_results_for_boxes, unchanged_agg_box_ids, agg_translated_map
