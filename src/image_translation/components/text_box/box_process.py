import re
from functools import lru_cache

import numpy as np

from image_translation.config import get_settings


@lru_cache(maxsize=1)
def _no_translate_patterns() -> tuple[re.Pattern, re.Pattern]:
    """Build matching patterns once from the managed term configuration."""
    terms = get_settings().text_translation.no_translate_terms
    escaped_terms = "|".join(re.escape(term) for term in terms)
    remove_units_regex = re.compile(
        r"(?<![a-zA-Z])(" + escaped_terms + r")\b",
        flags=re.IGNORECASE,
    )
    full_match_unit_pattern = re.compile(
        r"^\s*[-.0-9,]+\s*(?:" + escaped_terms + r")\s*$",
        flags=re.IGNORECASE,
    )
    return remove_units_regex, full_match_unit_pattern


def cal_box_id(box_coordinate):
    box_id = ''.join(f'{int(x)}{int(y)}' for x, y in box_coordinate)
    return box_id


def cal_box_width_height(box_coordinate):
    # 以左侧边和上侧边记为box的高和宽
    box_height = np.sqrt(
        (box_coordinate[0][0] - box_coordinate[3][0]) ** 2 + (box_coordinate[0][1] - box_coordinate[3][1]) ** 2)
    box_width = np.sqrt(
        (box_coordinate[0][0] - box_coordinate[1][0]) ** 2 + (box_coordinate[0][1] - box_coordinate[1][1]) ** 2)
    return box_height, box_width


def cal_box_angle_rad(box_coordinate):
    angle_rad = np.arctan2(-(box_coordinate[1][1] - box_coordinate[0][1]), box_coordinate[1][0] - box_coordinate[0][0])
    # print("angle_rad", angle_rad)
    return angle_rad


def process_ocr_res(ocr_res):
    o_box_ids = []
    ocr_box_map_text = {}
    ocr_box_map_coordinate = {}
    ocr_box_map_width_height = {}
    ocr_box_map_angle_rad = {}

    for item in ocr_res:
        o_box_coordinate = item[0]
        o_box_id = cal_box_id(o_box_coordinate)
        o_box_ids.append(o_box_id)
        o_text = item[1][0]
        ocr_box_map_text[o_box_id] = o_text  #
        ocr_box_map_coordinate[o_box_id] = o_box_coordinate  #
        ocr_box_map_angle_rad[o_box_id] = cal_box_angle_rad(o_box_coordinate)  # 倾斜度tanh值

        box_height, box_width = cal_box_width_height(o_box_coordinate)
        ocr_box_map_width_height[o_box_id] = int(np.ceil(box_width)), int(np.ceil(box_height))  # 向上取整

    return o_box_ids, ocr_box_map_text, ocr_box_map_coordinate, ocr_box_map_width_height, ocr_box_map_angle_rad


def is_need_translating(text, language="en_zh"):
    """
    判断文本是否值得翻译。
    优化思路：只要字符串中包含至少一个字母（任何语言），就认为它值得翻译。
    """
    # 首先处理 None 或空字符串的边缘情况
    if not text or not text.strip():
        return False

    remove_units_regex, full_match_unit_pattern = _no_translate_patterns()

    if full_match_unit_pattern.fullmatch(text.strip()):
        return False

    if remove_units_regex:
        cleaned_text = remove_units_regex.sub("", text)
        if not any(c.isalpha() for c in cleaned_text.strip()):
            return False

    if language == "en_zh":
        # 它确保我们只计算标准的ASCII字母（a-z, A-Z），而不会误算俄语、希腊语等其他字母。
        english_letter_count = sum(1 for char in text if char.isascii() and char.isalpha())
        return english_letter_count >= 2

    elif language == "zh_en":
        # 只有包含至少一个中文时才会翻译
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return True

    elif language == "any_zh":
        alphabetic_chars = [char for char in text if char.isalpha()]
        if not alphabetic_chars:  # 如果没有字母字符
            return False
        # 如果所有字母都是中文，则无需翻译
        if all('\u4e00' <= char <= '\u9fff' for char in alphabetic_chars):
            return False
        return True  # 否则，意味着存在非中文字母，需要翻译

    # any() 会在找到第一个True后立即停止，效率很高。
    # c.isalpha() 可以正确识别包括中文、日文、俄文等在内的所有Unicode字母。
    return any(c.isalpha() for c in text)


def process_and_filter_aggregated_boxes(agg_boxes, ocr_box_map_text, language="en_zh"):
    """
    一个统一处理聚合框的函数。
    它会遍历所有聚合框，生成ID，并根据文本内容进行分类。
    """
    agg_box_map_origin = {}
    agg_box_map_text_to_translate = {}
    # 新增：用于存储不需要翻译的文本
    agg_box_map_text_to_keep = {}
    agg_box_map_coordinate = {}

    for agg_box in agg_boxes:
        # --- 统一的ID生成点 ---
        # 假设 cal_box_id 是一个存在的辅助函数
        agg_box_id = cal_box_id(agg_box["box"])
        agg_box_map_coordinate[agg_box_id] = agg_box["box"]

        # 生成原始框ID列表
        o_box_ids = [cal_box_id(o_box) for o_box in agg_box["original_boxs"]]
        agg_box_map_origin[agg_box_id] = o_box_ids

        # 拼接文本
        ocr_texts = [ocr_box_map_text.get(o_id, "") for o_id in o_box_ids]
        aggregated_text = " ".join(filter(None, ocr_texts))

        # --- 统一的分类点 ---
        if is_need_translating(aggregated_text, language=language):
            # 放入待翻译字典
            agg_box_map_text_to_translate[agg_box_id] = aggregated_text
        else:
            # 放入待保留字典
            agg_box_map_text_to_keep[agg_box_id] = aggregated_text

    print("agg_box_map_text_to_translate:", agg_box_map_text_to_translate)
    print("agg_box_map_text_to_keep:", agg_box_map_text_to_keep)

    return agg_box_map_coordinate, agg_box_map_origin, agg_box_map_text_to_translate, agg_box_map_text_to_keep
