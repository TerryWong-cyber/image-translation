import numpy as np
from PIL import ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from image_translation.components.text_box.text_format import cal_sentence_char_len
from image_translation.config import get_settings

font_cache = {}  # 字体缓存


def get_font(size, font_path=None):
    resolved_path = str(font_path or get_settings().paths.font_file)
    cache_key = (resolved_path, size)
    if cache_key not in font_cache:
        font_cache[cache_key] = ImageFont.truetype(resolved_path, size)
    return font_cache[cache_key]


def get_text_dimensions(font_size: float, text: str) -> Tuple[int, int]:
    """获取文本的宽度和高度"""
    font_obj = get_font(font_size)
    text_bbox = font_obj.getbbox(text)
    return text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]


def cal_font_size(o_box_ids, ocr_box_map_width_height, agg_text, pre_text=None):
    if not o_box_ids or len(o_box_ids) == 0:
        print("o_box_ids is empty", o_box_ids)
        return

    text_len = cal_sentence_char_len(agg_text)

    if pre_text and len(pre_text) > 0:
        pre_text_len = cal_sentence_char_len(pre_text)
        # 原文和翻译后长度相比，使用更长的长度，避免翻译后字数变少导致字体太大
        if pre_text_len > text_len:
            text_len = pre_text_len

    total_width = 0
    min_box_height = -1

    for o_box_id in o_box_ids:
        box_width, box_height = ocr_box_map_width_height.get(o_box_id, (0, 0))
        total_width += box_width
        if min_box_height == -1 or 0 < box_height < min_box_height:
            min_box_height = box_height

    if min_box_height <= 0:
        print(f"Error o_boxes:{o_box_ids} box_width not exist!", min_box_height)

    # 得出字体大小
    min_font_size = 8
    ratio = 1.8
    font_size = round(ratio * (total_width / text_len), 1)
    if font_size < min_font_size:
        font_size = min_font_size

    max_attempts = 1000
    text_width, text_height = get_text_dimensions(font_size, agg_text)
    # 自适应调整字体大小
    while ((min_box_height <= text_height or total_width <= text_width)
           and font_size > min_font_size and max_attempts > 0):
        font_size -= 2
        text_width, text_height = get_text_dimensions(font_size, agg_text)
        max_attempts -= 1

    print("font_size", font_size, "total_width", total_width, "text_len", text_len)
    return font_size


def cal_box_font_size(agg_box_map_origin, ocr_box_map_width_height, ocr_box_map_translated_text,
                      ocr_box_map_source_text):
    ocr_box_font_size_map = {}
    for agg_box_id, o_box_ids in agg_box_map_origin.items():
        sum_translated_text = ""
        sum_source_text = ""
        for o_box_id in o_box_ids:
            sum_translated_text += ocr_box_map_translated_text.get(o_box_id, "unknown text")
            sum_source_text += ocr_box_map_source_text.get(o_box_id, "unknown text")

        print("sum_source_text:", sum_source_text, "sum_translated_text:", sum_translated_text)
        font_size = cal_font_size(o_box_ids, ocr_box_map_width_height, sum_translated_text, sum_source_text)

        for o_box_id in o_box_ids:
            ocr_box_font_size_map[o_box_id] = font_size

    return ocr_box_font_size_map
