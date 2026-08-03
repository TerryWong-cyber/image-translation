import numpy as np


def cal_boxs_length_ratios(box_ids, ocr_box_map_width_height):
    box_width_array = []

    # 计算每个 box 的长度
    for box_id in box_ids:
        box_width, box_height = ocr_box_map_width_height.get(box_id)
        if box_width is None or box_width == 0:
            print(f"Warning: box {box_id} width {box_width} invalid")

        box_width_array.append(box_width)

    total_width = sum(box_width_array)

    if total_width == 0:
        return [0] * len(box_ids)

    # 计算每个长度的比率
    ratios = [width / total_width for width in box_width_array]

    return ratios


def char_len(char) -> float:
    if not char:  # 检查空字符
        return 0

    # 中文字符 (CJK 统一汉字)
    if '\u4e00' <= char <= '\u9fff':
        return 1.8

    # 英文字符 (A-Z, a-z)
    if char.isalpha() and char.isascii():
        return 1

    # 中文标点符号 (常见)
    if char in '、，。！？；：（）【】《》':
        return 1.8

    return 1


def cal_sentence_char_len(sentence):
    char_length = 0
    for char in sentence:
        char_length += char_len(char)

    return char_length


def split_translated_texts(agg_box_id, translated_text, agg_box_map_origin,ocr_box_map_width_height):
    o_box_ids = agg_box_map_origin.get(agg_box_id)
    if not o_box_ids or len(o_box_ids) == 0:
        return [translated_text]

    text_length = 0
    for char in translated_text:
        text_length += char_len(char)
    ratios = cal_boxs_length_ratios(o_box_ids, ocr_box_map_width_height)

    split_ratios = []

    for ratio in ratios:
        split_ratios.append(text_length * ratio)

    j = 0
    cur = 0
    cur_length = 0
    split_texts = []
    for i, char in enumerate(translated_text):
        cur_length += char_len(char)  # 累积当前段宽度

        if j < len(split_ratios) and cur_length >= split_ratios[j]:
            split_texts.append(translated_text[cur:i + 1])  # 添加当前段
            cur = i + 1  # 更新起点
            cur_length = 0  # 重置长度
            j += 1  # 移动到下一个比例

        # 添加最后一段（如果有剩余文本）
    if cur < len(translated_text):
        split_texts.append(translated_text[cur:])

    # print("ratios:", ratios)
    # print("split_ratios:", split_ratios)
    # print("split_texts:", split_texts)
    # print(
    #     f"len(split_texts):{len(split_texts)},len(split_ratios):{len(split_ratios)},len(o_box_ids):{len(o_box_ids)}")
    # 按理说分割后的数量会与原始文本框数量一致
    if len(split_texts) != len(o_box_ids):
        print(f"Warning: Split mismatch, expected {len(o_box_ids)} segments, got {len(split_texts)}")

    return split_texts
