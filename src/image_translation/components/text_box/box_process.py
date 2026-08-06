import numpy as np

from image_translation.components.text_filter.translation_filter import should_translate


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
        if should_translate(aggregated_text, language=language):
            # 放入待翻译字典
            agg_box_map_text_to_translate[agg_box_id] = aggregated_text
        else:
            # 放入待保留字典
            agg_box_map_text_to_keep[agg_box_id] = aggregated_text

    print("agg_box_map_text_to_translate:", agg_box_map_text_to_translate)
    print("agg_box_map_text_to_keep:", agg_box_map_text_to_keep)

    return agg_box_map_coordinate, agg_box_map_origin, agg_box_map_text_to_translate, agg_box_map_text_to_keep
