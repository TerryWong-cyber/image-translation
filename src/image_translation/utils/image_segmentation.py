import time

import cv2
import numpy as np

# has_line_between 和 merge_boxes 函数保持不变
def has_line_between(box1, box2, lines_mask):
    """
    检查两个边界框之间或重叠区域是否存在线条。
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    union_x = min(x1, x2)
    union_y = min(y1, y2)
    union_x_end = max(x1 + w1, x2 + w2)
    union_y_end = max(y1 + h1, y2 + h2)
    union_lines_crop = lines_mask[union_y:union_y_end, union_x:union_x_end]
    total_line_pixels = np.count_nonzero(union_lines_crop)
    if total_line_pixels == 0: return False
    box1_rel_x, box1_rel_y = x1 - union_x, y1 - union_y
    box1_crop = union_lines_crop[box1_rel_y:box1_rel_y + h1, box1_rel_x:box1_rel_x + w1]
    box1_line_pixels = np.count_nonzero(box1_crop)
    box2_rel_x, box2_rel_y = x2 - union_x, y2 - union_y
    box2_crop = union_lines_crop[box2_rel_y:box2_rel_y + h2, box2_rel_x:box2_rel_x + w2]
    box2_line_pixels = np.count_nonzero(box2_crop)
    tolerance = 2
    if total_line_pixels > box1_line_pixels + box2_line_pixels + tolerance: return True
    return False


def merge_boxes(boxes, lines_mask, proximity_thresh=20):
    """
    对边界框列表进行高级合并和过滤。
    """
    if not boxes: return []
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    filtered_boxes = []
    for i in range(len(boxes)):
        is_contained = False
        for j in range(len(filtered_boxes)):
            x_j, y_j, w_j, h_j = filtered_boxes[j]
            x_i, y_i, w_i, h_i = boxes[i]
            if (x_i >= x_j and y_i >= y_j and x_i + w_i <= x_j + w_j and y_i + h_i <= y_j + h_j):
                is_contained = True
                break
        if not is_contained: filtered_boxes.append(boxes[i])
    boxes = filtered_boxes
    merged_in_pass = True
    while merged_in_pass:
        merged_in_pass = False
        merged_boxes = []
        merged_flags = [False] * len(boxes)
        for i in range(len(boxes)):
            if merged_flags[i]: continue
            current_box = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if merged_flags[j]: continue
                other_box = boxes[j]
                expanded_box = [current_box[0] - proximity_thresh, current_box[1] - proximity_thresh,
                                current_box[2] + 2 * proximity_thresh, current_box[3] + 2 * proximity_thresh]
                intersect = not (other_box[0] > expanded_box[0] + expanded_box[2] or other_box[0] + other_box[2] <
                                 expanded_box[0] or other_box[1] > expanded_box[1] + expanded_box[3] or other_box[1] +
                                 other_box[3] < expanded_box[1])
                if intersect:
                    if not has_line_between(current_box, other_box, lines_mask):
                        x, y, w, h = min(current_box[0], other_box[0]), min(current_box[1], other_box[1]), max(
                            current_box[0] + current_box[2], other_box[0] + other_box[2]) - min(current_box[0],
                                                                                                other_box[0]), max(
                            current_box[1] + current_box[3], other_box[1] + other_box[3]) - min(current_box[1],
                                                                                                other_box[1])
                        current_box = [x, y, w, h]
                        merged_flags[j] = True
                        merged_in_pass = True
            merged_boxes.append(tuple(current_box))
            merged_flags[i] = True
        boxes = merged_boxes
    return boxes


class ImageSegment:
    def __init__(self, image_data, x_offset, y_offset):
        self.image_data = image_data
        self.x_offset = x_offset
        self.y_offset = y_offset


def segment_image(image_bytes, min_area=2000, min_width=20, min_height=20):
    """
    主函数：执行从图像中提取文本区域的完整流程，并增加中间步骤的可视化。
    """
    # --- 1. 图像预处理 ---
    image_array = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- 2. 检测并移除线条 ---
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(thresh.shape[1] / 50), 1))
    detected_horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(thresh.shape[0] / 50)))
    detected_vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    lines_mask = detected_horizontal_lines + detected_vertical_lines
    kernel_dilate_lines = np.ones((3, 3), np.uint8)
    dilated_lines_mask = cv2.dilate(lines_mask, kernel_dilate_lines, iterations=1)
    no_lines_image = cv2.subtract(thresh, dilated_lines_mask)

    # --- 3. 合并文本并找到候选框 ---
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_text = cv2.dilate(no_lines_image, dilate_kernel, iterations=4)
    contours, _ = cv2.findContours(dilated_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    initial_boxes = []
    print(f"找到 {len(contours)} 个初始轮廓。")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > min_area and w > min_width and h > min_height:
            initial_boxes.append((x, y, w, h))
    print(f"经过宽度/高度/面积过滤后，剩余 {len(initial_boxes)} 个候选框。")

    # --- 【可视化 1: 聚合前】 ---
    image_before_merge = original_image.copy()
    for (x, y, w, h) in initial_boxes:
        cv2.rectangle(image_before_merge, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色

    # 执行高级合并
    merged_boxes = merge_boxes(initial_boxes, lines_mask, proximity_thresh=50)
    print(f"合并后剩余 {len(merged_boxes)} 个区域。")

    # --- 【可视化 2: 聚合后】 ---
    image_after_merge = original_image.copy()
    for (x, y, w, h) in merged_boxes:
        cv2.rectangle(image_after_merge, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色

    # --- 4. 后处理过滤器 ---
    truly_final_boxes = []
    LINE_OVERLAP_THRESHOLD = 0.8
    for box in merged_boxes:
        x, y, w, h = box
        if w < min_width or h < min_height: continue

        content_crop = thresh[y:y + h, x:x + w]
        content_pixels = np.count_nonzero(content_crop)
        if content_pixels == 0: continue

        line_crop = lines_mask[y:y + h, x:x + w]
        line_pixels = np.count_nonzero(line_crop)
        overlap_ratio = line_pixels / content_pixels
        if overlap_ratio >= LINE_OVERLAP_THRESHOLD:
            continue

        # 裁剪图像并转换为字节
        y_start, y_end = max(0, y), min(original_image.shape[0], y + h)
        x_start, x_end = max(0, x), min(original_image.shape[1], x + w)
        cropped_region = original_image[y_start:y_end, x_start:x_end]
        if cropped_region.size == 0:
            continue

        success, encoded_image_buffer = cv2.imencode('.jpg', cropped_region)
        if success:
            encoded_image_bytes = encoded_image_buffer.tobytes()
            # 创建 ImageSegment 对象并添加到最终列表
            truly_final_boxes.append(ImageSegment(image_data=encoded_image_bytes, x_offset=x_start, y_offset=y_start))
        else:
            print(f"警告：在偏移量 ({x_start}, {y_start}) 处编码图像区域失败。")

    return truly_final_boxes


def adjust_box_heights(ocr_result, height_tolerance_ratio=0.2):
    """
    调整OCR结果中边界框的高度，使相似大小的文本框高度一致。

    :param ocr_result: 原始的PaddleOCR识别结果。
    :param height_tolerance_ratio: 高度容忍度比例。用于判断哪些框属于同一组。
                                   例如，0.2表示高度差异在20%以内的框可以被分到一组。
    :return: 调整了高度后的新OCR结果。
    """
    if not ocr_result or not ocr_result[0]:
        return ocr_result

    lines = ocr_result[0]

    # 1. 计算每个框的初始高度和中心Y坐标
    boxes_with_info = []
    for i, line in enumerate(lines):
        box = line[0]
        # 计算近似高度 (取四个点的y坐标最大值和最小值的差)
        ys = [p[1] for p in box]
        height = max(ys) - min(ys)
        # 计算垂直中心
        center_y = np.mean(ys)
        boxes_with_info.append({'original_index': i, 'height': height, 'center_y': center_y})

    # 2. 按高度对框进行排序，方便分组
    boxes_with_info.sort(key=lambda x: x['height'])

    # 3. 根据高度相似性进行分组
    groups = []
    if not boxes_with_info:
        return ocr_result

    current_group = [boxes_with_info[0]]
    for i in range(1, len(boxes_with_info)):
        # 如果当前框的高度与组内平均高度相比，差异在容忍度范围内，则加入该组
        # 使用组内第一个元素的高度作为基准可以简化逻辑
        base_height = current_group[0]['height']
        if abs(boxes_with_info[i]['height'] - base_height) < base_height * height_tolerance_ratio:
            current_group.append(boxes_with_info[i])
        else:
            # 否则，当前组结束，开始一个新组
            groups.append(current_group)
            current_group = [boxes_with_info[i]]
    groups.append(current_group)  # 添加最后一组

    # 4. 计算每组的平均高度，并调整该组内所有框的坐标
    new_lines = [None] * len(lines)
    print("\n--- 开始修正边界框高度 ---")
    for group in groups:
        group_heights = [b['height'] for b in group]
        # 使用中位数高度更稳健，可以避免极端值的影响
        avg_height = np.median(group_heights)
        print(
            f"找到一组 {len(group)} 个框, 原始高度范围: [{min(group_heights):.2f} - {max(group_heights):.2f}], 修正为统一高度: {avg_height:.2f}")

        for box_info in group:
            original_index = box_info['original_index']
            original_line = lines[original_index]
            original_box = original_line[0]

            # 计算垂直中心 (使用原始框的中心)
            center_y = np.mean([p[1] for p in original_box])

            # 新的顶部和底部y坐标
            new_top_y = center_y - avg_height / 2
            new_bottom_y = center_y + avg_height / 2

            # 创建新的坐标点，x坐标保持不变，y坐标更新
            # box_points: [top_left, top_right, bottom_right, bottom_left]
            new_box = [
                [original_box[0][0], new_top_y],  # Top-left
                [original_box[1][0], new_top_y],  # Top-right
                [original_box[2][0], new_bottom_y],  # Bottom-right
                [original_box[3][0], new_bottom_y]  # Bottom-left
            ]

            # 用新的box替换旧的box，并保持文本和置信度不变
            new_lines[original_index] = [new_box, original_line[1]]

    print("-----------------------------\n")
    return [new_lines]
