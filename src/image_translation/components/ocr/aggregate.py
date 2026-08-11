import cv2
import os
import time
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


def calculate_box_height(box):
    points = np.array(box)
    height_left = np.linalg.norm(points[0] - points[3])
    height_right = np.linalg.norm(points[1] - points[2])
    return (height_left + height_right) / 2


def group_boxes_by_height(boxes, height_tolerance_px=5):
    if not boxes:
        return []

    # 1. 计算所有box的高度
    heights = [calculate_box_height(box) for box in boxes]

    # DBSCAN需要一个 (n_samples, n_features) 格式的数组，所以我们reshape
    # 我们的数据是一维的（只有高度），所以是 (n_samples, 1)
    heights_reshaped = np.array(heights).reshape(-1, 1)

    # 2. 使用DBSCAN进行聚类
    # eps: 两个样本被认为是邻居的最大距离。这是关键参数。
    # min_samples: 一个点被视为核心点的邻域中的样本数。设为1意味着即使单个box也能形成一个簇。
    db = DBSCAN(eps=height_tolerance_px, min_samples=1).fit(heights_reshaped)

    # 获取每个box的簇标签
    labels = db.labels_

    # 3. 根据标签对原始boxes进行分组
    grouped_boxes = defaultdict(list)
    for i, box in enumerate(boxes):
        label = labels[i]
        grouped_boxes[label].append(box)

    # 4. 将分组结果从字典转换为列表
    # 输出格式为 [[group1], [group2], ...]
    return list(grouped_boxes.values())


# def group_boxes_by_height(boxes, min_height, max_height, gap):
#     """
#     按最小行高的三分之一为阈值，将框分为大组。
#     """
#     all_groups_num = int((max_height - min_height) // gap + 1)  # 多一组，即使最后一组为空也无所谓
#     init_groups = [[] for _ in range(all_groups_num)]
#
#     for box in boxes:
#         height = box[3][1] - box[0][1]
#         groups_index = int((height - min_height) // gap)
#         init_groups[groups_index].append(box)
#
#     return init_groups


def merge_boxes(box1, box2):
    return [
        [min(box1[0][0], box2[0][0]), min(box1[0][1], box2[0][1])],
        [max(box1[1][0], box2[1][0]), min(box1[1][1], box2[1][1])],
        [max(box1[2][0], box2[2][0]), max(box1[2][1], box2[2][1])],
        [min(box1[3][0], box2[3][0]), max(box1[3][1], box2[3][1])]
    ]


def merge_boxes_v2(box1, box2):
    return [
        [min(box1[0][0], box2[0][0]), min(box1[0][1], box2[0][1])],
        [max(box1[1][0], box2[1][0]), min(box1[1][1], box2[1][1])],
        [max(box1[2][0], box2[2][0]), max(box1[2][1], box2[2][1])],
        [min(box1[3][0], box2[3][0]), max(box1[3][1], box2[3][1])]
    ]


def aggregate_within_group(group_boxes, img, min_height, cur_group_height,
                           height_aggregate_ratio=0.5, same_line_range_ratio=0.3):
    time_start = time.time()

    if not group_boxes:
        return []

    # 按纵向中心点排序并分组为行
    # 以前一个区域为标杆，中心距离超过0.3倍行高视为下一行
    centers_y = [(box[0][1] + box[3][1]) / 2 for box in group_boxes]
    sorted_indices = np.argsort(centers_y)
    group_boxes = [group_boxes[i] for i in sorted_indices]
    centers_y = [centers_y[i] for i in sorted_indices]
    #
    # print("step min max use", time.time() - time_start)

    rows = []
    current_row = [group_boxes[0]]
    for i in range(1, len(group_boxes)):
        if abs(centers_y[i] - (
                current_row[0][0][1] + current_row[0][3][1]) / 2) < cur_group_height * same_line_range_ratio:
            current_row.append(group_boxes[i])
            # current_row.append({"box":group_boxes[i], "original_boxs":[]})
        else:
            rows.append(current_row)
            current_row = [group_boxes[i]]
            # current_row = [{"box": group_boxes[i], "merged_box": []}]
    if current_row:
        rows.append(current_row)

    # print("step sort use", time.time() - time_start)

    # 1. 横向同行聚合，当同一行中，两个区域间距超过2倍行高（字符大小），可以认为不是一个连续的语义
    aggregated_rows = []
    # print("len(rows)", len(rows))
    for row in rows:
        if not row:
            continue
        # 按x坐标排序
        centers_x = [box[0][0] for box in row]
        sorted_row = sorted(range(len(row)), key=lambda i: centers_x[i])
        row = [row[i] for i in sorted_row]

        # merged_row = [row[0]]
        merged_row = [{"box": row[0], "original_boxs": [row[0]]}]
        for i in range(1, len(row)):
            next_box = row[i]

            if can_horizontal_aggregate(img, merged_row[-1]["box"], next_box, cur_group_height):
                # merged_row[-1] = merge_boxes(merged_row[-1], next_box)
                merged_box = merge_boxes(merged_row[-1]["box"], next_box)
                merged_row[-1]["original_boxs"].append(next_box)
                merged_row[-1]["box"] = merged_box
            else:
                # merged_row.append(next_box)
                merged_row.append({"box": next_box, "original_boxs": [next_box]})

        aggregated_rows.append(merged_row)

    # print("step horizontal_aggregate use", time.time() - time_start)

    if not aggregated_rows:
        return []

    # 2. 纵向隔行聚合
    # 每行逐个去匹配已有的区域，存在性能浪费，但代码清晰 todo
    res_boxes = aggregated_rows[0]

    for i in range(1, len(aggregated_rows)):
        next_row = aggregated_rows[i]
        for next_box in next_row:
            next_box_aggregated = False
            for aggregated_box_index in range(len(res_boxes)):
                aggregated_box = res_boxes[aggregated_box_index]
                if can_vertical_aggregate(aggregated_box["box"], next_box["box"], cur_group_height, img):
                    # res_boxes[aggregated_box_index] = merge_boxes(aggregated_box, next_box)
                    res_boxes[aggregated_box_index]["box"] = merge_boxes(aggregated_box["box"], next_box["box"])
                    res_boxes[aggregated_box_index]["original_boxs"].extend(next_box["original_boxs"])
                    next_box_aggregated = True
                    break
            # next_box 要么被之前box融合，要么添加
            if not next_box_aggregated:
                res_boxes.append(next_box)
    # print("step vertical_aggregate use", time.time() - time_start)
    return res_boxes


def can_horizontal_aggregate(img, left_box, right_box, line_height):
    return False
    # distance = right_box[0][0] - left_box[1][0]  # 同行下一个区域距离聚合区域最右侧距离
    # if distance > 0.1 * line_height:
    #     return False
    #
    # return True

    # # 两个box之间不能有竖线
    # x_start = left_box[0][0]
    # x_end = right_box[2][0]
    # y_start = left_box[0][1]
    # y_end = right_box[2][1]
    # has_line, lines = detect_oriented_lines_hough(img, x_start, x_end, y_start, y_end, orientation="vertical",
    #                                               max_line_gap_pixels=3)
    # if has_line:
    #     return False
    #
    # return True


def can_vertical_aggregate(upper_box, under_box, line_height, img):
    upper_box_center_x = (upper_box[2][0] + upper_box[0][0]) / 2
    upper_box_center_y = (upper_box[2][1] + upper_box[0][1]) / 2

    under_box_center_x = (under_box[2][0] + under_box[0][0]) / 2
    under_box_center_y = (under_box[2][1] + under_box[0][1]) / 2

    x_start = upper_box[0][0]  # up box 左上角x
    x_end = under_box[1][0]  # under box 右上角x

    # 如果上行长度过小或下行长度过小，不认为是一个语义被迫中断语句
    up_box_length = upper_box[1][0] - upper_box[0][0]
    under_box_length = under_box[1][0] - under_box[0][0]
    if up_box_length < 6 * line_height or under_box_length < 6 * line_height:
        return False

    # 如果下行起点超过上行起点较大距离，不认为是一个语义被迫中断语句
    if under_box[0][0] - upper_box[0][0] > 4 * line_height:
        return False

    # 如果两个box空白y距离大于行高，则认为不应该聚合
    center_y_gap = under_box[0][1] - upper_box[3][1]
    if center_y_gap > line_height:
        return False

    # 如果下行box x 在上行x左侧 超过2倍行高，则认为远超缩进
    start_x_gap = -(under_box[0][0] - upper_box[3][0])
    if start_x_gap > line_height * 2:
        return False

    # 如果下行中心距在上行中心右侧，且超过30%比例，则不认为是一个连续的语义聚合段落
    center_x_gap = under_box_center_x - upper_box_center_x
    if center_x_gap > up_box_length * 0.3:
        return False

    # # 两个box之间不能有横线
    # has_line, lines = detect_oriented_lines_hough(img, x_start, x_end, upper_box_center_y,
    #                                               under_box_center_y, max_line_gap_pixels=3)
    # if has_line:
    #     return False

    return True


def visualize_boxes_poly(image_path, boxes, output_path=None, color=(0, 0, 255), thickness=2):
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"错误：无法从路径 {image_path} 读取图片。")
        return None

    for box_points in boxes:
        pts = np.array(box_points, dtype=np.int32)

        if pts.ndim == 1 and len(pts) % 2 == 0:
            pts = pts.reshape((-1, 2))
        elif pts.ndim != 2 or pts.shape[1] != 2:
            print(f"警告：跳过无效的点集格式: {box_points}")
            continue

        # 3. 绘制多边形
        # pts 本身代表一个多边形，所以我们把它放到一个列表中: [pts]
        # isClosed=True 表示连接最后一个点和第一个点，形成闭合多边形。
        cv2.polylines(img_color, [pts], isClosed=True, color=color, thickness=thickness)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_aggregated_poly.jpg"

    try:
        success = cv2.imwrite(output_path, img_color)
        if success:
            print(f"标注后的图片已保存到: {output_path}")
        else:
            print(f"错误：无法将图片写入到: {output_path}")
    except Exception as e:
        print(f"保存图片时发生错误: {e}")
        # 即使保存失败，仍然返回标注后的图片

    return img_color


# 处理ocr 结果出现像素级差异的排序， 先按照粗粒度聚合行，再进行纵横排序
def _sort_aggregated_boxes(aggregated_obj_list, return_row=False):
    start_y_list = [box['box'][0][1] for box in aggregated_obj_list]
    sorted_indices = np.argsort(start_y_list)
    sorted_boxes = [aggregated_obj_list[i] for i in sorted_indices]
    sorted_y_list = [start_y_list[i] for i in sorted_indices]
    #
    # print("step min max use", time.time() - time_start)

    rows = []
    current_row = [sorted_boxes[0]]
    for i in range(1, len(sorted_boxes)):
        # cur_gap_height = (current_row[0][3][1] - current_row[0][2][1]) / 2 * 0.3
        cur_gap_height = 20
        if abs(sorted_boxes[i]['box'][0][1] - current_row[0]['box'][0][1]) < cur_gap_height:
            current_row.append(sorted_boxes[i])
            # current_row.append({"box":group_boxes[i], "original_boxs":[]})
        else:
            rows.append(current_row)
            current_row = [sorted_boxes[i]]
            # current_row = [{"box": group_boxes[i], "merged_box": []}]
    if current_row:
        rows.append(current_row)

    res = []
    for i in range(len(rows)):
        row = rows[i]
        start_x_list = [obj['box'][0][0] for obj in row]
        sorted_indices = np.argsort(start_x_list)
        temp = [row[sort_i] for sort_i in sorted_indices]
        if return_row:
            res.append(temp)
        else:
            res.extend(temp)

    return res


def aggregate_ocr_results(img, image_path, ocr_results, output_path=None, return_row=False):
    boxes = [res[0] for res in ocr_results[0]]
    heights = [box[3][1] - box[0][1] for box in boxes]  # y 左下减左上 为height
    min_height = min(heights)
    max_height = max(heights)
    gap = (max_height + min_height) // 6
    # 分组
    groups = group_boxes_by_height(boxes)
    if output_path is not None:
        for group_i in range(len(groups)):
            visualize_boxes_poly(image_path, groups[group_i], f"{group_i}_group.png")

    # 在每个大组内进行聚合
    aggregated_boxes = []
    for group_index in range(len(groups)):
        group = groups[group_index]
        if len(group) == 0:
            continue
        cur_group_height = gap * group_index + min_height
        group_aggregated = aggregate_within_group(group, img, min_height, cur_group_height)
        aggregated_boxes.extend(group_aggregated)

    output_img = None
    if image_path:
        output_img = visualize_boxes_poly(image_path, [aggregated_box["box"] for aggregated_box in aggregated_boxes],
                                          output_path)

    aggregated_boxes = _sort_aggregated_boxes(aggregated_boxes, return_row=return_row)

    return aggregated_boxes, output_img


# 示例用法
if __name__ == "__main__":
    import paddle
    from paddleocr import PaddleOCR

    from image_translation.components.ocr.device import paddleocr_init_kwargs
    from image_translation.components.ocr.paddleocr3_adapter import paddleocr_results_to_legacy
    from image_translation.config import get_settings

    ocr_settings = get_settings().ocr
    ocr = PaddleOCR(**paddleocr_init_kwargs(ocr_settings, paddle))

    # for i in range(2, 9):
    for i in range(1, 2):
        image_path = f"tt{i}.png"
        ocr_results = paddleocr_results_to_legacy(ocr.predict(image_path))
        time_start = time.time()
        img = cv2.imread(image_path, 0)

        print("len(ocr_results):", len(ocr_results[0]))

        aggregated_boxes, output_img = aggregate_ocr_results(img, image_path, ocr_results)
        print("聚合后的框数量：", len(aggregated_boxes))
        # print("聚合后的框：", aggregated_boxes)
        cv2.imwrite(f"ss_output/output_aggregated_{i}.jpg", output_img)
        print("use time", time.time() - time_start)
