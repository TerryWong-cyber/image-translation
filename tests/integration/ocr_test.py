import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

from image_translation.config import get_settings


# 这个函数保持不变
def adjust_box_heights(ocr_result, height_tolerance_ratio=0.2):
    """
    调整OCR结果中边界框的高度，使相似大小的文本框高度一致。
    """
    if not ocr_result or not ocr_result[0]:
        return ocr_result

    lines = ocr_result[0]

    boxes_with_info = []
    for i, line in enumerate(lines):
        box = line[0]
        ys = [p[1] for p in box]
        height = max(ys) - min(ys)
        center_y = np.mean(ys)
        boxes_with_info.append({'original_index': i, 'height': height, 'center_y': center_y})

    boxes_with_info.sort(key=lambda x: x['height'])

    groups = []
    if not boxes_with_info:
        return ocr_result

    current_group = [boxes_with_info[0]]
    for i in range(1, len(boxes_with_info)):
        base_height = current_group[0]['height']
        if abs(boxes_with_info[i]['height'] - base_height) < base_height * height_tolerance_ratio:
            current_group.append(boxes_with_info[i])
        else:
            groups.append(current_group)
            current_group = [boxes_with_info[i]]
    groups.append(current_group)

    new_lines = [None] * len(lines)
    print("\n--- 开始修正边界框高度 ---")
    for group in groups:
        group_heights = [b['height'] for b in group]
        avg_height = np.median(group_heights)
        print(
            f"找到一组 {len(group)} 个框, 原始高度范围: [{min(group_heights):.2f} - {max(group_heights):.2f}], 修正为统一高度: {avg_height:.2f}")

        for box_info in group:
            original_index = box_info['original_index']
            original_line = lines[original_index]
            original_box = original_line[0]
            center_y = np.mean([p[1] for p in original_box])
            new_top_y = center_y - avg_height / 2
            new_bottom_y = center_y + avg_height / 2
            new_box = [
                [original_box[0][0], new_top_y],
                [original_box[1][0], new_top_y],
                [original_box[2][0], new_bottom_y],
                [original_box[3][0], new_bottom_y]
            ]
            new_lines[original_index] = [new_box, original_line[1]]

    print("-----------------------------\n")
    return [new_lines]


# 这个函数保持不变
def draw_ocr_boxes(image_path, output_path, ocr_result):
    """
    在图片上绘制OCR识别的边界框，并打印识别出的文本。
    """
    if not os.path.exists(image_path):
        print(f"错误: 图片文件未找到 at {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
    except Exception as e:
        print(f"错误: 无法打开或处理图片 {image_path}. 错误信息: {e}")
        return

    print("\n--- 开始绘制边界框并打印对应文本 ---")
    if ocr_result and ocr_result[0]:
        lines = ocr_result[0]

        for i, line in enumerate(lines):
            if line is None: continue
            box = line[0]
            text_info = line[1]
            text = text_info[0]
            confidence = text_info[1]

            print(f"框 {i + 1}: 文字 = '{text}', 置信度 = {confidence:.4f}")

            int_box = [(int(p[0]), int(p[1])) for p in box]
            draw.polygon(int_box, outline='lime', width=2)

    image.save(output_path)
    print("\n------------------------------------")
    print(f"成功！带有边界框的图片已保存到: {output_path}")


# 这个只检测不识别时，绘制函数
def draw_det_boxes(image_path, output_path, ocr_result):
    """
    在图片上绘制OCR识别的边界框，并打印识别出的文本。
    """
    if not os.path.exists(image_path):
        print(f"错误: 图片文件未找到 at {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
    except Exception as e:
        print(f"错误: 无法打开或处理图片 {image_path}. 错误信息: {e}")
        return

    print("\n--- 开始绘制边界框并打印对应文本 ---")
    if ocr_result and ocr_result[0]:
        boxes = ocr_result[0]

        # 1. 先绘制所有框
        for i, box in enumerate(boxes):
            if box is None: continue
            int_box = [tuple(map(int, p)) for p in box]  # 更简洁的转换方式
            draw.polygon(int_box, outline='lime', width=2)

    image.save(output_path)
    print("\n------------------------------------")
    print(f"成功！带有边界框的图片已保存到: {output_path}")


# --- 新增的主函数，用于批量处理 ---
def batch_process_images(input_folder, output_folder, ocr_engine):
    """
    批量处理指定文件夹中的所有图片。
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
        return

    # 检查并创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    print(f"结果将保存到: {output_folder}")

    # 支持的图片格式
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"在文件夹 '{input_folder}' 中没有找到支持的图片文件。")
        return

    print(f"共找到 {len(image_files)} 张图片，开始处理...")

    # 循环处理每一张图片
    for i, filename in enumerate(image_files):
        print(f"\n================== [ {i + 1}/{len(image_files)} ] 正在处理: {filename} ==================")

        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        # --- 1. OCR识别 ---
        # 注意：为了让 draw_ocr_boxes 能工作，rec必须为True，因为它需要文本和置信度
        print(f"正在对图片进行OCR识别: {input_path}")

        rec = False
        result = ocr_engine.ocr(input_path, cls=True, rec=rec)
        print("OCR识别完成。")

        if rec:
            # --- 2. 调整边界框高度 ---
            # 你可以调整 height_tolerance_ratio 来控制分组的宽松程度
            adjusted_result = adjust_box_heights(result, height_tolerance_ratio=0.5)

            # --- 3. 绘制并保存结果 ---
            # 构建输出文件路径，添加后缀以区分
            base_name, extension = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{base_name}_adjusted{extension}")

            draw_ocr_boxes(input_path, output_path, adjusted_result)
        else:
            # --- 3. 绘制并保存结果 ---
            # 构建输出文件路径，添加后缀以区分
            base_name, extension = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{base_name}_det{extension}")

            draw_det_boxes(input_path, output_path, result)

    print("\n================== 所有图片处理完毕！ ==================")


if __name__ == "__main__":
    settings = get_settings()

    print("正在初始化 PaddleOCR 模型 (这可能需要一些时间)...")
    # 初始化OCR引擎，开启方向分类、检测和识别
    # ocr = PaddleOCR(lang='ch',  det=True, rec=True, det_db_unclip_ratio=1.5)
    ocr = PaddleOCR(
        lang=settings.ocr.language,
        det=True,
        rec=False,
        det_db_unclip_ratio=settings.ocr.unclip_ratio,
    )

    print("模型初始化完成。")

    # 调用批量处理函数
    batch_process_images(
        str(settings.paths.test_input_dir),
        str(settings.paths.test_output_dir),
        ocr,
    )
