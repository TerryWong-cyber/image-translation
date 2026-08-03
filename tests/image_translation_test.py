import base64
import sys
import time
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import concurrent.futures

current_file_path = os.path.abspath(__file__)
pic_translate_dir = os.path.dirname(current_file_path)
ocr_dir = os.path.dirname(pic_translate_dir)
project_root = os.path.dirname(ocr_dir)

# 将项目根目录添加到 Python 搜索路径的开头
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.image_translation.config import get_settings
from src.image_translation.image_translation import translate_image
from clients.call_llm_recognition import call_llm_recognition_api_batch


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


def det_to_ocr_result(image_path, det_result):
    st = time.time()
    if not os.path.exists(image_path):
        print(f"错误: 图片文件未找到 at {image_path}")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法使用OpenCV读取图片: {image_path}")
        return []

    if not det_result or not det_result[0]:
        print("警告: 检测结果为空，无法进行裁切。")
        return []

    boxes = det_result[0]
    base64_images = []  # 用于存储所有裁切图的base64编码

    print(f"--- 正在从 '{os.path.basename(image_path)}' 中裁切 {len(boxes)} 个区域 ---")

    # 1. 先循环准备好所有图片数据，不要在循环里发请求
    for i, box in enumerate(boxes):
        src_pts = np.array(box, dtype="float32")

        # (透视变换代码与原来相同，此处省略...)
        width_top = np.sqrt(((src_pts[0][0] - src_pts[1][0]) ** 2) + ((src_pts[0][1] - src_pts[1][1]) ** 2))
        width_bottom = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
        max_width = int(max(width_top, width_bottom))

        height_left = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
        height_right = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
        max_height = int(max(height_left, height_right))

        if max_width == 0 or max_height == 0:
            print(f"警告: 第 {i + 1} 个裁切区域尺寸为0，已跳过。")
            continue

        dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                           dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_image = cv2.warpPerspective(image, M, (max_width, max_height))

        _, buffer = cv2.imencode('.png', warped_image)
        if not _:
            print(f"警告: 无法编码第 {i + 1} 个裁切区域。")
            continue

        base64_string = base64.b64encode(buffer).decode('utf-8')
        base64_images.append(base64_string)

    if not base64_images:
        print("没有可识别的图像区域。")
        return []

    # 2. 一次性发送所有图片进行识别
    print(f"--- 准备发送 {len(base64_images)} 个图像进行批量识别 ---")
    rec_results = call_llm_recognition_api_batch(base64_images)

    if not rec_results or len(rec_results) != len(boxes):
        print("错误: 批量识别返回结果数量与请求不匹配或返回为空。")
        return []

    # 3. 将识别结果与原始box对应起来
    ocr_result = []
    for box, rec_text in zip(boxes, rec_results):
        ocr_result.append([box, (rec_text, 0.99)])  # 假设置信度为0.99

    print(f"提取文字总耗时: {time.time() - st:.4f}s")
    return ocr_result


def batch_translate_images(input_folder, output_folder, ocr_engine):
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
        st = time.time()
        print(f"\n================== [ {i + 1}/{len(image_files)} ] 正在处理: {filename} ==================")

        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        # --- 1. OCR识别 ---
        # 注意：为了让 draw_ocr_boxes 能工作，rec必须为True，因为它需要文本和置信度
        print(f"正在对图片进行OCR识别: {input_path}")

        result = ocr_engine.ocr(input_path, cls=True, rec=False)
        print("OCR识别完成。")
        print(f"OCR耗时: {time.time() - st:.4f}s")

        # --- 3. 绘制并保存结果 ---
        # 构建输出文件路径，添加后缀以区分
        base_name, extension = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{base_name}_translated{extension}")

        draw_det_boxes(input_path, output_path, result)

        ocr_res = det_to_ocr_result(input_path, result)

        image_obj = cv2.imread(input_path)
        painted_image, agg_translation_result = translate_image(image_obj, [ocr_res], "any_zh")

        cv2.imwrite(output_path, painted_image)
        print("\n------------------------------------")
        print(f"成功！带有边界框的图片已保存到: {output_path}")
        print("agg_translation_result", agg_translation_result)
        print(f"图片翻译总耗时: {time.time() - st:.4f}s")

    print("\n================== 所有图片处理完毕！ ==================")


# --- 主程序入口 ---
if __name__ == "__main__":
    start_time = time.time()
    settings = get_settings()
    input_folder = settings.paths.test_input_dir
    output_folder = settings.paths.test_translation_output_dir
    ocr = PaddleOCR(
        lang=settings.ocr.language,
        det=True,
        rec=False,
        det_db_unclip_ratio=settings.ocr.unclip_ratio,
    )
    print("模型初始化完成。")

    batch_translate_images(str(input_folder), str(output_folder), ocr_engine=ocr)
    print(f"批量图片翻译总耗时: {time.time() - start_time:.4f}s")
