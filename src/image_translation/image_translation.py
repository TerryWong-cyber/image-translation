import time
import cv2
import numpy as np

from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

from ocr.aggregate import aggregate_ocr_results
from src.image_translation.components.font.font_process import cal_box_font_size, get_font
from src.image_translation.components.text_box.box_process import (
    process_ocr_res,
    process_and_filter_aggregated_boxes,
)
from clients.call_llm_translation import llm_translate


# ==============================================================================
# 定义上下文类来管理所有状态
# ==============================================================================
class ImageTranslationContext:
    """
    一个用于封装单次图片翻译任务所有状态的上下文对象。
    """

    def __init__(self, original_image):
        self.original_image = original_image
        self.ocr_box_ids = []
        self.ocr_box_map_text = {}
        self.ocr_box_map_coordinate = {}
        self.ocr_box_map_width_height = {}
        self.ocr_box_map_angle_rad = {}
        self.ocr_box_map_translated_text = {}
        self.ocr_box_font_size_map = {}
        self.agg_box_map_coordinate = {}
        self.agg_box_map_origin = {}
        self.agg_box_map_text_to_translate = {}
        self.agg_box_map_text_to_keep = {}


# 添加模糊蒙层
def apply_blurred_overlay(img, box_ids, colors, context, blur_ksize=(7, 7), blur_sigma=1.0):
    if len(box_ids) != len(colors):
        raise ValueError("The number of boxes must match the number of colors")

    colors = [tuple(int(c) for c in color) if isinstance(color, tuple) and len(color) == 3 else (255, 255, 255)
              for color in colors]

    overlay = img.copy()
    box_nps = []
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    expansion_margin = 5  # 设置蒙层区域往外膨胀5个像素
    kernel = np.ones((expansion_margin, expansion_margin), np.uint8)

    for box_id, color in zip(box_ids, colors):
        # 从上下文中获取坐标数据
        box_coord = context.ocr_box_map_coordinate.get(box_id)
        if box_coord is not None:
            box_np = np.array(box_coord, dtype=np.int32)
            box_nps.append(box_np)
            bgr_color = (color[2], color[1], color[0])
            if expansion_margin > 0:
                # 1. 为当前框创建一个临时蒙版
                temp_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(temp_mask, [box_np], 255)
                # 2. 膨胀这个临时蒙版
                dilated_mask = cv2.dilate(temp_mask, kernel, iterations=1)
                # 3. 使用膨胀后的蒙版在 overlay 上填充颜色
                overlay[dilated_mask == 255] = bgr_color
                # 4. 将膨胀后的蒙版合并到 combined_mask 中
                cv2.bitwise_or(combined_mask, dilated_mask, combined_mask)
            else:
                # 如果不扩张，则使用原始逻辑
                cv2.fillPoly(overlay, [box_np], bgr_color)
                cv2.fillPoly(combined_mask, [box_np], 255)  # 同样需要更新 combined_mask

    if not box_nps:
        return img  # 如果没有有效的框，直接返回原图

    x, y, w, h = cv2.boundingRect(np.concatenate(box_nps))
    margin = max(blur_ksize) // 2
    x, y, w, h = max(0, x - margin), max(0, y - margin), w + 2 * margin, h + 2 * margin
    sub_mask = combined_mask[y:y + h, x:x + w]
    sub_mask = cv2.GaussianBlur(sub_mask, blur_ksize, blur_sigma)
    combined_mask[y:y + h, x:x + w] = sub_mask

    alpha = combined_mask[..., np.newaxis] / 255.0
    result = (1 - alpha) * img + alpha * overlay
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_translated_text(img, box_ids, text_colors, context, bold=False):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    for o_box_id, color in zip(box_ids, text_colors):
        # 从上下文中获取所需数据
        box = context.ocr_box_map_coordinate.get(o_box_id)
        translated_text = context.ocr_box_map_translated_text.get(o_box_id)

        if not box or not translated_text or len(translated_text.strip()) == 0:
            print(f"Skipping box {o_box_id}: missing coordinate or translated text.")
            continue

        font_size = context.ocr_box_font_size_map.get(o_box_id, 30)
        font = get_font(font_size)
        text_bbox = font.getbbox(translated_text)
        print(f'add_translated_text:{translated_text},font_size:{font_size}')

        box_width, box_height = context.ocr_box_map_width_height.get(o_box_id, (0, 0))
        # print(f"box_id: {o_box_id}, box_coord: {box}")
        # print(f"box_width: {box_width}, box_height: {box_height}")

        padding = 0
        img_height = box_height + padding
        img_width = box_width + padding
        text_img = Image.new('RGBA', (int(img_width), int(img_height)), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)

        fill_color = color + (255,)
        if bold:
            text_draw.text((-text_bbox[0], -text_bbox[1]), translated_text, fill=fill_color, font=font, stroke_width=1)
        else:
            text_draw.text((-text_bbox[0], -text_bbox[1]), translated_text, fill=fill_color, font=font)

        # text_img.save(f"piece/{o_box_id}.png") # Debug line

        src_points = np.array([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]], dtype=np.float32)
        dst_rec = np.array(box).astype(np.float32)

        angle_rad = context.ocr_box_map_angle_rad.get(o_box_id, 0)
        if 0 < np.abs(angle_rad) < 1:  # 这里小于1代表弧度，角度为1x180/π
            dst_rec = np.array([box[0], [box[1][0], box[0][1]], [box[1][0], box[3][1]], [box[0][0], box[3][1]]],
                               dtype=np.float32)

        min_x, min_y = np.min(dst_rec, axis=0).astype(int)
        max_x, max_y = np.max(dst_rec, axis=0).astype(int)
        warped_width = int(np.ceil(max_x - min_x))
        warped_height = int(np.ceil(max_y - min_y))

        if warped_width <= 0 or warped_height <= 0:
            continue

        dst_points_local = dst_rec - np.array(dst_rec[0], dtype=np.float32)
        local_matrix = cv2.getPerspectiveTransform(src_points, dst_points_local)

        warped_text_small_cv = cv2.warpPerspective(np.array(text_img), local_matrix, (warped_width, warped_height),
                                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=(0, 0, 0, 0))
        warped_text_small_pil = Image.fromarray(warped_text_small_cv)
        img_pil.paste(warped_text_small_pil, (min_x, min_y), warped_text_small_pil)

    result_img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
    return result_img_cv


def _process_single_box(box_id, img, mask, context, bg_padding=5, text_padding=2):
    k = 2
    attempts = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    DEFAULT_BG_COLOR_RGB = (255, 255, 255)
    DEFAULT_TEXT_COLOR_RGB = (0, 0, 0)

    # 从上下文中获取坐标数据
    box_coord = context.ocr_box_map_coordinate.get(box_id)
    if not box_coord:
        print(f"Warning: Box ID {box_id} not found in coordinate map.")
        return DEFAULT_BG_COLOR_RGB, DEFAULT_TEXT_COLOR_RGB

    x_coords, y_coords = zip(*box_coord)
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    bg_x_min = max(0, x_min - bg_padding)
    bg_y_min = max(0, y_min - bg_padding)
    bg_x_max = min(img.shape[1], x_max + bg_padding)
    bg_y_max = min(img.shape[0], y_max + bg_padding)

    bg_mask_region = mask[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
    bg_img_region = img[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
    bg_pixels = bg_img_region[bg_mask_region == 0]

    if len(bg_pixels) == 0:
        print(f"Warning: No background pixels found for box {box_id}. Using default colors.")
        return DEFAULT_BG_COLOR_RGB, DEFAULT_TEXT_COLOR_RGB

    unique_colors, counts = np.unique(bg_pixels, axis=0, return_counts=True)
    dominant_color_bgr = unique_colors[counts.argmax()]
    bg_color_rgb = (int(dominant_color_bgr[2]), int(dominant_color_bgr[1]), int(dominant_color_bgr[0]))

    text_x_min = max(0, x_min + text_padding)
    text_y_min = max(0, y_min + text_padding)
    text_x_max = min(img.shape[1], x_max - text_padding)
    text_y_max = min(img.shape[0], y_max - text_padding)

    if text_x_min >= text_x_max or text_y_min >= text_y_max:
        print(f"Warning: Text region for box {box_id} is too small. Using default text color.")
        return bg_color_rgb, DEFAULT_TEXT_COLOR_RGB

    text_mask = (mask[text_y_min:text_y_max, text_x_min:text_x_max] == 255)
    text_pixels = img[text_y_min:text_y_max, text_x_min:text_x_max][text_mask]

    if len(text_pixels) == 0:
        print(f"Warning: No text pixels found for box {box_id}. Using default text color.")
        return bg_color_rgb, DEFAULT_TEXT_COLOR_RGB

    MIN_PIXELS_FOR_KMEANS = 10
    if len(text_pixels) < MIN_PIXELS_FOR_KMEANS:
        text_color_bgr = np.mean(text_pixels, axis=0)
    else:
        text_pixels_float = text_pixels.reshape(-1, 3).astype(np.float32)
        _, _, centers = cv2.kmeans(text_pixels_float, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        bg_color_bgr_float = np.array([bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0]], dtype=np.float32)
        distances = np.linalg.norm(centers - bg_color_bgr_float, axis=1)
        color_index = np.argmax(distances)
        text_color_bgr = centers[color_index]

    text_color_rgb = (int(text_color_bgr[2]), int(text_color_bgr[1]), int(text_color_bgr[0]))
    return bg_color_rgb, text_color_rgb


def get_boxes_color_multiprocess(img, mask, box_ids, context):
    with ThreadPoolExecutor() as executor:
        # 将 context 传递给每个工作线程
        results = list(
            executor.map(lambda b_id: _process_single_box(b_id, img, mask, context), box_ids)
        )
    bg_colors, text_colors = zip(*results) if results else ([], [])
    return bg_colors, text_colors


def process_llm_image_translate(context, batch_size=128, language="en_zh"):
    items = list(context.agg_box_map_text_to_translate.items())
    aggregated_boxes = [item[0] for item in items]
    aggregated_texts = [item[1] for item in items]

    all_unchanged_agg_box_ids = []
    agg_translation_result = []

    for i in range(0, len(aggregated_texts), batch_size):
        batch_agg_texts = aggregated_texts[i:i + batch_size]
        batch_agg_box_ids = aggregated_boxes[i:i + batch_size]

        # 将 context 传递给翻译函数
        translated_map, batch_unchanged, agg_translated_map = llm_translate(
            batch_agg_box_ids, batch_agg_texts, context, len(batch_agg_texts), language=language
        )

        # 将翻译结果更新到 context 中
        context.ocr_box_map_translated_text.update(translated_map)
        for agg_box_id, (source_text, translated_text) in agg_translated_map.items():
            agg_translation_result.append(
                {
                    "coordinate": context.agg_box_map_coordinate[agg_box_id],
                    "source_text": source_text,
                    "translated_text": translated_text
                }
            )

        all_unchanged_agg_box_ids.extend(batch_unchanged)

    return all_unchanged_agg_box_ids, agg_translation_result


def inpaint_image(context, text_boxes, o_box_ids):
    cv_img = context.original_image.copy()  # 使用上下文中的原始图像
    start_at = time.time()

    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
    if text_boxes:
        formatted_boxes = [np.array(box, dtype=np.int32) for box in text_boxes]
        cv2.fillPoly(mask, formatted_boxes, 255)

    mask_done_at = time.time()
    print("Mask generation takes", (mask_done_at - start_at))

    # 将 context 传递给颜色提取函数
    box_bg_colors, box_text_colors = get_boxes_color_multiprocess(cv_img, mask, o_box_ids, context)
    # print("box_bg_colors", box_bg_colors, "box_text_colors", box_text_colors)
    get_boxes_color_done_at = time.time()
    print("Get boxes color takes", (get_boxes_color_done_at - mask_done_at))

    # 将 context 传递给模糊和文本添加函数
    inpainted_img = apply_blurred_overlay(cv_img, o_box_ids, box_bg_colors, context)
    print("Blur takes", (time.time() - get_boxes_color_done_at))
    # cv2.imwrite('bg_image.png', inpainted_img) # Debug line

    inpainted_img = add_translated_text(inpainted_img, o_box_ids, box_text_colors, context)
    return inpainted_img


# ==============================================================================
# 3. 重构主流程函数 translate_image
# ==============================================================================
def translate_image(img, ocr_result, language):
    st = time.time()
    if img is None:
        raise ValueError("Failed to load image")

    # 步骤1: 创建上下文对象
    context = ImageTranslationContext(img)

    # 步骤2: 获取OCR结果
    result = ocr_result
    if not result or not result[0] or len(result[0]) == 0:
        print("Warning: No text detected in image", result)
        return img, []  # 如果没有文字，直接返回原图

    # 步骤3: 解析OCR结果并存入上下文
    (context.ocr_box_ids, context.ocr_box_map_text, context.ocr_box_map_coordinate,
     context.ocr_box_map_width_height, context.ocr_box_map_angle_rad) = process_ocr_res(result[0])

    # 步骤4: 聚合OCR结果
    aggregated_boxes, _ = aggregate_ocr_results(img, None, result)

    # 步骤5: 过滤和处理聚合框，并将结果存入上下文
    (context.agg_box_map_coordinate, context.agg_box_map_origin, context.agg_box_map_text_to_translate,
     context.agg_box_map_text_to_keep) = process_and_filter_aggregated_boxes(aggregated_boxes, context.ocr_box_map_text,
                                                                             language)

    # 步骤6: 调用大模型进行翻译，函数内部会更新上下文中的翻译结果
    unchanged_agg_box_ids, agg_translation_result = process_llm_image_translate(context, language=language)

    # 步骤7: 确定需要重绘的原始框
    o_box_ids_to_inpaint = set()
    for agg_box_id, o_box_ids in context.agg_box_map_origin.items():
        if agg_box_id not in unchanged_agg_box_ids and agg_box_id not in context.agg_box_map_text_to_keep:
            o_box_ids_to_inpaint.update(o_box_ids)

    if not o_box_ids_to_inpaint:
        print("No text boxes needed inpainting. Returning original image.")
        return img, []

    # 步骤8: 计算字体大小并存入上下文
    filtered_agg_map_origin = {
        agg_id: o_ids for agg_id, o_ids in context.agg_box_map_origin.items()
        if agg_id not in unchanged_agg_box_ids and agg_id not in context.agg_box_map_text_to_keep
    }
    context.ocr_box_font_size_map = cal_box_font_size(
        filtered_agg_map_origin,
        context.ocr_box_map_width_height,
        context.ocr_box_map_translated_text,
        context.ocr_box_map_text,
    )

    # 步骤9: 获取需要重绘的框的坐标
    coords_to_inpaint = [context.ocr_box_map_coordinate.get(box_id) for box_id in o_box_ids_to_inpaint]
    coords_to_inpaint = [coord for coord in coords_to_inpaint if coord is not None]

    # 步骤10: 执行图像修复和文本重绘
    inpainted_image = inpaint_image(context, coords_to_inpaint, o_box_ids_to_inpaint)

    print("translate_image takes", time.time() - st)
    return inpainted_image, agg_translation_result


# 辅助函数，用于将paddleocr格式转为老格式，在画图时使用
def convert_format(data):
    output = []
    if not data: return output
    for sublist in data:
        for item in sublist:
            coords, (text, confidence) = item
            coords_array = np.array(coords, dtype=np.float32)
            output.append((coords_array, text, confidence))
    return output
