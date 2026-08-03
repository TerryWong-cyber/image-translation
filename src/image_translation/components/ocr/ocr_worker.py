import time
import cv2
import base64
import numpy as np
from paddleocr import PaddleOCR

from clients.call_llm_recognition import call_llm_recognition_api_batch
from image_translation.config import get_settings


def det_to_ocr_result(image, det_result):
    st = time.time()

    if image is None:
        print(f"错误: 传入的图像对象为 None。")
        return []

    if not det_result or not det_result[0]:
        print("警告: 检测结果为空，无法进行裁切。")
        return []

    boxes = det_result[0]
    base64_images = []  # 用于存储所有裁切图的base64编码

    # 1. 先循环准备好所有图片数据
    for i, box in enumerate(boxes):
        src_pts = np.array(box, dtype="float32")

        width_top = np.sqrt(((src_pts[0][0] - src_pts[1][0]) ** 2) + ((src_pts[0][1] - src_pts[1][1]) ** 2))
        width_bottom = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
        max_width = int(max(width_top, width_bottom))

        height_left = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
        height_right = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
        max_height = int(max(height_left, height_right))

        if max_width <= 0 or max_height <= 0:
            print(f"警告: 第 {i + 1} 个裁切区域尺寸无效 (w:{max_width}, h:{max_height})，已跳过。")
            continue

        dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                           dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_image = cv2.warpPerspective(image, M, (max_width, max_height))

        # 将裁切图编码为base64
        is_success, buffer = cv2.imencode('.png', warped_image)
        if not is_success:
            print(f"警告: 无法编码第 {i + 1} 个裁切区域。")
            continue

        base64_string = base64.b64encode(buffer).decode('utf-8')
        base64_images.append(base64_string)

    if not base64_images:
        print("没有可识别的图像区域。")
        return []

    # 2. 一次性发送所有图片进行识别 (这部分逻辑不变)
    print(f"--- 准备发送 {len(base64_images)} 个图像进行批量识别 ---")
    rec_results = call_llm_recognition_api_batch(base64_images)

    if not rec_results or len(rec_results) != len(boxes):
        print("错误: 批量识别返回结果数量与请求不匹配或返回为空。")
        return []

    # 3. 将识别结果与原始box对应起来 (这部分逻辑不变)
    ocr_result = []
    for box, rec_text in zip(boxes, rec_results):
        ocr_result.append([box, (rec_text, 0.99)])  # 假设置信度为0.99

    print(f"提取文字总耗时: {time.time() - st:.4f}s")
    return [ocr_result]


def has_non_zh_en_text(image):
    """
    判断图片文字是否包含非中文或英文的语言

    Returns:
        (bool, str)
        bool: 是否包含非中英语言
        str : 错误原因，没有错误则为 ""
    """

    try:
        settings = get_settings().llm
        # 编码图片
        is_success, buffer = cv2.imencode(".jpg", image)

        if not is_success:
            return False, "image_encode_failed"

        base64_string = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # 调用 VLM
        rec_results = call_llm_recognition_api_batch(
            [base64_string],
            prompt=settings.prompt_language_detection,
            url=settings.recognition_batch_url,
        )

        if not rec_results:
            return False, "empty_model_response"

        if len(rec_results) != 1:
            return False, "unexpected_response_length"

        result = rec_results[0].strip().lower()

        if result == "true":
            return True, ""
        elif result == "false":
            return False, ""
        else:
            return False, f"invalid_model_output: {result}"

    except Exception as e:
        return False, str(e)


def ocr_worker_process(task_queue, result_queue):
    """
    这是一个在独立进程中运行的函数。
    它负责初始化OCR引擎，并处理来自任务队列的任务。
    """
    print("[OCR Worker] Process started. Initializing PaddleOCR engine...")
    try:
        settings = get_settings().ocr
        # 在工作进程内部初始化，保证资源隔离
        ocr_engine = PaddleOCR(
            lang=settings.language,
            det=True,
            rec=True,
            det_db_unclip_ratio=settings.unclip_ratio,
        )

        # 针对非中英的外语情况，paddle只负责文字检测，识别部分交给调用LLM的方式来做，所以需要再初始化一个只负责检测的引擎
        det_engine = PaddleOCR(
            lang=settings.language,
            det=True,
            rec=False,
            det_db_unclip_ratio=settings.unclip_ratio,
        )
        print("[OCR Worker] PaddleOCR engine initialized successfully.")
    except Exception as e:
        print(f"[OCR Worker] FATAL: Failed to initialize PaddleOCR engine: {e}")
        # 将一个错误信号放入结果队列，通知主进程
        # 这里使用一个特殊的元组来表示初始化失败
        result_queue.put(("INIT_FAILED", str(e)))
        return  # 退出进程

    # 发送一个信号，表示初始化成功
    result_queue.put(("INIT_OK", None))

    while True:
        try:
            # 从任务队列中获取任务，这是一个阻塞操作
            task_id, image_content_bytes = task_queue.get()

            # 检查是否有退出信号
            if image_content_bytes is None:
                print("[OCR Worker] Received shutdown signal. Exiting.")
                break

            print(f"[OCR Worker] Received task {task_id}. Performing OCR...")

            # --- 核心OCR处理逻辑 ---
            image_array = np.frombuffer(image_content_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image in worker.")

            start_time = time.time()

            ocr_result = None
            has, err_msg = has_non_zh_en_text(image=img)
            if err_msg and err_msg != "":
                print(f"[OCR Worker] Warning: Error during language check: {err_msg}. Defaulting to standard OCR.")

            if has:
                det_result = det_engine.ocr(img, cls=settings.use_angle_classifier, rec=False)
                ocr_result = det_to_ocr_result(img, det_result)
            else:
                ocr_result = ocr_engine.ocr(img, cls=settings.use_angle_classifier)

            elapsed = time.time() - start_time
            print(f"[OCR Worker] Task {task_id} OCR finished in {elapsed:.2f}s.")

            # 将处理结果和任务ID一起放回结果队列
            result_queue.put((task_id, ocr_result))

        except Exception as e:
            print(f"[OCR Worker] Error processing task: {e}")
            # 如果处理出错，也需要向结果队列发送一个信号
            if 'task_id' in locals():
                result_queue.put((task_id, e))  # 将异常对象作为结果
            else:  # 如果在 get() 之前就出错
                pass
