import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4
import multiprocessing as mp

import httpx
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from clients.oss_client import oss_client_instance
from src.image_translation.config import get_settings
from src.image_translation.image_translation import translate_image
from src.image_translation.components.ocr.ocr_worker import ocr_worker_process
from src.image_translation.image_segmentation import segment_image, adjust_box_heights

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = get_settings()

# --- 全局资源，由 lifespan 管理 ---
# 我们不再在全局定义 Queue 和 Process
# 它们将在 lifespan 中创建并附加到 app.state

# 使用一个字典来存储等待结果的 Event 对象
# key 是 task_id, value 是 (asyncio.Event, result_storage)
pending_tasks = {}
pending_tasks_lock = asyncio.Lock()


async def result_listener(result_q: mp.Queue):
    """
    一个后台任务，持续监听结果队列，并将结果分发给等待的请求。
    **这个版本能够解析子任务ID并处理批处理任务**
    """
    print("[Result Listener] Started.")
    loop = asyncio.get_running_loop()
    while True:
        try:
            # 在一个线程中执行阻塞的 get()，以避免阻塞事件循环
            task_id, result = await loop.run_in_executor(None, result_q.get)

            # 收到关闭信号
            if task_id is None:
                print("[Result Listener] Received shutdown signal. Exiting.")
                break

            # 约定子任务ID格式为 "request_id_segment_index"
            parts = task_id.rsplit('_', 1)
            is_batch_subtask = len(parts) == 2 and parts[1].isdigit()

            async with pending_batch_tasks_lock:
                if is_batch_subtask:
                    request_id, segment_index_str = parts
                    segment_index = int(segment_index_str)

                    if request_id in pending_batch_tasks:
                        batch_info = pending_batch_tasks[request_id]
                        # 将结果存入对应的 segment_index
                        batch_info['results'][segment_index] = result

                        # 检查是否所有子任务都已完成
                        if len(batch_info['results']) == batch_info['expected_count']:
                            print(
                                f"[Result Listener] All {batch_info['expected_count']} parts received for request {request_id}. Notifying main task.")
                            batch_info['event'].set()  # 通知主任务
                    else:
                        # 这里的警告现在更有针对性
                        logger.warning(
                            f"Received result for an unknown or expired batch request_id: {request_id} (from task_id: {task_id})")
                else:
                    # 如果遇到不符合 "request_id_index" 格式的 task_id
                    logger.warning(f"Received result for a non-standard task_id format: {task_id}")

        except Exception as e:
            logger.error(f"Error in result_listener: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Main App] Starting up...")
    ctx = mp.get_context("spawn")

    # 在 lifespan 中创建队列
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    # 启动 OCR 工作进程
    ocr_process = ctx.Process(target=ocr_worker_process, args=(task_q, result_q))
    ocr_process.start()

    # 等待 OCR 进程初始化完成的信号
    print("[Main App] Waiting for OCR worker to initialize...")
    init_status, init_error = result_q.get()
    if init_status == "INIT_FAILED":
        raise RuntimeError(f"OCR Worker initialization failed: {init_error}")
    print("[Main App] OCR worker is ready.")

    # 启动后台的结果监听任务
    listener_task = asyncio.create_task(result_listener(result_q))

    # 将需要的对象保存在 app.state 中
    app.state.task_queue = task_q
    app.state.ocr_process = ocr_process
    app.state.listener_task = listener_task

    # 设置 OSS 客户端
    limits = httpx.Limits(
        max_connections=settings.http.max_connections,
        max_keepalive_connections=settings.http.max_keepalive_connections,
        keepalive_expiry=settings.http.keepalive_expiry_seconds,
    )

    shared_client = httpx.AsyncClient(timeout=settings.http.request_timeout_seconds, limits=limits)
    oss_client_instance.set_shared_client(shared_client)

    yield

    # --- 应用关闭 ---
    print("[Main App] Shutting down...")
    await oss_client_instance.close_shared_client()

    # 优雅地关闭后台任务和工作进程
    print("[Main App] Stopping result listener...")
    app.state.task_queue.put((None, None))  # 发送信号给 listener 和 worker
    await asyncio.sleep(1)  # 等待信号被处理
    app.state.listener_task.cancel()

    ocr_process = app.state.ocr_process
    ocr_process.join(timeout=settings.ocr.worker_shutdown_timeout_seconds)
    if ocr_process.is_alive():
        print("[Main App] OCR worker did not terminate gracefully, killing.")
        ocr_process.terminate()
    print("[Main App] Shutdown complete.")


app = FastAPI(lifespan=lifespan)

pending_batch_tasks = {}
pending_batch_tasks_lock = asyncio.Lock()


async def run_ocr(request: Request, image_content: bytes, segment_enabled: bool = False) -> list:
    task_queue = request.app.state.task_queue
    request_id = str(uuid4())  # 这是整个请求的唯一ID

    tasks_to_submit = []  # 存储 (sub_task_id, segment_bytes, (x_offset, y_offset))

    if segment_enabled:
        # --- 1.1 拆分大图片 ---
        # segment_image 现在返回 [(segment_bytes, (x, y)), ...]
        image_segments = segment_image(image_content)

        if not image_segments:
            logger.warning(
                f"Image segmentation yielded no segments for request {request_id}. Returning empty OCR result.")
            return []  # 没有分片，返回空结果

        for i, image_segment in enumerate(image_segments):
            sub_task_id = f"{request_id}_{i}"
            tasks_to_submit.append(
                (sub_task_id, image_segment.image_data, (image_segment.x_offset, image_segment.y_offset)))
        num_expected_results = len(tasks_to_submit)
        logger.info(
            f"[Main App] Request {request_id}: Segmenting image. Submitting {num_expected_results} OCR sub-tasks.")
    else:
        # --- 1.2 不分割图片，直接将整张图片作为单个任务 ---
        sub_task_id = f"{request_id}_0"  # 即使不分割，也使用统一的命名约定
        # 整张图片，偏移量为 (0, 0)
        tasks_to_submit.append((sub_task_id, image_content, (0, 0)))
        num_expected_results = 1
        logger.info(f"[Main App] Request {request_id}: Not segmenting image. Submitting 1 OCR task.")

    if num_expected_results == 0:
        return []  # 再次检查，以防万一

    # --- 2. 准备等待所有任务的结果 ---
    event = asyncio.Event()
    batch_info = {
        'event': event,
        'results': {},  # 存储以 segment_index 为键的结果
        'expected_count': num_expected_results
    }

    async with pending_batch_tasks_lock:
        pending_batch_tasks[request_id] = batch_info

    # --- 3. 提交所有任务 ---
    for sub_task_id, segment_bytes, _ in tasks_to_submit:
        task_queue.put((sub_task_id, segment_bytes))

    # --- 4. 等待所有子任务完成 ---
    try:
        await asyncio.wait_for(event.wait(), timeout=settings.ocr.task_timeout_seconds)

        # 确保按照原始提交顺序获取结果，即使 result_listener 可能乱序写入
        # result_listener 使用 segment_index 作为 key 存储结果
        # tasks_to_submit[i] 包含了原始的 (sub_task_id, segment_bytes, offset)
        # 我们可以通过 sub_task_id 提取 segment_index
        processed_results = []
        for i, (sub_task_id, _, offset) in enumerate(tasks_to_submit):
            parts = sub_task_id.rsplit('_', 1)
            segment_index = int(parts[1])  # 获取原始的 segment_index

            ocr_result_segment = batch_info['results'][segment_index]
            processed_results.append((ocr_result_segment, offset))

        # 检查是否有任何一个子任务失败，并抛出异常
        for ocr_res, _ in processed_results:
            if isinstance(ocr_res, Exception):
                raise ocr_res

    except Exception:
        raise  # 捕获任何异常以确保清理，然后重新抛出
    finally:
        async with pending_batch_tasks_lock:
            if request_id in pending_batch_tasks:
                del pending_batch_tasks[request_id]

    logger.info(f"[Main App] Request {request_id}: All {num_expected_results} OCR results received.")

    # --- 5. 聚合OCR结果并调整坐标 ---
    combined_ocr_result = []
    for ocr_result_segment, (x_offset, y_offset) in processed_results:
        if not ocr_result_segment:  # 跳过没有结果的分片
            continue

        ocr_lines = ocr_result_segment[0]
        if not ocr_lines:
            continue

        for line_result in ocr_lines:
            if not line_result:
                continue

            box, (text, score) = line_result
            # 调整box中每个点的坐标
            adjusted_box = [[point[0] + x_offset, point[1] + y_offset] for point in box]
            combined_ocr_result.append([adjusted_box, (text, score)])

    # PaddleOCR期望的结果格式是一个包含单个列表的列表: [[...]]

    final_ocr_result = adjust_box_heights(
        [combined_ocr_result], settings.ocr.box_height_tolerance_ratio
    )

    return final_ocr_result


@app.post(settings.api.image_translate_path)
async def process_image_translate(request: Request):
    time_start = time.time()
    req_data = await request.json()
    image_url = req_data.get('image_url')
    language = req_data.get('language', 'en_zh')
    bucket_name = req_data.get('bucket')
    save_bucket_name = req_data.get('save_bucket', bucket_name)
    segment_enabled = req_data.get('segment', False)

    if not all([image_url, bucket_name]):
        return JSONResponse(status_code=400, content={"error": "Missing 'image_url' or 'bucket'"})

    # --- 1. 下载图片 ---
    image_data_info = await oss_client_instance.get_oss_file(bucket_name, image_url, as_string=False)
    image_content = image_data_info['content']

    # --- 2. 执行分片OCR并聚合结果 (调用新函数) ---
    try:
        ocr_result = await run_ocr(request, image_content, segment_enabled)
    except asyncio.TimeoutError:
        logger.error(f"OCR processing timed out for image_url: {image_url}")
        return JSONResponse(status_code=504, content={"error": "OCR processing timed out"})
    except Exception as e:
        logger.error(f"An error occurred during OCR processing for {image_url}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"An internal error occurred during OCR: {e}"})

    # --- 3. 解码图片 ---
    # 这个操作很快，可以不在线程中运行，但为了安全也可以保留
    image_array = np.frombuffer(image_content, np.uint8)
    image_obj = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # --- 4. 核心翻译逻辑 ---
    res_img, res_msg = await asyncio.to_thread(translate_image, image_obj, ocr_result, language)

    # --- 5. 编码并上传 ---
    def encode_image(img_obj):
        """一个包裹编码操作的函数"""
        print("Encoding image in a thread...")
        success, buffer = cv2.imencode('.jpg', img_obj)
        if not success:
            raise ValueError("Failed to encode image.")
        return buffer.tobytes()

    res_img_bytes = await asyncio.to_thread(encode_image, res_img)

    # async with OSS_SEMAPHORE:
    print("Uploading translated image to OSS...")
    new_image_key = f"{language}_translated_{image_url}"
    oss_resp = await oss_client_instance.upload_image(save_bucket_name, new_image_key, res_img_bytes)
    print("Upload response:", oss_resp)

    print("res_msg", res_msg)
    print(f"Image translate took {time.time() - time_start:.2f} seconds.")

    # 5. 在成功路径中直接返回成功的 JSON 响应
    source_image_url = f"{bucket_name}/{image_url}"
    translated_image_url = f"{save_bucket_name}/{new_image_key}"
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "source_url": source_image_url,
            "translated_url": translated_image_url,  # 假设上传后会返回URL
            "data": res_msg,
        }
    )


# 7. 使用 uvicorn 来运行 FastAPI 应用
if __name__ == "__main__":
    # 这种方式适合在开发时直接运行脚本
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
