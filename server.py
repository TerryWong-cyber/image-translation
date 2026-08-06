# server.py
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_translation.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print("🚀 正在启动 Image Translation 服务...")

    # 启动 FastAPI 服务
    uvicorn.run(
        settings.server.app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        reload_dirs=list(settings.server.reload_dirs) if settings.server.reload else None,
    )
