#!/usr/bin/env python3
"""Download and initialize the exact PaddleOCR models used by the image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-home", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--ocr-version", default="PP-OCRv4")
    parser.add_argument("--model-source", default="modelscope")
    parser.add_argument("--use-textline-orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_home = args.cache_home.resolve()
    cache_home.mkdir(parents=True, exist_ok=True)

    # PaddleX reads these variables at import time, so set them first.
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home)
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = args.model_source
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from paddleocr import PaddleOCR

    PaddleOCR(
        lang=args.language,
        ocr_version=args.ocr_version,
        device="cpu",
        engine="paddle",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=args.use_textline_orientation,
    )
    print(f"PaddleOCR models are available under {cache_home / 'official_models'}")


if __name__ == "__main__":
    main()
