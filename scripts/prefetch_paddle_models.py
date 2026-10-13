#!/usr/bin/env python3
"""Download production model files without importing Paddle or running inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


REQUIRED_FILES = ("inference.json", "inference.yml", "inference.pdiparams")
OCR_MODELS = ("PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec")
ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori"


def download_models(cache_home: Path, use_textline_orientation: bool,
                    snapshot_download: Callable[..., str]) -> None:
    models = ((ORIENTATION_MODEL,) if use_textline_orientation else ()) + OCR_MODELS
    for name in models:
        destination = cache_home.resolve() / "official_models" / name
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(model_id=f"PaddlePaddle/{name}", local_dir=str(destination))
        missing = [filename for filename in REQUIRED_FILES
                   if not (destination / filename).is_file() or (destination / filename).stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Incomplete model {name}: missing or empty files: {', '.join(missing)}")
        print(f"Downloaded model files: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-home", type=Path, required=True)
    # This helper matches Dockerfile.gpu and the runtime's explicit model paths.
    parser.add_argument("--language", choices=["en"], default="en")
    parser.add_argument("--ocr-version", choices=["PP-OCRv4"], default="PP-OCRv4")
    parser.add_argument("--model-source", choices=["modelscope"], default="modelscope")
    parser.add_argument("--use-textline-orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_home = args.cache_home.resolve()
    cache_home.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    download_models(cache_home, args.use_textline_orientation, snapshot_download)
    print(f"PaddleOCR model files are available under {cache_home / 'official_models'}")


if __name__ == "__main__":
    main()
