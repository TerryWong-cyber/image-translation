#!/usr/bin/env python3
"""Validate local Paddle model artifacts and optionally write a hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = ("inference.json", "inference.yml", "inference.pdiparams")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--write-manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, model_dirs: list[str]) -> dict[str, object]:
    root = root.resolve()
    models: dict[str, object] = {}
    manifest: dict[str, object] = {"root": str(root), "models": models}

    for relative_dir in model_dirs:
        model_dir = root / relative_dir
        if not model_dir.is_dir():
            raise ValueError(f"Missing Paddle model directory: {model_dir}")
        missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ValueError(
                f"Paddle model directory {model_dir} is missing: {', '.join(missing)}"
            )
        files = {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(model_dir.rglob("*"))
            if path.is_file()
        }
        models[relative_dir] = files
        print(f"Verified {relative_dir}: {len(files)} files")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.root, args.model_dir)

    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote model manifest: {args.write_manifest}")


if __name__ == "__main__":
    main()
