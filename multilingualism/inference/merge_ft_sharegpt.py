#!/usr/bin/env python3
"""Merge and shuffle ShareGPT JSONL files discovered below input roots."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_FILE_NAME = "ft_sharegpt_top20pct.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Root to search recursively; repeat for multiple roots.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-name", default=DEFAULT_FILE_NAME)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_files(roots: list[Path], file_name: str) -> list[Path]:
    files: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Input root not found: {root}")
        for path in root.rglob(file_name):
            files[str(path.resolve())] = path
    return sorted(files.values(), key=lambda path: str(path.resolve()))


def collect_lines(files: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in files:
        with path.open("r", encoding="utf-8") as source:
            lines.extend(line.rstrip("\r\n") for line in source if line.strip())
    return lines


def main() -> int:
    args = parse_args()
    files = discover_files(args.input_root, args.file_name)
    if not files:
        print(f"No {args.file_name!r} files found.")
        return 1

    lines = collect_lines(files)
    random.Random(args.seed).shuffle(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for line in lines:
            output.write(line + "\n")

    print(f"Input roots: {len(args.input_root)}")
    print(f"JSONL files: {len(files)}")
    print(f"Records: {len(lines)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Shuffle seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
