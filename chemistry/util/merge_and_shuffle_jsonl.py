#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge jsonl files from subfolders and shuffle all lines."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Input root directory",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL file (default: <input-dir>/merged_shuffled.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    return parser.parse_args()


def collect_jsonl_files(input_dir: Path, output_file: Path) -> list[Path]:
    jsonl_files: list[Path] = []
    output_file_resolved = output_file.resolve()
    for path in sorted(input_dir.rglob("*.jsonl")):
        if path.resolve() == output_file_resolved:
            continue
        # Keep files in subfolders; ignore jsonl directly under input root.
        if path.parent.resolve() == input_dir.resolve():
            continue
        jsonl_files.append(path)
    return jsonl_files


def merge_and_shuffle(jsonl_files: list[Path], seed: int) -> list[str]:
    lines: list[str] = []
    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

    rnd = random.Random(seed)
    rnd.shuffle(lines)
    return lines


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_file: Path = args.output_file or input_dir / "merged_shuffled.jsonl"
    seed: int = args.seed

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    jsonl_files = collect_jsonl_files(input_dir, output_file)
    if not jsonl_files:
        raise SystemExit(f"No jsonl files found under subfolders of: {input_dir}")

    merged_lines = merge_and_shuffle(jsonl_files, seed)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for line in merged_lines:
            f.write(line + "\n")

    print(f"Input dir      : {input_dir}")
    print(f"JSONL file count: {len(jsonl_files)}")
    print(f"Output file    : {output_file}")
    print(f"Total lines    : {len(merged_lines)}")
    print(f"Shuffle seed   : {seed}")


if __name__ == "__main__":
    main()
