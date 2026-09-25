#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all jsonl files under a directory, shuffle, then sample N lines."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to recursively search for jsonl files. Default: script directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output jsonl path. Default: <input-dir>/merged_shuffled_sampled_51957_seed42.jsonl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=51957,
        help="How many samples to keep after shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffle.",
    )
    return parser.parse_args()


def collect_jsonl_files(input_dir: Path, output_file: Path | None) -> list[Path]:
    files = sorted(p for p in input_dir.rglob("*.jsonl") if p.is_file())
    if output_file is not None:
        output_abs = output_file.resolve()
        files = [p for p in files if p.resolve() != output_abs]
    return files


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_file = (
        args.output_file.resolve()
        if args.output_file is not None
        else input_dir / "merged_shuffled_sampled_51957_seed42.jsonl"
    )

    jsonl_files = collect_jsonl_files(input_dir, output_file)
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found under: {input_dir}")

    all_lines: list[str] = []
    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_lines.append(line)

    total = len(all_lines)
    if total < args.sample_size:
        raise ValueError(
            f"Not enough samples: found {total}, requested {args.sample_size}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(all_lines)
    sampled = all_lines[: args.sample_size]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    print(f"Input directory: {input_dir}")
    print(f"Found jsonl files: {len(jsonl_files)}")
    print(f"Total lines merged: {total}")
    print(f"Sample size: {len(sampled)}")
    print(f"Seed: {args.seed}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
