#!/usr/bin/env python3
"""Merge two ShareGPT JSONL files and shuffle the combined records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-a", type=Path, required=True)
    parser.add_argument("--input-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    for path in (args.input_a, args.input_b):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")

    records_a = read_jsonl(args.input_a)
    records_b = read_jsonl(args.input_b)
    merged = records_a + records_b
    random.Random(args.seed).shuffle(merged)
    write_jsonl(args.output, merged)

    print(f"First input records: {len(records_a)}")
    print(f"Second input records: {len(records_b)}")
    print(f"Merged records: {len(merged)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Shuffle seed: {args.seed}")


if __name__ == "__main__":
    main()
