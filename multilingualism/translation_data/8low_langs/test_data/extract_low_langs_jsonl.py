#!/usr/bin/env python3
"""Extract selected English-centric language pairs from a JSONL file."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path


DEFAULT_TARGET_LANGS = ("bn", "te", "sr", "th", "vi", "hu", "cs")


def extract_records(
    input_path: Path,
    output_dir: Path,
    target_langs: list[str],
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[tuple[str, str], Path] = {}
    for language in target_langs:
        output_paths[("en", language)] = output_dir / f"en2{language}.jsonl"
        output_paths[(language, "en")] = output_dir / f"{language}2en.jsonl"

    counts = {f"{source}->{target}": 0 for source, target in output_paths}
    with input_path.open("r", encoding="utf-8") as source_file:
        with ExitStack() as stack:
            writers = {
                pair: stack.enter_context(path.open("w", encoding="utf-8"))
                for pair, path in output_paths.items()
            }
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

                pair = (record.get("src_lg"), record.get("trg_lg"))
                if pair in writers:
                    writers[pair].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[f"{pair[0]}->{pair[1]}"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-langs",
        nargs="+",
        default=list(DEFAULT_TARGET_LANGS),
        help="Language codes paired with English.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = extract_records(args.input, args.output_dir, args.target_langs)
    for pair, count in sorted(counts.items()):
        print(f"{pair}: {count}")


if __name__ == "__main__":
    main()
