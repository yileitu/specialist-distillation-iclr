#!/usr/bin/env python3
"""Extract English-Swahili records in both directions from a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_records(
    input_path: Path,
    en_sw_output_path: Path,
    sw_en_output_path: Path,
) -> tuple[int, int]:
    en_sw_count = 0
    sw_en_count = 0
    en_sw_output_path.parent.mkdir(parents=True, exist_ok=True)
    sw_en_output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as source,
        en_sw_output_path.open("w", encoding="utf-8") as en_sw_file,
        sw_en_output_path.open("w", encoding="utf-8") as sw_en_file,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            language_pair = (record.get("src_lg"), record.get("trg_lg"))
            if language_pair == ("en", "sw"):
                en_sw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                en_sw_count += 1
            elif language_pair == ("sw", "en"):
                sw_en_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                sw_en_count += 1
    return en_sw_count, sw_en_count


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--en-sw-output",
        type=Path,
        default=output_dir / "en2sw.jsonl",
    )
    parser.add_argument(
        "--sw-en-output",
        type=Path,
        default=output_dir / "sw2en.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    en_sw_count, sw_en_count = extract_records(
        args.input,
        args.en_sw_output,
        args.sw_en_output,
    )
    print(f"Done. en->sw: {en_sw_count}, sw->en: {sw_en_count}")
    print(f"en->sw output: {args.en_sw_output}")
    print(f"sw->en output: {args.sw_en_output}")


if __name__ == "__main__":
    main()
