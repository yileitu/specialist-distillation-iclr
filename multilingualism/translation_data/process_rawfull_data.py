#!/usr/bin/env python3
"""Convert ShareGPT translation records into inference data by direction."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path


REMOVE_SENTENCE = (
    "Please remember to output only the translation, without any additional "
    "explanation or commentary."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_direction_from_filename(filename: str) -> str | None:
    lower_name = filename.lower()
    if re.search(r"(^|[_-])en2[a-z]{2,3}([_-]|$)", lower_name):
        return "en2xx"
    if re.search(r"(^|[_-])[a-z]{2,3}2en([_-]|$)", lower_name):
        return "xx2en"
    if re.search(r"(^|[_-])en[_-][a-z]{2,3}([_-]|$)", lower_name):
        return "en2xx"
    if re.search(r"(^|[_-])[a-z]{2,3}[_-]en([_-]|$)", lower_name):
        return "xx2en"
    return None


def parse_direction_from_record(data: dict) -> str | None:
    source_language = data.get("src_lg")
    target_language = data.get("trg_lg")
    if (
        source_language == "en"
        and isinstance(target_language, str)
        and target_language != "en"
    ):
        return "en2xx"
    if (
        target_language == "en"
        and isinstance(source_language, str)
        and source_language != "en"
    ):
        return "xx2en"
    return None


def convert_record(data: dict) -> dict:
    user_content = ""
    gold_answer = ""
    for message in data.get("messages", []):
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            user_content = content.replace(REMOVE_SENTENCE, "").strip()
        elif role == "assistant":
            gold_answer = content.strip()
    return {
        "uuid": str(uuid.uuid4()),
        "user_content": user_content,
        "gold_answer": gold_answer,
    }


def process_one_file(input_path: Path, output_dir: Path) -> tuple[int, int]:
    en2xx_dir = output_dir / "en2xx"
    xx2en_dir = output_dir / "xx2en"
    en2xx_dir.mkdir(parents=True, exist_ok=True)
    xx2en_dir.mkdir(parents=True, exist_ok=True)
    out_en2xx = en2xx_dir / input_path.name
    out_xx2en = xx2en_dir / input_path.name
    count_en2xx = 0
    count_xx2en = 0

    with (
        input_path.open("r", encoding="utf-8") as source,
        out_en2xx.open("w", encoding="utf-8") as en2xx_file,
        out_xx2en.open("w", encoding="utf-8") as xx2en_file,
    ):
        filename_direction = parse_direction_from_filename(input_path.stem)
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} on line {line_number}: {exc}"
                ) from exc

            direction = parse_direction_from_record(data) or filename_direction
            if direction not in {"en2xx", "xx2en"}:
                continue
            output = convert_record(data)
            if direction == "en2xx":
                en2xx_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                count_en2xx += 1
            else:
                xx2en_file.write(json.dumps(output, ensure_ascii=False) + "\n")
                count_xx2en += 1

    if count_en2xx == 0:
        out_en2xx.unlink(missing_ok=True)
    if count_xx2en == 0:
        out_xx2en.unlink(missing_ok=True)
    return count_en2xx, count_xx2en


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    input_files = sorted(args.input_dir.glob("*.jsonl"))
    total_en2xx = 0
    total_xx2en = 0
    for index, input_file in enumerate(input_files, start=1):
        count_en2xx, count_xx2en = process_one_file(input_file, args.output_dir)
        total_en2xx += count_en2xx
        total_xx2en += count_xx2en
        print(
            f"[{index}/{len(input_files)}] {input_file.name}: "
            f"en2xx={count_en2xx}, xx2en={count_xx2en}",
            flush=True,
        )

    print(f"Input directory: {args.input_dir.resolve()}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Total en2xx records: {total_en2xx}")
    print(f"Total xx2en records: {total_xx2en}")


if __name__ == "__main__":
    main()
