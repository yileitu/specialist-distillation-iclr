#!/usr/bin/env python3
"""Validate ShareGPT JSONL data and register it with LLaMA-Factory.

Each non-empty input line must be a JSON object containing a ``messages`` list.
Every message must contain ``role`` and ``content`` fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name (defaults to the input file name).",
    )
    parser.add_argument(
        "--dataset_info_path",
        type=Path,
        required=True,
        help="Path to LLaMA-Factory's dataset_info.json file.",
    )
    parser.add_argument(
        "--copy_to_dir",
        type=Path,
        default=None,
        help="Optional directory to which the data file is copied.",
    )
    return parser.parse_args()


def validate_data_format(data_file: Path) -> int:
    """Validate that a JSONL file follows the expected ShareGPT format."""
    count = 0
    with data_file.open("r", encoding="utf-8") as source:
        for line_num, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc

            if "messages" not in data:
                raise ValueError(f"Line {line_num} is missing the 'messages' field")
            messages = data["messages"]
            if not isinstance(messages, list):
                raise ValueError(f"Line {line_num}: 'messages' must be a list")
            for message in messages:
                if "role" not in message or "content" not in message:
                    raise ValueError(
                        f"Line {line_num}: a message is missing 'role' or 'content'"
                    )
            count += 1
    return count


def register_dataset(
    dataset_info_path: Path,
    dataset_name: str,
    data_file_path: Path,
) -> None:
    """Add or replace a dataset entry in ``dataset_info.json``."""
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as registry_file:
            dataset_info = json.load(registry_file)
    else:
        dataset_info = {}
        dataset_info_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_info[dataset_name] = {
        "file_name": str(data_file_path),
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }

    with dataset_info_path.open("w", encoding="utf-8") as registry_file:
        json.dump(dataset_info, registry_file, indent=2, ensure_ascii=False)

    print(f"Registered dataset '{dataset_name}' in {dataset_info_path}")


def main() -> None:
    args = parse_args()
    if not args.data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    dataset_name = args.dataset_name or args.data_file.stem
    print(f"Validating data: {args.data_file}")
    sample_count = validate_data_format(args.data_file)
    print(f"Validation passed: {sample_count} samples")

    final_data_path = args.data_file.resolve()
    if args.copy_to_dir:
        args.copy_to_dir.mkdir(parents=True, exist_ok=True)
        final_data_path = args.copy_to_dir / args.data_file.name
        shutil.copy2(args.data_file, final_data_path)
        final_data_path = final_data_path.resolve()
        print(f"Copied data file to: {final_data_path}")

    register_dataset(
        args.dataset_info_path,
        dataset_name,
        final_data_path,
    )
    print(f"Dataset name: {dataset_name}")
    print(f"Data file: {final_data_path}")
    print(f"Sample count: {sample_count}")


if __name__ == "__main__":
    main()
