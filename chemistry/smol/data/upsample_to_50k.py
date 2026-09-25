#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upsample tasks under sample_50k_all_subtasks that have fewer than 50,000 records.

- Leave tasks with 50,000 records unchanged.
- If at most five copies reach 50,000 records, sample up to 50,000.
- Otherwise create five copies, producing original_count * 5 records.

Write the results to a new directory and create a report similar to
task_statistics.txt.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

TARGET = 500000
MAX_COPIES = 3
SEED = 42


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a single JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_upsampled_count(original_count: int) -> int:
    """
    Compute the upsampled record count.

    - If original_count >= 50,000, return 50,000; callers preserve the data without truncation.
    - If original_count * 5 >= 50,000, return 50,000.
    - Otherwise return original_count * 5.
    """
    if original_count >= TARGET:
        return original_count
    if original_count * MAX_COPIES >= TARGET:
        return TARGET
    return original_count * MAX_COPIES


def upsample_data(data: List[Dict[str, Any]], target_count: int, seed: int) -> List[Dict[str, Any]]:
    """Sample data with replacement up to target_count; return it unchanged when the target is not larger."""
    if not data:
        return data
    if target_count <= len(data):
        return data
    random.seed(seed)
    return [data[i] for i in random.choices(range(len(data)), k=target_count)]


def repeat_data(data: List[Dict[str, Any]], times: int) -> List[Dict[str, Any]]:
    """Repeat data sequentially the requested number of times."""
    return data * times


def run_upsample(
    input_dir: Path,
    output_dir: Path,
    seed: int = SEED,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    total_original = 0
    total_final = 0
    details = []

    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        if jsonl_path.name.startswith("."):
            continue
        task_name = jsonl_path.stem

        data = load_jsonl(jsonl_path)
        actual_count = len(data)

        total_original += actual_count
        target_count = compute_upsampled_count(actual_count)

        if actual_count >= TARGET:
            # Leave tasks with at least 50,000 records unchanged
            out_data = data
            final_count = actual_count
            note = "unchanged"
        elif actual_count * MAX_COPIES >= TARGET:
            # Upsample to 50,000 records
            out_data = upsample_data(data, TARGET, seed)
            final_count = len(out_data)
            note = f"upsampled to {TARGET}"
        else:
            # Create only five copies
            out_data = repeat_data(data, MAX_COPIES)
            final_count = len(out_data)
            note = f"repeated {MAX_COPIES} times"

        out_path = output_dir / f"{task_name}.jsonl"
        save_jsonl(out_path, out_data)
        total_final += final_count
        details.append({
            "task": task_name,
            "original_count": actual_count,
            "final_count": final_count,
            "output_file": f"{task_name}.jsonl",
            "note": note,
        })
        report_lines.append(f"  {task_name}: {actual_count} -> {final_count} ({note})")

    # Write the statistics report
    report_path = output_dir / "task_statistics.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Per-task upsampling statistics (upsample to 50k or repeat up to 5 times)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Target records: {TARGET}\n")
        f.write(f"Maximum repetitions: {MAX_COPIES}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total tasks: {len(details)}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Task':<42} {'Original':>12} {'Upsampled':>12} {'Output file':<36}\n")
        f.write("-" * 80 + "\n")
        for d in details:
            f.write(f"{d['task']:<42} {d['original_count']:>12,} {d['final_count']:>12,} {d['output_file']:<36}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Total':<42} {total_original:>12,} {total_final:>12,}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Additional information:\n")
        at_target = sum(1 for d in details if d["final_count"] >= TARGET)
        below_target = sum(1 for d in details if d["final_count"] < TARGET)
        f.write(f"- Tasks reaching {TARGET} records after upsampling: {at_target}\n")
        f.write(f"- Tasks below {TARGET} records after upsampling (repeated only 5 times): {below_target}\n")
        insufficient = [d for d in details if d["final_count"] < TARGET]
        if insufficient:
            f.write("\nTasks repeated only 5 times:\n")
            for d in insufficient:
                f.write(f"  - {d['task']}: {d['original_count']} -> {d['final_count']} records\n")

    print("Upsampling results:")
    for line in report_lines:
        print(line)
    print(f"\nTotal: {total_original} -> {total_final}")
    print(f"Statistics report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Upsample underrepresented tasks and write a statistics report")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing one JSONL file per task",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>_upsampled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for upsampling",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.with_name(f"{input_dir.name}_upsampled")
    run_upsample(
        input_dir=input_dir,
        output_dir=output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
