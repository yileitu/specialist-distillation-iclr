"""
Sample from the difference set between full_all_subtasks and upsampled_50k_all_subtasks.

For each subtask (same-named JSONL), find records present in full but absent in upsampled,
then sample up to n_sample_per_task records per task from its diff set.
Tasks with an empty diff set are skipped.
"""

import json
import os
import random
import argparse
from pathlib import Path
from datetime import datetime

FULL_DIR = Path(__file__).resolve().parent.parent.parent / "smol" / "data" / "full_all_subtasks"
UPSAMPLED_DIR = Path(__file__).resolve().parent.parent.parent / "smol" / "data" / "upsampled_50k_all_subtasks"
OUTPUT_ROOT = Path(__file__).resolve().parent


def load_sample_ids(filepath: Path) -> set:
    ids = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ids.add(record["sample_id"])
    return ids


def load_diff_records(full_path: Path, upsampled_ids: set) -> list:
    diff = []
    with open(full_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["sample_id"] not in upsampled_ids:
                diff.append(record)
    return diff


def main():
    parser = argparse.ArgumentParser(description="Sample from diff set between full and upsampled data.")
    parser.add_argument("--n_sample_per_task", type=int, default=500, help="Max number of samples to draw per task from its diff set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    n_sample_per_task = args.n_sample_per_task

    full_files = sorted(FULL_DIR.glob("*.jsonl"))
    report_lines = []
    report_lines.append(f"Sampling Report")
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Random seed: {args.seed}")
    report_lines.append(f"Requested sample size per task: {n_sample_per_task}")
    report_lines.append(f"Full data dir: {FULL_DIR}")
    report_lines.append(f"Upsampled data dir: {UPSAMPLED_DIR}")
    
    OUTPUT_DIR = OUTPUT_ROOT / f"diff_sampled_{args.n_sample_per_task}_per_task"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines.append(f"Output dir: {OUTPUT_DIR}")
    report_lines.append("=" * 80)

    subtask_diffs: dict[str, list] = {}

    for full_file in full_files:
        subtask_name = full_file.stem
        upsampled_file = UPSAMPLED_DIR / full_file.name

        if not upsampled_file.exists():
            report_lines.append(f"\n[WARN] {subtask_name}: no matching upsampled file, skipping.")
            continue

        print(f"Processing {subtask_name}...")

        upsampled_ids = load_sample_ids(upsampled_file)
        # Deduplicate upsampled IDs (upsampled files may contain duplicated records)
        upsampled_unique_ids = upsampled_ids

        full_count = 0
        with open(full_file, "r") as f:
            for _ in f:
                full_count += 1

        diff_records = load_diff_records(full_file, upsampled_unique_ids)
        diff_count = len(diff_records)

        report_lines.append(f"\n[Subtask] {subtask_name}")
        report_lines.append(f"  Full count:          {full_count}")
        report_lines.append(f"  Upsampled unique IDs:{len(upsampled_unique_ids)}")
        report_lines.append(f"  Diff count:          {diff_count}")

        if diff_count == 0:
            report_lines.append(f"  -> Diff is empty, no sampling for this subtask.")
        else:
            subtask_diffs[subtask_name] = diff_records

    # Per-task sampling
    total_diff = sum(len(records) for records in subtask_diffs.values())

    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"Total diff records across all subtasks: {total_diff}")

    if total_diff == 0:
        report_lines.append("No records to sample. Exiting.")
        report_path = OUTPUT_DIR / "sampling_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"Report saved to {report_path}")
        return

    grouped: dict[str, list] = {}
    total_sampled = 0
    report_lines.append("\nSampled distribution by subtask:")
    for subtask_name in sorted(subtask_diffs.keys()):
        records = subtask_diffs[subtask_name]
        diff_total = len(records)
        actual = min(n_sample_per_task, diff_total)
        sampled = random.sample(records, actual) if actual < diff_total else records
        grouped[subtask_name] = sampled
        total_sampled += actual
        report_lines.append(f"  {subtask_name}: {actual} sampled (from {diff_total} diff records)")

    report_lines.append(f"\nTotal sampled across all subtasks: {total_sampled}")

    # Save per-subtask JSONL files
    saved_files = []
    for subtask_name, records in sorted(grouped.items()):
        out_path = OUTPUT_DIR / f"{subtask_name}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        saved_files.append((subtask_name, len(records), str(out_path)))
        print(f"  Saved {len(records)} records to {out_path}")

    report_lines.append("\nSaved files:")
    for name, cnt, path in saved_files:
        report_lines.append(f"  {path}  ({cnt} records)")

    report_path = OUTPUT_DIR / "sampling_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
