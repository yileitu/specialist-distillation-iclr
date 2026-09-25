#!/usr/bin/env python3
"""Compute mean spBLEU for a fixed language subset in a FLORES-style CSV.

The first CSV column must contain language codes and the second column must
contain spBLEU scores.

Usage:
  python util/mean_spbleu_subset.py /path/to/en2xx_results.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys


TARGET_LANGS = frozenset("hu vi es cs fr de ru bn sr ko ja ar th sw zh te".split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean spBLEU over a fixed language subset."
    )
    parser.add_argument("csv_path", help="Path to a FLORES-style evaluation CSV.")
    args = parser.parse_args()

    values: list[float] = []
    skipped_empty: list[str] = []
    seen_target: set[str] = set()

    with open(args.csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            lang = row[0].strip()
            if lang not in TARGET_LANGS:
                continue
            seen_target.add(lang)
            cell = row[1].strip()
            if not cell:
                skipped_empty.append(lang)
                continue
            try:
                values.append(float(cell))
            except ValueError:
                skipped_empty.append(lang)

    missing = sorted(TARGET_LANGS - seen_target)
    if not values:
        print("No spBLEU values collected.", file=sys.stderr)
        if missing:
            print(f"No row for: {', '.join(missing)}", file=sys.stderr)
        if skipped_empty:
            invalid = ", ".join(sorted(set(skipped_empty)))
            print(f"Empty/invalid spBLEU: {invalid}", file=sys.stderr)
        raise SystemExit(1)

    mean = statistics.mean(values)
    print(f"Languages (n={len(values)}): mean spBLEU = {mean:.4f}")
    if skipped_empty:
        invalid = ", ".join(sorted(set(skipped_empty)))
        print(f"Skipped empty/invalid spBLEU: {invalid}")
    if missing:
        print(f"No row in CSV for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
