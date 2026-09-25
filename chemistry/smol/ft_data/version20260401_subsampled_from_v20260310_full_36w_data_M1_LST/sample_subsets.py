"""
Sample subsets (100k, 150k, 200k, 250k) from the 36w aligned FT data without replacement.
"""

import json
import random
from pathlib import Path

SEED = 42
SOURCE = Path(__file__).resolve().parent.parent / (
    "version20260310_full_data_new_filtration_qwen3_8b/aligned_data/"
    "aligned_ft_36w_data_M1_LST.jsonl"
)
OUTPUT_DIR = Path(__file__).resolve().parent
# SAMPLE_SIZES = [100_000, 150_000, 200_000, 250_000]
SAMPLE_SIZES = [300_000]


def main():
    print(f"Loading data from {SOURCE} ...")
    with open(SOURCE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    print(f"Total lines: {total}")

    random.seed(SEED)
    indices = list(range(total))
    random.shuffle(indices)

    max_needed = max(SAMPLE_SIZES)
    assert max_needed <= total, f"Requested {max_needed} but only {total} available"

    for size in SAMPLE_SIZES:
        subset_indices = sorted(indices[:size])
        out_path = OUTPUT_DIR / f"{size // 10000}w_data_M1_LST_from_full_36w.jsonl"
        print(f"Writing {size} samples -> {out_path.name}")
        with open(out_path, "w", encoding="utf-8") as fout:
            for idx in subset_indices:
                fout.write(lines[idx])
        print(f"  Done. ({out_path})")

    print("All subsets created.")


if __name__ == "__main__":
    main()
