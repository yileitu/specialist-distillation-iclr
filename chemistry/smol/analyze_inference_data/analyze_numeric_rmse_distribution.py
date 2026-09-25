#!/usr/bin/env python3
"""
Analyze RMSE for numeric property prediction tasks from postprocessed JSONL files.

For each input JSONL file:
- Compute RMSE between extracted_core_answer (3 generations) and ground_truth, separately per generation.
- Save RMSEs and their mean to a local TXT file.
- Plot a single distribution (histogram) of (prediction - ground_truth) across all generations mixed together,
  and annotate mean/median/std on the figure.

Usage:
  python analyze_numeric_rmse_distribution.py <jsonl1> [<jsonl2> ...] [--output-dir OUT]

Example:
  python analyze_numeric_rmse_distribution.py \
    /path/to/property_prediction-esol/split1000_num_postprocessed_tolerance1.0.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dict records."""
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
    return data


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _sorted_generation_keys(keys: Iterable[str]) -> List[str]:
    # Prefer generation1, generation2, generation3 ordering if present.
    def key_fn(k: str) -> Tuple[int, str]:
        if k.startswith("generation"):
            suffix = k[len("generation") :]
            try:
                return (0, f"{int(suffix):08d}")
            except ValueError:
                return (1, k)
        return (2, k)

    return sorted(set(keys), key=key_fn)


def extract_pairs_and_diffs(
    data: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Tuple[float, float]]], List[float]]:
    """
    Extract per-generation (pred, gt) pairs and the mixed diffs (pred - gt).

    Returns:
      - gen_to_pairs: dict gen_key -> list[(pred, gt)]
      - all_diffs: list[pred - gt] across all generations combined
    """
    gen_keys: List[str] = []
    for rec in data:
        extracted = rec.get("extracted_core_answer") or {}
        if isinstance(extracted, dict):
            gen_keys.extend([k for k in extracted.keys() if isinstance(k, str)])

    gen_keys_sorted = _sorted_generation_keys(gen_keys)
    gen_to_pairs: Dict[str, List[Tuple[float, float]]] = {k: [] for k in gen_keys_sorted}
    all_diffs: List[float] = []

    for rec in data:
        gt = _to_float(rec.get("ground_truth"))
        if gt is None:
            continue

        extracted = rec.get("extracted_core_answer") or {}
        if not isinstance(extracted, dict):
            continue

        for gen_key in gen_keys_sorted:
            pred = _to_float(extracted.get(gen_key))
            if pred is None:
                continue
            gen_to_pairs[gen_key].append((pred, gt))
            all_diffs.append(pred - gt)

    # Drop empty generations
    gen_to_pairs = {k: v for k, v in gen_to_pairs.items() if len(v) > 0}
    return gen_to_pairs, all_diffs


def compute_rmse_iqr(
    pairs: List[Tuple[float, float]],
    iqr_multiplier: float = 1.5,
    verbose: bool = False,
) -> float:
    """
    Compute RMSE after filtering outliers using the IQR rule on |error|.
    """
    arr = np.asarray(pairs, dtype=np.float64)
    preds = arr[:, 0]
    gts = arr[:, 1]

    abs_errors = np.abs(preds - gts)

    # IQR statistics
    q1 = np.percentile(abs_errors, 25)
    q3 = np.percentile(abs_errors, 75)
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr

    mask = abs_errors <= threshold

    if verbose:
        print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"Threshold={threshold:.2f}")
        print(f"Retained: {np.sum(mask)}/{len(pairs)} ({np.sum(mask)/len(pairs)*100:.1f}%)")

    filtered_preds = preds[mask]
    filtered_gts = gts[mask]

    return float(np.sqrt(np.mean((filtered_preds - filtered_gts) ** 2)))


def compute_rmse(pairs: List[Tuple[float, float]]) -> float:
    """Default RMSE used in this script: IQR-filtered RMSE."""
    return compute_rmse_iqr(pairs, iqr_multiplier=1.5, verbose=False)


def write_rmse_statistics(
    *,
    input_file: str,
    output_path: str,
    gen_to_pairs: Dict[str, List[Tuple[float, float]]],
) -> None:
    rmse_lines: List[str] = []
    rmse_lines.append("=" * 80)
    rmse_lines.append("Numeric Task RMSE Summary (per generation)")
    rmse_lines.append("=" * 80)
    rmse_lines.append("")
    rmse_lines.append(f"Input file: {input_file}")
    rmse_lines.append("")

    rmses: List[float] = []
    # Per-generation RMSE
    for gen_key in _sorted_generation_keys(gen_to_pairs.keys()):
        pairs = gen_to_pairs[gen_key]
        rmse = compute_rmse(pairs)
        rmses.append(rmse)
        rmse_lines.append(f"{gen_key}: RMSE={rmse:.6f} (N={len(pairs)})")

    if rmses:
        rmse_mean = float(np.mean(np.asarray(rmses, dtype=np.float64)))
        rmse_lines.append("")
        rmse_lines.append(f"Mean RMSE (across generations): {rmse_mean:.6f}")
    else:
        rmse_lines.append("No valid (prediction, ground_truth) pairs found.")

    # Absolute error statistics across all generations
    all_abs_errors: List[float] = []
    for pairs in gen_to_pairs.values():
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        preds = arr[:, 0]
        gts = arr[:, 1]
        all_abs_errors.extend(np.abs(preds - gts).tolist())

    if all_abs_errors:
        abs_arr = np.asarray(all_abs_errors, dtype=np.float64)
        rmse_lines.append("")
        rmse_lines.append("Absolute Error Statistics (all generations combined)")
        rmse_lines.append("-" * 80)
        rmse_lines.append(f"Count: {len(abs_arr)}")
        rmse_lines.append(f"Min |error|: {np.min(abs_arr):.6f}")
        q1 = np.percentile(abs_arr, 25)
        q2 = np.percentile(abs_arr, 50)
        q3 = np.percentile(abs_arr, 75)
        rmse_lines.append(f"25th percentile |error|: {q1:.6f}")
        rmse_lines.append(f"50th percentile |error| (median): {q2:.6f}")
        rmse_lines.append(f"75th percentile |error|: {q3:.6f}")
        rmse_lines.append(f"Max |error|: {np.max(abs_arr):.6f}")

        # 1.5 IQR-based threshold on |error|
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        num_outside = int(np.sum(abs_arr > threshold))
        frac_outside = num_outside / len(abs_arr) if len(abs_arr) > 0 else 0.0
        rmse_lines.append("")
        rmse_lines.append(
            f"Threshold (1.5 IQR rule on |error|): {threshold:.6f}"
        )
        rmse_lines.append(
            f"Num |error| > threshold: {num_outside} "
            f"({frac_outside*100:.2f}% of all points)"
        )

    rmse_lines.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rmse_lines) + "\n")


def plot_diff_distribution(
    diffs: List[float],
    *,
    output_dir: str,
    base_filename: str,
) -> None:
    if len(diffs) == 0:
        print("No valid diffs found; skipping plot.")
        return

    # Convert to absolute differences and apply IQR-based outlier filtering
    diffs_arr_all = np.asarray(diffs, dtype=np.float64)
    abs_errors = np.abs(diffs_arr_all)
    if len(abs_errors) == 0:
        print("No valid absolute diffs found; skipping plot.")
        return

    q1 = np.percentile(abs_errors, 25)
    q3 = np.percentile(abs_errors, 75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    mask = abs_errors <= threshold
    filtered_abs_errors = abs_errors[mask]

    if len(filtered_abs_errors) == 0:
        print("No data left after IQR filtering; skipping plot.")
        return

    mean_abs = float(np.mean(filtered_abs_errors))
    median_abs = float(np.median(filtered_abs_errors))
    std_abs = float(np.std(filtered_abs_errors))
    min_abs = float(np.min(filtered_abs_errors))
    max_abs = float(np.max(filtered_abs_errors))

    fig, ax = plt.subplots(figsize=(6, 6))

    bp = ax.boxplot(
        filtered_abs_errors,
        vert=True,
        patch_artist=True,
        showfliers=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="skyblue", alpha=0.7)

    ax.set_ylabel("|Prediction - Ground truth|", fontsize=12)
    ax.set_xticks([])
    ax.set_ylim(0.0, 5.0)
    ax.set_title(
        "Absolute Difference Boxplot (IQR-filtered)\n"
        f"N={len(filtered_abs_errors)}, Mean={mean_abs:.4f}, Median={median_abs:.4f}, Std={std_abs:.4f}\n"
        f"Min={min_abs:.4f}, Max={max_abs:.4f}",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.25)

    # Extra annotation box with key statistics
    text = (
        f"Mean |diff|:   {mean_abs:.6f}\n"
        f"Median |diff|: {median_abs:.6f}\n"
        f"Std |diff|:    {std_abs:.6f}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{base_filename}_abs_diff_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved diff distribution plot to: {out_path}")


def process_single_file(input_file: str, output_dir: Optional[str] = None) -> int:
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        os.makedirs(output_dir, exist_ok=True)

    base_filename = Path(input_file).stem

    print(f"\n{'=' * 80}")
    print(f"Processing file: {input_file}")
    print(f"{'=' * 80}")

    data = read_jsonl(input_file)
    print(f"Total records read: {len(data)}")

    gen_to_pairs, diffs = extract_pairs_and_diffs(data)
    if not gen_to_pairs:
        print("Error: No valid generations/pairs found in extracted_core_answer.")
        return 1

    stats_output_path = os.path.join(output_dir, f"{base_filename}_rmse_statistics.txt")
    write_rmse_statistics(
        input_file=input_file,
        output_path=stats_output_path,
        gen_to_pairs=gen_to_pairs,
    )
    print(f"Saved RMSE statistics to: {stats_output_path}")

    plot_diff_distribution(diffs, output_dir=output_dir, base_filename=base_filename)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze RMSE and diff distribution for numeric postprocessed JSONL files."
    )
    parser.add_argument("input_files", type=str, nargs="+", help="Path(s) to JSONL file(s)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for txt/plots (default: same as input file directory)",
    )
    args = parser.parse_args()

    print(f"Total files to process: {len(args.input_files)}")
    errors = 0
    for input_file in args.input_files:
        res = process_single_file(input_file, args.output_dir)
        if res != 0:
            errors += 1

    print(f"\n{'=' * 80}")
    print("All files processed!")
    print(f"Total files: {len(args.input_files)}")
    print(f"Successful: {len(args.input_files) - errors}")
    print(f"Failed: {errors}")
    print(f"{'=' * 80}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

