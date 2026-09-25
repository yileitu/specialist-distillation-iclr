#!/usr/bin/env python3
"""
Analyze think-tag quality and token statistics from postprocessed JSONL files.

For all *postprocessed*.jsonl files under an inference directory, this script
computes merged statistics across subtasks:
1) Complete think-tag ratio:
   text contains exactly one <think> and exactly one </think>, with close tag
   after open tag.
2) Empty-think ratio:
   among complete think-tag samples, ratio whose content between tags is empty
   after stripping whitespace/newlines.
3) Word stats for think content:
   among complete think-tag samples, mean/std of whitespace-split word count
   for content between <think> and </think>.
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm


@dataclass
class ThinkParseResult:
    is_complete: bool
    think_content: Optional[str]


def parse_think_tags(text: str) -> ThinkParseResult:
    """
    Parse think tags from text.

    Complete means:
    - exactly one '<think>'
    - exactly one '</think>'
    - '</think>' appears after '<think>'
    """
    open_tag = "<think>"
    close_tag = "</think>"

    open_count = text.count(open_tag)
    close_count = text.count(close_tag)

    if open_count != 1 or close_count != 1:
        return ThinkParseResult(is_complete=False, think_content=None)

    open_start = text.find(open_tag)
    close_start = text.find(close_tag)
    if open_start < 0 or close_start < 0:
        return ThinkParseResult(is_complete=False, think_content=None)

    content_start = open_start + len(open_tag)
    if close_start < content_start:
        return ThinkParseResult(is_complete=False, think_content=None)

    think_content = text[content_start:close_start]
    return ThinkParseResult(is_complete=True, think_content=think_content)


def iter_generated_texts(record: Dict) -> Iterable[str]:
    """
    Yield all generated text fields from a record.

    Primary format expected by postprocessed inference files:
    {
      "generated": {
         "generation0": "...",
         "generation1": "...",
         ...
      }
    }
    """
    generated = record.get("generated")
    if isinstance(generated, dict):
        for value in generated.values():
            if isinstance(value, str):
                yield value


def find_postprocessed_files(inference_dir: Path) -> List[Path]:
    """Find all *postprocessed*.jsonl files recursively under inference_dir."""
    return sorted(inference_dir.rglob("*postprocessed*.jsonl"))


def count_words(text: str) -> int:
    """Count words by whitespace splitting."""
    return len(text.split())


def analyze_files(jsonl_files: List[Path]) -> Dict:
    """
    Analyze all files and return merged + per-file statistics.
    """
    merged_total_texts = 0
    merged_complete = 0
    merged_complete_empty = 0
    merged_complete_token_counts: List[int] = []

    per_file_stats: List[Dict] = []

    for file_path in tqdm(jsonl_files, desc="Processing files", unit="file"):
        file_total_texts = 0
        file_complete = 0
        file_complete_empty = 0
        file_token_counts: List[int] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(
                tqdm(
                    f,
                    desc=f"  {file_path.parent.name}/{file_path.name}",
                    unit="line",
                    leave=False,
                ),
                start=1,
            ):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: JSON decode failed at {file_path}:{line_idx}")
                    continue

                for text in iter_generated_texts(record):
                    file_total_texts += 1
                    parse_result = parse_think_tags(text)
                    if not parse_result.is_complete:
                        continue

                    file_complete += 1
                    think_content = parse_result.think_content or ""

                    if not think_content.strip():
                        file_complete_empty += 1

                    word_count = count_words(think_content)
                    file_token_counts.append(word_count)

        merged_total_texts += file_total_texts
        merged_complete += file_complete
        merged_complete_empty += file_complete_empty
        merged_complete_token_counts.extend(file_token_counts)

        per_file_stats.append(
            {
                "file_path": str(file_path),
                "total_texts": file_total_texts,
                "complete_think": file_complete,
                "complete_ratio": (file_complete / file_total_texts) if file_total_texts else 0.0,
                "empty_in_complete": file_complete_empty,
                "empty_ratio_in_complete": (
                    file_complete_empty / file_complete if file_complete else 0.0
                ),
                "word_mean_complete": (
                    statistics.mean(file_token_counts) if file_token_counts else 0.0
                ),
                "word_std_complete": (
                    statistics.stdev(file_token_counts) if len(file_token_counts) > 1 else 0.0
                ),
            }
        )

    merged_token_mean = (
        statistics.mean(merged_complete_token_counts) if merged_complete_token_counts else 0.0
    )
    merged_token_std = (
        statistics.stdev(merged_complete_token_counts)
        if len(merged_complete_token_counts) > 1
        else 0.0
    )

    return {
        "num_files": len(jsonl_files),
        "merged": {
            "total_texts": merged_total_texts,
            "complete_think": merged_complete,
            "complete_ratio": (merged_complete / merged_total_texts) if merged_total_texts else 0.0,
            "empty_in_complete": merged_complete_empty,
            "empty_ratio_in_complete": (
                merged_complete_empty / merged_complete if merged_complete else 0.0
            ),
            "think_word_mean_complete": merged_token_mean,
            "think_word_std_complete": merged_token_std,
        },
        "per_file": per_file_stats,
    }


def format_report(
    inference_dir: Path,
    analysis: Dict,
) -> str:
    """Format text report."""
    merged = analysis["merged"]
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("POSTPROCESSED THINK TAG + TOKEN STATISTICS REPORT")
    lines.append("=" * 100)
    lines.append(f"Inference directory: {inference_dir}")
    lines.append("Word counting method: whitespace splitting")
    lines.append(f"Matched *postprocessed*.jsonl files: {analysis['num_files']}")
    lines.append("")
    lines.append("MERGED RESULTS (across all matched files)")
    lines.append("-" * 100)
    lines.append(f"Total generated texts analyzed: {merged['total_texts']:,}")
    lines.append(
        "Complete think-tag count (exactly one <think> and one </think>): "
        f"{merged['complete_think']:,}"
    )
    lines.append(f"Complete think-tag ratio: {merged['complete_ratio'] * 100:.4f}%")
    lines.append(
        "Empty think content count (among complete think-tag texts): "
        f"{merged['empty_in_complete']:,}"
    )
    lines.append(
        "Empty think content ratio (among complete think-tag texts): "
        f"{merged['empty_ratio_in_complete'] * 100:.4f}%"
    )
    lines.append(
        "Think-content word mean (complete think-tag texts only): "
        f"{merged['think_word_mean_complete']:.4f}"
    )
    lines.append(
        "Think-content word std (complete think-tag texts only): "
        f"{merged['think_word_std_complete']:.4f}"
    )
    lines.append("")
    lines.append("PER-FILE RESULTS")
    lines.append("-" * 100)

    for idx, file_stats in enumerate(analysis["per_file"], start=1):
        lines.append(f"[{idx}] {file_stats['file_path']}")
        lines.append(f"  Total generated texts: {file_stats['total_texts']:,}")
        lines.append(
            "  Complete think-tag count / ratio: "
            f"{file_stats['complete_think']:,} / {file_stats['complete_ratio'] * 100:.4f}%"
        )
        lines.append(
            "  Empty-in-complete count / ratio: "
            f"{file_stats['empty_in_complete']:,} / "
            f"{file_stats['empty_ratio_in_complete'] * 100:.4f}%"
        )
        lines.append(
            "  Think-content words mean / std (complete only): "
            f"{file_stats['word_mean_complete']:.4f} / {file_stats['word_std_complete']:.4f}"
        )
        lines.append("")

    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze complete/empty think tags and think-content word stats "
        "from *postprocessed*.jsonl files."
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        required=True,
        help="Root inference directory that contains subtask folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report path. Default: <inference-dir>/postprocessed_think_token_stats.txt",
    )
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir).resolve()
    if not inference_dir.exists() or not inference_dir.is_dir():
        print(f"Error: invalid inference directory: {inference_dir}")
        return 1

    jsonl_files = find_postprocessed_files(inference_dir)
    if not jsonl_files:
        print(f"Error: no *postprocessed*.jsonl files found under {inference_dir}")
        return 1

    print(f"Found {len(jsonl_files)} *postprocessed*.jsonl files.")
    analysis = analyze_files(jsonl_files)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else inference_dir / "postprocessed_thinking_stats_by_token.txt"
    )
    report_text = format_report(inference_dir, analysis)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved to: {output_path}")
    merged = analysis["merged"]
    print("\nMerged results:")
    print(f"  total_texts={merged['total_texts']:,}")
    print(f"  complete_ratio={merged['complete_ratio'] * 100:.4f}%")
    print(f"  empty_ratio_in_complete={merged['empty_ratio_in_complete'] * 100:.4f}%")
    print(f"  think_word_mean_complete={merged['think_word_mean_complete']:.4f}")
    print(f"  think_word_std_complete={merged['think_word_std_complete']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
