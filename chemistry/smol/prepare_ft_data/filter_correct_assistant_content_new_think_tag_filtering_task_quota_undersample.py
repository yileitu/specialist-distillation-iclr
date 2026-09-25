#!/usr/bin/env python3
"""
Prepare fine-tuning data with think-tag filtering and per-task quota undersampling.

This script extends the logic in:
  filter_correct_assistant_content_new_think_tag_filtering.py

Pipeline per task:
1. Keep samples with at least one correct generation.
2. Keep only generations that satisfy think-tag format constraints.
3. Randomly pick one valid generation per sample.
4. Convert to ShareGPT format.
5. Undersample each task to a fixed target quota (keep all if fewer than target).
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import ALL_TASKS


# Target quota copied from:
# smol/ft_data/version20260313_upsampled_50k_new_filtration_M0_Qwen3_8_13B_M1M2_Qwen3_8B/aligned_data/aligned_ft_data_statistics.txt
TARGET_SAMPLES_PER_TASK: Dict[str, int] = {
    "forward_synthesis": 1349,
    "molecule_captioning": 6,
    "molecule_generation": 908,
    "name_conversion-i2f": 223,
    "name_conversion-i2s": 46,
    "name_conversion-s2f": 2906,
    "name_conversion-s2i": 10,
    "property_prediction-bbbp": 3344,
    "property_prediction-clintox": 2182,
    "property_prediction-esol": 2350,
    "property_prediction-hiv": 5498,
    "property_prediction-lipo": 9614,
    "property_prediction-sider": 23416,
    "retrosynthesis": 105,
}



def find_postprocessed_jsonl(task_dir: Path) -> Optional[Path]:
    """Find the postprocessed JSONL file in a task directory."""
    patterns = [
        "*_postprocessed*.jsonl",
        "*_bool_postprocessed.jsonl",
        "*_num_postprocessed*.jsonl",
        "*_smiles_postprocessed.jsonl",
        "*_postprocessed_lev*.jsonl",
        "*_postprocessed_METEOR*.jsonl",
    ]
    for pattern in patterns:
        matches = list(task_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def extract_text_from_prompt(prompt: List[Dict[str, Any]]) -> str:
    """Extract user text content from prompt format."""
    text_parts: List[str] = []
    for item in prompt:
        if item.get("role") != "user":
            continue
        content = item.get("content", [])
        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_parts.append(content_item.get("text", ""))
        elif isinstance(content, str):
            text_parts.append(content)
    return "".join(text_parts)


def _has_substantive_content(text: str) -> bool:
    """Check if text contains substantive content (not only whitespace/punctuation)."""
    return bool(re.search(r"\w", text))


def convert_to_sharegpt_format(
    prompt: List[Dict[str, Any]],
    generated_text: str,
    task: str,
    sample_id: str,
    ground_truth: str,
    output_core_tag_left: str,
    output_core_tag_right: str,
) -> Dict[str, Any]:
    """Convert one sample to ShareGPT format with metadata."""
    user_content = extract_text_from_prompt(prompt)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": generated_text},
    ]
    return {
        "messages": messages,
        "task": task,
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "output_core_tag_left": output_core_tag_left,
        "output_core_tag_right": output_core_tag_right,
    }


def collect_filtered_samples_from_jsonl(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Run original filtering logic and return all kept samples before quota undersampling.
    """
    stats = {
        "total": 0,
        "filtered": 0,
        "correct_generations": 0,
        "valid_after_think_filter": 0,
    }
    collected_samples: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())

    with open(jsonl_path, "r", encoding="utf-8") as f_in:
        pbar = tqdm(
            total=total_lines,
            desc=f"  Filtering {jsonl_path.name}",
            unit="lines",
            leave=False,
        )

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            pbar.update(1)
            data = json.loads(line)

            correctness = data.get("extracted_core_answer_correctness", {})
            if not correctness:
                continue

            true_generations = [
                gen_key for gen_key, is_correct in correctness.items() if is_correct is True
            ]
            if not true_generations:
                continue

            stats["filtered"] += 1
            stats["correct_generations"] += len(true_generations)

            generated = data.get("generated", {})

            valid_generations: List[Tuple[str, str]] = []
            for gen_key in true_generations:
                gen_text = generated.get(gen_key, "")
                if not gen_text:
                    continue

                think_open_count = gen_text.count("<think>")
                think_close_count = gen_text.count("</think>")

                # Case 1: one <think> and one </think>, with valid content between/after tags.
                if think_open_count == 1 and think_close_count == 1:
                    if not gen_text.lstrip().startswith("<think>"):
                        continue
                    open_idx = gen_text.index("<think>")
                    close_idx = gen_text.index("</think>")
                    if close_idx <= open_idx:
                        continue
                    between = gen_text[open_idx + len("<think>") : close_idx]
                    after = gen_text[close_idx + len("</think>") :]
                    if _has_substantive_content(between) and _has_substantive_content(after):
                        valid_generations.append((gen_key, "case1"))

                # Case 2: no <think>, one </think>, with valid content before/after </think>.
                elif think_open_count == 0 and think_close_count == 1:
                    close_idx = gen_text.index("</think>")
                    before = gen_text[:close_idx]
                    after = gen_text[close_idx + len("</think>") :]
                    if _has_substantive_content(before) and _has_substantive_content(after):
                        valid_generations.append((gen_key, "case2"))

            if not valid_generations:
                continue

            selected_gen, case_type = random.choice(valid_generations)
            generated_text = generated.get(selected_gen, "")
            if not generated_text:
                continue

            if case_type == "case2":
                generated_text = "<think> " + generated_text

            prompt = data.get("prompt", [])
            if not prompt:
                continue

            sharegpt_data = convert_to_sharegpt_format(
                prompt=prompt,
                generated_text=generated_text,
                task=data.get("task", ""),
                sample_id=data.get("sample_id", ""),
                ground_truth=data.get("ground_truth", ""),
                output_core_tag_left=data.get("output_core_tag_left", ""),
                output_core_tag_right=data.get("output_core_tag_right", ""),
            )
            collected_samples.append(sharegpt_data)

        pbar.close()

    stats["valid_after_think_filter"] = len(collected_samples)
    return collected_samples, stats


def undersample_to_quota(samples: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Randomly undersample to target_count if needed; otherwise keep all."""
    if len(samples) <= target_count:
        return samples
    return random.sample(samples, target_count)


def write_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    """Write samples to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_inference_folder(inference_folder: Path, output_base: Path) -> Dict[str, Any]:
    """Process one inference folder with task-level quota undersampling."""
    folder_parts = inference_folder.parts
    if len(folder_parts) >= 2:
        model_name = folder_parts[-2]
        param_folder = folder_parts[-1]
        inference_mode = f"{model_name}_{param_folder}"
    else:
        inference_mode = inference_folder.name
    inference_mode = inference_mode.replace("/", "_").replace("\\", "_")
    output_dir = output_base / f"{inference_mode}_task_quota_undersampled"

    print("\n" + "=" * 70)
    print(f"Processing inference folder: {inference_folder}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    available_tasks: List[Tuple[str, Path]] = []
    for task_name in ALL_TASKS:
        task_dir = inference_folder / task_name
        if not (task_dir.exists() and task_dir.is_dir()):
            continue
        jsonl_file = find_postprocessed_jsonl(task_dir)
        if jsonl_file is not None:
            available_tasks.append((task_name, jsonl_file))

    print(f"Found {len(available_tasks)} tasks with postprocessed files.\n")

    all_stats: Dict[str, Dict[str, int]] = {}
    total_stats = {
        "total": 0,
        "filtered": 0,
        "correct_generations": 0,
        "valid_after_think_filter": 0,
        "saved_after_quota": 0,
    }

    for task_name, jsonl_file in tqdm(available_tasks, desc="Processing tasks", unit="task"):
        print(f"\nTask {task_name}: {jsonl_file.name}")
        filtered_samples, task_stats = collect_filtered_samples_from_jsonl(jsonl_file)
        before_quota = len(filtered_samples)

        target = TARGET_SAMPLES_PER_TASK.get(task_name, before_quota)
        final_samples = undersample_to_quota(filtered_samples, target)
        after_quota = len(final_samples)
        dropped_by_quota = before_quota - after_quota

        output_file = output_dir / task_name / "ft_data.jsonl"
        write_jsonl(final_samples, output_file)

        task_stats["target_quota"] = target
        task_stats["saved_after_quota"] = after_quota
        task_stats["dropped_by_quota"] = dropped_by_quota
        all_stats[task_name] = task_stats

        total_stats["total"] += task_stats["total"]
        total_stats["filtered"] += task_stats["filtered"]
        total_stats["correct_generations"] += task_stats["correct_generations"]
        total_stats["valid_after_think_filter"] += task_stats["valid_after_think_filter"]
        total_stats["saved_after_quota"] += task_stats["saved_after_quota"]

        print(
            "  Total: {total}, Valid after think-filter: {valid}, Target: {target}, Saved: {saved}, Dropped by quota: {dropped}".format(
                total=task_stats["total"],
                valid=before_quota,
                target=target,
                saved=after_quota,
                dropped=dropped_by_quota,
            )
        )

    stats_file = output_dir / "statistics.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Fine-tuning Data Statistics with Task Quota Undersampling\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source folder: {inference_folder.resolve()}\n")
        f.write("\n")
        f.write("Target quota per task:\n")
        f.write("-" * 70 + "\n")
        for task_name in ALL_TASKS:
            if task_name in TARGET_SAMPLES_PER_TASK:
                f.write(f"  {task_name}: {TARGET_SAMPLES_PER_TASK[task_name]}\n")
        f.write("\n")
        f.write("Per-task Statistics:\n")
        f.write("-" * 70 + "\n")

        for task_name in ALL_TASKS:
            if task_name not in all_stats:
                continue
            s = all_stats[task_name]
            f.write(f"\n{task_name}:\n")
            f.write(f"  Total samples: {s['total']}\n")
            f.write(f"  Filtered by correctness: {s['filtered']}\n")
            f.write(f"  Total correct generations: {s['correct_generations']}\n")
            f.write(f"  Valid after think-tag filter: {s['valid_after_think_filter']}\n")
            f.write(f"  Target quota: {s['target_quota']}\n")
            f.write(f"  Saved after quota: {s['saved_after_quota']}\n")
            f.write(f"  Dropped by quota: {s['dropped_by_quota']}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Overall Summary:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples: {total_stats['total']}\n")
        f.write(f"Filtered by correctness: {total_stats['filtered']}\n")
        f.write(f"Total correct generations: {total_stats['correct_generations']}\n")
        f.write(f"Valid after think-tag filter: {total_stats['valid_after_think_filter']}\n")
        f.write(f"Saved after quota: {total_stats['saved_after_quota']}\n")
        f.write("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total samples: {total_stats['total']}")
    print(f"  Valid after think-tag filter: {total_stats['valid_after_think_filter']}")
    print(f"  Saved after quota: {total_stats['saved_after_quota']}")
    print(f"  Statistics file: {stats_file}")
    print("=" * 70)

    return {
        "inference_mode": inference_mode,
        "stats": all_stats,
        "total_stats": total_stats,
        "output_dir": str(output_dir),
        "stats_file": str(stats_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter correct assistant content with think-tag constraints, then "
            "undersample each task to fixed target counts."
        )
    )
    parser.add_argument(
        "--inference-folder",
        type=str,
        required=True,
        help="Path to the inference folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: smol/ft_data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation selection and quota undersampling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    inference_folder = Path(args.inference_folder)
    if not inference_folder.exists():
        raise FileNotFoundError(f"Inference folder does not exist: {inference_folder}")
    if not inference_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {inference_folder}")

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "ft_data"
    else:
        output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fine-tuning Data Preparation with Task Quota Undersampling")
    print("=" * 70)
    print(f"Inference folder: {inference_folder}")
    print(f"Output base directory: {output_base}")
    print(f"Random seed: {args.seed}")
    print(f"Quota tasks configured: {len(TARGET_SAMPLES_PER_TASK)}")
    print("=" * 70)

    result = process_inference_folder(inference_folder, output_base)

    print("\nDone.")
    print(f"Output directory: {result['output_dir']}")
    print(f"Statistics file: {result['stats_file']}")


if __name__ == "__main__":
    main()
