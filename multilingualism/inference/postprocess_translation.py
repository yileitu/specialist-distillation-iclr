#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess translation inference results.

Features:
1. Compute spBLEU between extracted_translation and gold_answer for every record.
   - Write spBLEU to the source JSONL, skipping records already scored.
   - Batch pending translation pairs across files for parallel scoring.
2. Select the top K percent of records by spBLEU.
3. Generate ShareGPT JSONL suitable for LLaMA-Factory fine-tuning.

Usage:
  # Process one file.
  python postprocess_translation.py --input_path /path/to/inference_output.jsonl

  # Process every JSONL file in a directory.
  python postprocess_translation.py --input_path /path/to/split_dir/

  # Customize selection and parallelism.
  python postprocess_translation.py --input_path /path/to/dir --top_ratio 0.2 --num_workers 16
"""

import json
import re
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import random


# ---------------------------------------------------------------------------
# Logging tee — duplicate stdout to a file
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Write to both the original stream and a log file."""

    def __init__(self, log_path: Path, stream):
        self._stream = stream
        self._file = open(log_path, "w", encoding="utf-8")

    def write(self, text):
        self._stream.write(text)
        self._file.write(text)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Parallel spBLEU computation
# ---------------------------------------------------------------------------

def _compute_spbleu_chunk(pairs: List[Tuple[str, str]]) -> List[float]:
    """Worker function: one BLEU object per chunk (SPM tokenizer loaded once)."""
    bleu = BLEU(tokenize="flores200", effective_order=True, smooth_method="exp")
    results: List[float] = []
    for hyp, ref in pairs:
        if not hyp.strip() or not ref.strip():
            results.append(0.0)
            continue
        try:
            results.append(round(bleu.sentence_score(hyp, [ref]).score, 4))
        except Exception:
            results.append(0.0)
    return results


def batch_compute_spbleu(
    hypotheses: List[str],
    references: List[str],
    num_workers: int = 1,
) -> List[float]:
    """Batch sentence-level spBLEU across all pairs, parallelised over workers."""
    n = len(hypotheses)
    if n == 0:
        return []

    pairs = list(zip(hypotheses, references))

    if num_workers <= 1 or n < 100:
        return _compute_spbleu_chunk(pairs)

    effective_workers = min(num_workers, n)
    chunk_size = (n + effective_workers - 1) // effective_workers
    chunks = [pairs[i : i + chunk_size] for i in range(0, n, chunk_size)]

    with mp.Pool(processes=effective_workers) as pool:
        chunk_results = list(tqdm(
            pool.imap(_compute_spbleu_chunk, chunks),
            total=len(chunks),
            desc="spBLEU (parallel chunks)",
            unit="chunk",
        ))

    return [score for chunk in chunk_results for score in chunk]


# ---------------------------------------------------------------------------
# <think> tag completeness validation
# ---------------------------------------------------------------------------

_RE_HAS_WORD = re.compile(r"\w", re.UNICODE)


def validate_think_tags(text: str) -> Tuple[bool, str]:
    """
    Validate <think>...</think> tag completeness; fix when possible.

    Rules
    -----
    1. No <think> / </think> at all          → valid, unchanged
    2. Exactly one </think>, real content both before & after it:
       a. One <think> at the very start      → valid, unchanged
       b. Zero <think>                        → valid, prepend "<think> "
    *  Everything else                        → invalid

    Returns (is_valid, possibly_fixed_text).
    """
    stripped = text.strip()
    n_open = stripped.count("<think>")
    n_close = stripped.count("</think>")

    if n_open == 0 and n_close == 0:
        return True, text
        # return False, text

    if n_close != 1:
        return False, text

    close_pos = stripped.index("</think>")
    raw_before = stripped[:close_pos]
    raw_after = stripped[close_pos + len("</think>"):]

    thinking_content = raw_before.replace("<think>", "")
    has_real_thinking = bool(_RE_HAS_WORD.search(thinking_content))
    has_real_translation = bool(_RE_HAS_WORD.search(raw_after))

    if not (has_real_thinking and has_real_translation):
        return False, text

    if n_open == 1 and stripped.startswith("<think>"):
        return True, text

    if n_open == 0:
        return True, "<think>\n" + stripped

    return False, text


def apply_think_tag_filter(
    file_records: List[Tuple[Path, List[Dict[str, Any]]]],
) -> Tuple[Dict[str, int], Set[int]]:
    """
    Validate / fix <think> tags in every generation of every record.
    Recompute best_generation_key from valid generations only.

    Returns (stats_dict, set_of_modified_file_indices).
    """
    stats = {
        "total_gens": 0,
        "ok": 0,
        "fixed": 0,
        "invalid": 0,
        "records_all_invalid": 0,
    }
    modified_files: Set[int] = set()

    for fidx, (_fpath, records) in enumerate(file_records):
        for rec in records:
            generated = rec.get("generated", {})
            spbleu_dict = rec.get("spbleu", {})
            if not generated:
                continue

            valid_scores: Dict[str, float] = {}

            for gen_key, gen_text in generated.items():
                stats["total_gens"] += 1
                is_valid, fixed_text = validate_think_tags(gen_text)

                if is_valid:
                    if fixed_text != gen_text:
                        generated[gen_key] = fixed_text
                        modified_files.add(fidx)
                        stats["fixed"] += 1
                    else:
                        stats["ok"] += 1
                    if gen_key in spbleu_dict:
                        valid_scores[gen_key] = spbleu_dict[gen_key]
                else:
                    stats["invalid"] += 1

            if valid_scores:
                best_key = max(valid_scores, key=valid_scores.get)
                if rec.get("best_generation_key") != best_key:
                    modified_files.add(fidx)
                rec["best_spbleu"] = valid_scores[best_key]
                rec["best_generation_key"] = best_key
                rec["think_tag_valid"] = True
            else:
                if rec.get("think_tag_valid") is not False:
                    modified_files.add(fidx)
                rec["best_spbleu"] = -1.0
                rec["best_generation_key"] = None
                rec["think_tag_valid"] = False
                stats["records_all_invalid"] += 1

    return stats, modified_files


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl_atomic(filepath: Path, records: List[Dict[str, Any]]):
    """Write atomically to prevent partial files"""
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(filepath)


def process_input_files(input_path: Path) -> List[Path]:
    """Return the JSONL files to process"""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        # files = sorted(input_path.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in directory: {input_path}")
        return files
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


# ---------------------------------------------------------------------------
# Core: batch augment spBLEU across ALL loaded files at once
# ---------------------------------------------------------------------------

def augment_spbleu_batch(
    file_records: List[Tuple[Path, List[Dict[str, Any]]]],
    num_workers: int = 1,
) -> Tuple[int, Set[int]]:
    """
    Collect every pending (hyp, ref) pair across all files, compute in one
    parallel batch, scatter scores back to the original records.

    Returns (newly_scored_record_count, set_of_modified_file_indices).
    """
    pair_map: List[Tuple[int, int, str]] = []   # (file_idx, rec_idx, gen_key)
    all_hyps: List[str] = []
    all_refs: List[str] = []
    empty_pending: List[Tuple[int, int]] = []   # pending records with no translations
    modified_files: Set[int] = set()

    for fidx, (_fpath, records) in enumerate(file_records):
        for ridx, rec in enumerate(records):
            if "spbleu" in rec:
                continue
            modified_files.add(fidx)
            gold = rec.get("gold_answer", "")
            extracted = rec.get("extracted_translation", {})
            if not extracted:
                empty_pending.append((fidx, ridx))
                continue
            for gen_key, translation in extracted.items():
                pair_map.append((fidx, ridx, gen_key))
                all_hyps.append(translation)
                all_refs.append(gold)

    for fidx, ridx in empty_pending:
        rec = file_records[fidx][1][ridx]
        rec["spbleu"] = {}
        rec["best_spbleu"] = 0.0
        rec["best_generation_key"] = None

    if not all_hyps:
        return len(empty_pending), modified_files

    pending_recs = len(set((f, r) for f, r, _ in pair_map)) + len(empty_pending)
    print(f"Pending: {pending_recs} records, {len(all_hyps)} translation pairs, "
          f"using {min(num_workers, len(all_hyps))} processes")

    scores = batch_compute_spbleu(all_hyps, all_refs, num_workers=num_workers)

    rec_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (fidx, ridx, gen_key), score in zip(pair_map, scores):
        rec_scores.setdefault((fidx, ridx), {})[gen_key] = score

    for (fidx, ridx), spbleu_dict in rec_scores.items():
        rec = file_records[fidx][1][ridx]
        rec["spbleu"] = spbleu_dict
        best_key = max(spbleu_dict, key=spbleu_dict.get)
        rec["best_spbleu"] = spbleu_dict[best_key]
        rec["best_generation_key"] = best_key

    return pending_recs, modified_files


# ---------------------------------------------------------------------------
# ShareGPT formatting
# ---------------------------------------------------------------------------

def build_sharegpt_record(
    rec: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one ShareGPT record compatible with LLaMA-Factory.
    """
    best_key = rec.get("best_generation_key", "generation1")
    generated = rec.get("generated", {})
    assistant_content = generated.get(best_key, "")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rec["user_content"]})
    messages.append({"role": "assistant", "content": assistant_content})

    return {
        "messages": messages,
        "sample_id": rec.get("sample_id", ""),
        "gold_answer": rec.get("gold_answer", ""),
        "best_spbleu": rec.get("best_spbleu", 0.0),
        "best_generation_key": best_key,
        "spbleu": rec.get("spbleu", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess inference: compute spBLEU, select top K%, and create ShareGPT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Inference result file or directory containing split JSONL files",
    )
    parser.add_argument(
        "--top_ratio", type=float, default=0.2,
        help="Top spBLEU ratio to retain (default: 0.2, or 20%%)",
    )
    parser.add_argument(
        "--output_ft", type=str, default=None,
        help="Output ShareGPT path (default: next to input_path)",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Optional system prompt to include in ShareGPT messages",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Worker processes for spBLEU (default: CPU count)",
    )
    parser.add_argument(
        "--skip_think_tag_validation",
        action="store_true",
        help="Skip <think> tag validation and repair",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)
    top_ratio = args.top_ratio
    num_workers = args.num_workers or mp.cpu_count()

    # Store the fine-tuning data and log together next to the input.
    if args.output_ft:
        output_ft_path = Path(args.output_ft)
    else:
        if input_path.is_dir():
            output_dir = input_path.parent
        else:
            output_dir = input_path.parent
        output_ft_path = output_dir / f"ft_sharegpt_top{int(top_ratio * 100)}pct.jsonl"

    output_ft_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_ft_path.with_suffix(".log.txt")

    tee = _TeeWriter(log_path, sys.stdout)
    sys.stdout = tee

    try:
        _run_pipeline(args, input_path, top_ratio, num_workers, output_ft_path)
    finally:
        sys.stdout = tee._stream
        tee.close()
        print(f"Log saved to: {log_path}")


def _run_pipeline(
    args,
    input_path: Path,
    top_ratio: float,
    num_workers: int,
    output_ft_path: Path,
):
    # 1. Discover JSONL files.
    jsonl_files = process_input_files(input_path)
    print(f"Found {len(jsonl_files)} JSONL files")

    # 2. Load all files.
    file_records: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for fpath in jsonl_files:
        records = load_jsonl(fpath)
        already = sum(1 for r in records if "spbleu" in r)
        print(f"  {fpath.name}: {len(records)} records, existing spBLEU: {already}")
        file_records.append((fpath, records))

    # 3. Compute spBLEU across files in parallel.
    newly, modified_files = augment_spbleu_batch(file_records, num_workers=num_workers)

    # 4. Validate and repair <think> tags.
    think_modified: Set[int] = set()
    if args.skip_think_tag_validation:
        print("\n<think> tag validation skipped; all records accepted")
    else:
        think_stats, think_modified = apply_think_tag_filter(file_records)
        modified_files |= think_modified
        print(f"\n<think> tag filtering:")
        print(f"  total generations: {think_stats['total_gens']}, "
              f"valid: {think_stats['ok']}(original) + {think_stats['fixed']}(repaired), "
              f"invalid: {think_stats['invalid']}")
        if think_stats["records_all_invalid"] > 0:
            print(f"  records with no valid generations: {think_stats['records_all_invalid']} (excluded)")

    # 5. Write modified source files.
    if modified_files:
        for fidx in modified_files:
            fpath, records = file_records[fidx]
            save_jsonl_atomic(fpath, records)
        parts = []
        if newly > 0:
            parts.append(f"new spBLEU scores: {newly} records")
        if think_modified:
            parts.append(f"think-tag repairs/updates")
        print(f"Updated {len(modified_files)} files ({', '.join(parts)})")
    else:
        print("No changes required")

    # 6. Summarize records with valid think tags.
    all_records = [rec for _, records in file_records for rec in records]
    total = len(all_records)
    if args.skip_think_tag_validation:
        valid_records = all_records
    else:
        valid_records = [r for r in all_records if r.get("think_tag_valid", True)]
    excluded = total - len(valid_records)
    print(f"\n{'=' * 60}")
    print(f"All files processed. Total records: {total}")
    if excluded > 0:
        print(f"  invalid think tags excluded: {excluded}, valid records: {len(valid_records)}")

    scores = [r.get("best_spbleu", 0.0) for r in valid_records]
    if scores:
        scores_sorted = sorted(scores)
        avg = sum(scores) / len(scores)
        median = scores_sorted[len(scores_sorted) // 2]
        print(f"spBLEU statistics (valid records): avg={avg:.2f}, median={median:.2f}, "
              f"min={scores_sorted[0]:.2f}, max={scores_sorted[-1]:.2f}")

    # 7. Select the top K% by spBLEU from valid records.
    n_valid = len(valid_records)
    top_k = max(1, int(total * top_ratio))
    top_k = min(top_k, n_valid)
    valid_sorted = sorted(
        valid_records, key=lambda r: r.get("best_spbleu", 0.0), reverse=True,
    )
    top_records = valid_sorted[:top_k]

    threshold = top_records[-1].get("best_spbleu", 0.0) if top_records else 0.0
    print(f"\nTop {top_ratio * 100:.0f}% selection: {top_k}/{total} records "
          f"(from {n_valid} valid records)")
    print(f"  spBLEU threshold: >= {threshold:.2f}")
    top_scores = [r.get("best_spbleu", 0.0) for r in top_records]
    if top_scores:
        print(f"  top subset avg={sum(top_scores) / len(top_scores):.2f}, "
              f"min={min(top_scores):.2f}, max={max(top_scores):.2f}")

    # 8. Generate the ShareGPT dataset.
    ft_records = [
        build_sharegpt_record(rec, system_prompt=args.system_prompt)
        for rec in top_records
    ]
    # Shuffle the ft data
    print(f"Shuffling the ft data...")
    random.seed(42)
    random.shuffle(ft_records)
    
    with open(output_ft_path, "w", encoding="utf-8") as f:
        for ft_rec in ft_records:
            f.write(json.dumps(ft_rec, ensure_ascii=False) + "\n")

    print(f"\nShareGPT fine-tuning data saved to: {output_ft_path}")
    print(f"  Records: {len(ft_records)}")

    print(f"\nExamples (first three):")
    for i, ft_rec in enumerate(ft_records[:3]):
        user_msg = ft_rec["messages"][-2]["content"]
        asst_msg = ft_rec["messages"][-1]["content"]
        print(f"  --- Example {i + 1} (spBLEU={ft_rec['best_spbleu']:.2f}) ---")
        print(f"  user: {user_msg[:120]}...")
        print(f"  assistant: {asst_msg[:120]}...")
        print(f"  gold: {ft_rec['gold_answer'][:120]}...")

    print(f"\n{'=' * 60}")
    print("Postprocessing complete!")


if __name__ == "__main__":
    main()
