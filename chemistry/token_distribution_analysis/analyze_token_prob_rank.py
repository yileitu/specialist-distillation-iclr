#!/usr/bin/env python3
"""
Token-level probability and rank analysis for model outputs.

For each subtask JSONL in the data directory:
1. Convert each record to ShareGPT-format prompt (system + user, excluding assistant)
2. Use teacher forcing (single forward pass) to get next-token probability distributions
3. Record probability and rank of each ground-truth output token
4. Average per sample, then per subtask
5. Save per-subtask results and cross-subtask summary
6. Save raw per-sample (and per-token) data to {output_dir}_raw/ for later analysis
7. Report std, skewness, and kurtosis alongside mean
"""

import json
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NOTHINK_SYSTEM_PROMPT = (
    "You are an expert with extensive knowledge in all areas. "
    "Provide direct answers to the user's question without showing your reasoning process. "
    "Do not use <think> and </think> tags."
)

SMOL_TASK_INSTRUCTION = {
    "forward_synthesis": (
        "You are an expert chemist. Given the SMILES representation of reactants and reagents, "
        "your task is to predict the potential product using your chemical reaction knowledge.\n"
        "    The input contains both reactants and reagents, and different reactants and reagents "
        'are separated by ".". Your reply should contain the SMILES representation of the predicted '
        "product wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically "
        "reasonable.\nQuestion: "
    ),
    "retrosynthesis": (
        "You are an expert chemist. Given the SMILES representation of the product, your task is "
        "to predict the potential reactants and reagents using your chemical reaction knowledge.\n"
        "    The input contains the SMILES representation of the product. Your reply should contain "
        "the SMILES representation of both reactants and reagents, and all reactants and reagents "
        "should be enclosed **together** within a single pair of <SMILES> and </SMILES> tags, "
        'separated by ".". Your reply must be valid and chemically reasonable.\nQuestion: '
    ),
    "molecule_captioning": (
        "You are an expert chemist. Given the SMILES representation of a molecule, your task is "
        "to describe the molecule in natural language.\n"
        "    The input contains the SMILES representation of the molecule. Your reply should contain "
        "a natural language description of the molecule. Your reply must be valid and chemically "
        "reasonable.\nQuestion: "
    ),
    "molecule_generation": (
        "You are an expert chemist. Given the description of a molecule, your task is to generate "
        "the potential SMILES representation of the molecule.\n"
        "    The input contains the description of the molecule. Your reply should contain the "
        "potential SMILES representation of the molecule wrapped in <SMILES> and </SMILES> tags. "
        "Your reply must be valid and chemically reasonable.\nQuestion: "
    ),
    "name_conversion-i2f": (
        "You are an expert chemist. Given the IUPAC representation of compounds, your task is to "
        "predict the molecular formula of the compound.\n"
        "    The input contains the IUPAC representation of the compound. Your reply should contain "
        "only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags "
        "and no other text. Your reply must be valid and chemically reasonable.\nQuestion: "
    ),
    "name_conversion-i2s": (
        "You are an expert chemist. Given the IUPAC representation of compounds, your task is to "
        "predict the SMILES representation of the compound.\n"
        "    The input contains the IUPAC representation of the compound. Your reply should contain "
        "only the SMILES representation of the compound wrapped in <SMILES> and </SMILES> tags and "
        "no other text. Your reply must be valid and chemically reasonable.\nQuestion: "
    ),
    "name_conversion-s2f": (
        "You are an expert chemist. Given the SMILES representation of compounds, your task is to "
        "predict the molecular formula of the compound.\n"
        "    The input contains the SMILES representation of the compound. Your reply should contain "
        "only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags "
        "and no other text. Your reply must be valid and chemically reasonable.\nQuestion: "
    ),
    "name_conversion-s2i": (
        "You are an expert chemist. Given the SMILES representation of compounds, your task is to "
        "predict the IUPAC representation of the compound.\n"
        "    The input contains the SMILES representation of the compound. Your reply should contain "
        "only the IUPAC representation of the compound wrapped in <IUPAC> and </IUPAC> tags and no "
        "other text. Your reply must be valid and chemically reasonable.\nQuestion: "
    ),
    "property_prediction-esol": (
        "You are an expert chemist. Given the SMILES representation of compounds, your task is to "
        "predict the log solubility of the compound.\n"
        "    The input contains the SMILES representation of the compound. Your reply should contain "
        "the log solubility of the compound wrapped in \\boxed{}. Your reply must be valid and "
        "chemically reasonable.\nQuestion: "
    ),
    "property_prediction-lipo": (
        "You are an expert chemist. Given the SMILES representation of compounds, your task is to "
        "predict the octanol/water partition coefficient of the compound.\n"
        "    The input contains the SMILES representation of the compound. Your reply should contain "
        "the octanol/water partition coefficient of the compound wrapped in \\boxed{}. Your reply "
        "must be valid and chemically reasonable.\nQuestion: "
    ),
    "property_prediction-bbbp": (
        "You are an expert chemist. Given the smiles representation of the compound, your task is "
        "to predict whether blood-brain barrier permeability (BBBP) is a property of the compound.\n"
        "    The input contains the compound. Your reply should only contain Yes or No. Your reply "
        "must be valid and chemically reasonable.\nQuestion: "
    ),
    "property_prediction-clintox": (
        "You are an expert chemist. Given the smiles representation of the compound, your task is "
        "to predict whether the compound is toxic.\n"
        "    The input contains the compound. Your reply should contain only Yes or No. Your reply "
        "must be valid and chemically reasonable.\nQuestion: "
    ),
    "property_prediction-hiv": (
        "You are an expert chemist. Given the smiles representation of the compound, your task is "
        "to predict whether the compound serve as an inhibitor of HIV replication.\n"
        "    The input contains the compound. Your reply should contain only Yes or No. Your reply "
        "must be valid and chemically reasonable.\nQuestion: "
    ),
    "property_prediction-sider": (
        "You are an expert chemist. Given the smiles representation of the compound, your task is "
        "to predict whether the compound has any side effects.\n"
        "    The input contains the compound. Your reply should contain only Yes or No. Your reply "
        "must be valid and chemically reasonable.\nQuestion: "
    ),
}


def compute_distribution_stats(values: list) -> Dict[str, float]:
    """Compute mean, std (ddof=1), skewness, and excess kurtosis.

    Uses the adjusted Fisher-Pearson formulae (unbiased estimators),
    consistent with scipy.stats.skew(bias=False) and
    scipy.stats.kurtosis(bias=False, fisher=True).
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    if n < 2:
        return {"mean": mean_val, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    std_val = float(np.std(arr, ddof=1))
    if std_val < 1e-15:
        return {"mean": mean_val, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    z = (arr - mean_val) / std_val
    s3 = float(np.sum(z ** 3))
    s4 = float(np.sum(z ** 4))
    skewness = s3 * n / ((n - 1) * (n - 2)) if n > 2 else 0.0
    if n > 3:
        kurtosis = (
            n * (n + 1) * s4 / ((n - 1) * (n - 2) * (n - 3))
            - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
    else:
        kurtosis = 0.0
    return {"mean": mean_val, "std": std_val, "skewness": skewness, "kurtosis": kurtosis}


def build_prompt_messages(record: Dict[str, Any], task: str):
    """Build ShareGPT prompt messages (system + user), excluding assistant."""
    task_instruction = SMOL_TASK_INSTRUCTION.get(task, "")
    user_content = task_instruction + record["input"] + "\nAnswer: "
    return [
        {"role": "system", "content": NOTHINK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compute_trailing_special_count(tokenizer, enable_thinking: bool = False) -> int:
# def compute_trailing_special_count(tokenizer) -> int:
    """
    Determine how many trailing special tokens the chat template appends
    after the assistant content (e.g. <|im_end|>).
    """
    test_prompt = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    test_full = test_prompt + [{"role": "assistant", "content": ""}]

    prompt_ids = tokenizer.apply_chat_template(
        test_prompt, 
        add_generation_prompt=True, 
        tokenize=True, 
        enable_thinking=enable_thinking # Setting enable_thinking=False disables thinking mode
    )
    full_ids = tokenizer.apply_chat_template(
        test_full, 
        tokenize=True, 
        enable_thinking=enable_thinking
        )
    return len(full_ids) - len(prompt_ids)


def analyze_sample(
    model,
    tokenizer,
    prompt_messages,
    raw_output: str,
    n_trailing: int,
    max_seq_len: Optional[int] = None,
    enable_thinking: bool = False,
) -> Optional[
    Tuple[
        float, float, float, int, List[float], List[int], List[float], List[str]
    ]
]:
    """
    Single forward pass (teacher forcing) to compute per-token probability and
    rank for the ground-truth output tokens.

    Because the model is causal (uses a causal attention mask), each position's
    logits only depend on tokens at or before that position. This means a single
    forward pass over [q, a0, a1, ..., a_{n-1}] is equivalent to running n
    separate forward passes:
      - input [q]             -> logits predicting a0
      - input [q, a0]         -> logits predicting a1
      - input [q, a0, a1]     -> logits predicting a2
      - ...
    We simply read off logits[prompt_len-1+i] to get the prediction for a_i,
    avoiding the O(n) forward passes that a naive loop would require.

    Returns (
        avg_prob, avg_rank, avg_top1_hit_rate, n_content_tokens,
        token_probs, token_ranks, token_top1_hits, token_strs
    ) or None if the sample is skipped.
    """
    full_messages = prompt_messages + [
        {"role": "assistant", "content": raw_output}
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, 
        enable_thinking=enable_thinking
        )
    full_ids = tokenizer.apply_chat_template(
        full_messages, 
        tokenize=True, 
        enable_thinking=enable_thinking
        )

    prompt_len = len(prompt_ids)
    n_content = len(full_ids) - prompt_len - n_trailing

    if n_content <= 0:
        return None

    end_idx = prompt_len + n_content
    if max_seq_len and end_idx > max_seq_len:
        end_idx = max_seq_len
        n_content = end_idx - prompt_len
        if n_content <= 0:
            return None

    input_tensor = torch.tensor(
        [full_ids[:end_idx]], dtype=torch.long, device=model.device
    )

    with torch.no_grad():
        logits = model(input_tensor).logits[0]  # (seq_len, vocab_size)

    # logits[prompt_len - 1 + i] predicts the token at position prompt_len + i
    output_logits = logits[prompt_len - 1 : end_idx - 1]  # (n_content, vocab_size)
    target_ids = torch.tensor(
        full_ids[prompt_len:end_idx], dtype=torch.long, device=model.device
    )

    probs = torch.softmax(output_logits.float(), dim=-1)
    target_probs = probs[torch.arange(n_content, device=model.device), target_ids]
    ranks = (probs > target_probs.unsqueeze(-1)).sum(dim=-1) + 1
    top1_hits = (ranks == 1).float()

    avg_prob = target_probs.mean().item()
    avg_rank = ranks.float().mean().item()
    avg_top1_hit_rate = top1_hits.mean().item()
    token_probs_list = target_probs.cpu().tolist()
    token_ranks_list = ranks.cpu().tolist()
    token_top1_hits_list = top1_hits.cpu().tolist()
    token_strs = [tokenizer.decode([tid]) for tid in full_ids[prompt_len:end_idx]]

    del input_tensor, logits, output_logits, probs, target_probs, ranks, top1_hits
    return (
        avg_prob,
        avg_rank,
        avg_top1_hit_rate,
        n_content,
        token_probs_list,
        token_ranks_list,
        token_top1_hits_list,
        token_strs,
    )


def _fmt_stats(s: Dict[str, float], val_fmt: str = ".6f") -> str:
    """Format a stats dict into a compact one-line string."""
    return (
        f"Std: {s['std']:{val_fmt}}  "
        f"Skewness: {s['skewness']:.4f}  "
        f"Kurtosis: {s['kurtosis']:.4f}"
    )


def _leading_think_prefix_len(token_strs: List[str]) -> int:
    """Return 4 when token_strs starts with <think> \n\n </think> \n\n, else 0."""
    think_prefix = ["<think>", "\n\n", "</think>", "\n\n"]
    if len(token_strs) >= 4 and token_strs[:4] == think_prefix:
        return 4
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Token probability and rank analysis for model outputs"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to HuggingFace model (local or hub identifier)",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="token_distribution_analysis/data/diff_sampled_10000",
        help="Directory containing per-subtask JSONL files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="token_distribution_analysis/results",
        help="Directory to save result txt files",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=None,
        help="Truncate sequences exceeding this length (default: no limit)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per subtask (for quick debugging)",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=False,
        help="Enable thinking mode for the model",
    )
    args = parser.parse_args()

    # ---- Load model & tokenizer ----
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")

    print(f"ℹ️ℹ️ℹ️ enable_thinking: {args.enable_thinking}")
    n_trailing = compute_trailing_special_count(tokenizer, args.enable_thinking)
    # n_trailing = compute_trailing_special_count(tokenizer)
    print(f"Trailing special tokens after assistant content: {n_trailing}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(str(output_dir) + "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raw per-sample data will be saved to: {raw_dir}")

    # ---- Process each subtask ----
    all_results: Dict[str, Dict[str, Any]] = {}
    all_sample_probs: List[float] = []
    all_sample_ranks: List[float] = []
    all_sample_top1_hit_rates: List[float] = []
    all_sample_probs_excl_think_prefix: List[float] = []
    all_sample_ranks_excl_think_prefix: List[float] = []
    all_sample_top1_hit_rates_excl_think_prefix: List[float] = []

    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        task = jsonl_path.stem
        print(f"\n{'=' * 60}\nSubtask: {task}")

        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        if args.max_samples:
            records = records[: args.max_samples]
        print(f"  Samples to process: {len(records)}")

        sample_probs = []
        sample_ranks = []
        sample_top1_hit_rates = []
        sample_probs_excl_think_prefix = []
        sample_ranks_excl_think_prefix = []
        sample_top1_hit_rates_excl_think_prefix = []
        raw_records = []
        skipped = 0
        n_samples_with_think_prefix = 0

        for record in tqdm(records, desc=f"  {task}"):
            raw_output = record["output"]
            t = record.get("task") or task
            prompt_msgs = build_prompt_messages(record, t)

            try:
                result = analyze_sample(
                    model, tokenizer, prompt_msgs, raw_output,
                    n_trailing, args.max_seq_len,
                    # args.enable_thinking
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            (
                avg_prob,
                avg_rank,
                avg_top1_hit_rate,
                n_tokens,
                token_probs_list,
                token_ranks_list,
                token_top1_hits_list,
                token_strs,
            ) = result
            sample_probs.append(avg_prob)
            sample_ranks.append(avg_rank)
            sample_top1_hit_rates.append(avg_top1_hit_rate)

            n_prefix = _leading_think_prefix_len(token_strs)
            if n_prefix > 0:
                n_samples_with_think_prefix += 1
            if 0 < n_prefix < len(token_probs_list):
                avg_prob_excl = float(np.mean(token_probs_list[n_prefix:]))
                avg_rank_excl = float(np.mean(token_ranks_list[n_prefix:]))
                avg_top1_hit_rate_excl = float(np.mean(token_top1_hits_list[n_prefix:]))
            else:
                avg_prob_excl = avg_prob
                avg_rank_excl = avg_rank
                avg_top1_hit_rate_excl = avg_top1_hit_rate
            sample_probs_excl_think_prefix.append(avg_prob_excl)
            sample_ranks_excl_think_prefix.append(avg_rank_excl)
            sample_top1_hit_rates_excl_think_prefix.append(avg_top1_hit_rate_excl)

            raw_records.append({
                "sample_id": record.get("sample_id", ""),
                "task": task,
                "output": raw_output,
                "avg_prob": avg_prob,
                "avg_rank": avg_rank,
                "avg_top1_hit_rate": avg_top1_hit_rate,
                "avg_prob_excl_leading_think_prefix": avg_prob_excl,
                "avg_rank_excl_leading_think_prefix": avg_rank_excl,
                "avg_top1_hit_rate_excl_leading_think_prefix": avg_top1_hit_rate_excl,
                "n_tokens": n_tokens,
                "token_strs": token_strs,
                "token_probs": token_probs_list,
                "token_ranks": token_ranks_list,
                "token_top1_hits": token_top1_hits_list,
            })

        n_valid = len(sample_probs)
        subtask_prob = sum(sample_probs) / n_valid if n_valid else 0.0
        subtask_rank = sum(sample_ranks) / n_valid if n_valid else 0.0
        subtask_top1_hit_rate = (
            sum(sample_top1_hit_rates) / n_valid if n_valid else 0.0
        )
        subtask_prob_excl = (
            sum(sample_probs_excl_think_prefix) / n_valid if n_valid else 0.0
        )
        subtask_rank_excl = (
            sum(sample_ranks_excl_think_prefix) / n_valid if n_valid else 0.0
        )
        subtask_top1_hit_rate_excl = (
            sum(sample_top1_hit_rates_excl_think_prefix) / n_valid if n_valid else 0.0
        )

        prob_stats = (
            compute_distribution_stats(sample_probs) if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        rank_stats = (
            compute_distribution_stats(sample_ranks) if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        top1_hit_rate_stats = (
            compute_distribution_stats(sample_top1_hit_rates) if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        prob_stats_excl = (
            compute_distribution_stats(sample_probs_excl_think_prefix) if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        rank_stats_excl = (
            compute_distribution_stats(sample_ranks_excl_think_prefix) if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        top1_hit_rate_stats_excl = (
            compute_distribution_stats(sample_top1_hit_rates_excl_think_prefix)
            if n_valid
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )

        all_results[task] = {
            "avg_prob": subtask_prob,
            "avg_rank": subtask_rank,
            "avg_top1_hit_rate": subtask_top1_hit_rate,
            "avg_prob_excl_leading_think_prefix": subtask_prob_excl,
            "avg_rank_excl_leading_think_prefix": subtask_rank_excl,
            "avg_top1_hit_rate_excl_leading_think_prefix": subtask_top1_hit_rate_excl,
            "n_samples": n_valid,
            "n_skipped": skipped,
            "n_samples_with_think_prefix": n_samples_with_think_prefix,
            "prob_stats": prob_stats,
            "rank_stats": rank_stats,
            "top1_hit_rate_stats": top1_hit_rate_stats,
            "prob_stats_excl_leading_think_prefix": prob_stats_excl,
            "rank_stats_excl_leading_think_prefix": rank_stats_excl,
            "top1_hit_rate_stats_excl_leading_think_prefix": top1_hit_rate_stats_excl,
        }

        # ---- Save raw per-sample data (with per-token arrays) ----
        raw_path = raw_dir / f"{task}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for rec in raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_sample_probs.extend(sample_probs)
        all_sample_ranks.extend(sample_ranks)
        all_sample_top1_hit_rates.extend(sample_top1_hit_rates)
        all_sample_probs_excl_think_prefix.extend(sample_probs_excl_think_prefix)
        all_sample_ranks_excl_think_prefix.extend(sample_ranks_excl_think_prefix)
        all_sample_top1_hit_rates_excl_think_prefix.extend(
            sample_top1_hit_rates_excl_think_prefix
        )

        # ---- Per-subtask text report ----
        result_path = output_dir / f"{task}.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"Subtask: {task}\n")
            f.write(f"Processed samples: {n_valid}\n")
            f.write(f"Skipped samples: {skipped}\n")
            f.write(f"Average probability: {subtask_prob:.6f}\n")
            f.write(f"  {_fmt_stats(prob_stats, '.6f')}\n")
            f.write(f"Average rank: {subtask_rank:.2f}\n")
            f.write(f"  {_fmt_stats(rank_stats, '.2f')}\n")
            f.write(
                "Average Top-1 hit rate (ground-truth token is rank-1): "
                f"{subtask_top1_hit_rate:.6f}\n"
            )
            f.write(f"  {_fmt_stats(top1_hit_rate_stats, '.6f')}\n")

        print(f"  Valid: {n_valid}, Skipped: {skipped}")
        print(f"  Avg prob: {subtask_prob:.6f}  ({_fmt_stats(prob_stats, '.6f')})")
        print(f"  Avg rank: {subtask_rank:.2f}  ({_fmt_stats(rank_stats, '.2f')})")
        print(
            "  Avg top-1 hit rate (ground-truth token is rank-1): "
            f"{subtask_top1_hit_rate:.6f}  ({_fmt_stats(top1_hit_rate_stats, '.6f')})"
        )
        print(f"  Saved: {result_path}")
        print(f"  Raw data: {raw_path}")

    # ---- Cross-subtask summary ----
    if all_results:
        valid_entries = {
            k: v for k, v in all_results.items() if v["n_samples"] > 0
        }
        if valid_entries:
            total_n = sum(v["n_samples"] for v in valid_entries.values())
            global_prob = (
                sum(v["avg_prob"] * v["n_samples"] for v in valid_entries.values()) / total_n
            )
            global_rank = (
                sum(v["avg_rank"] * v["n_samples"] for v in valid_entries.values()) / total_n
            )
            global_top1_hit_rate = (
                sum(
                    v["avg_top1_hit_rate"] * v["n_samples"]
                    for v in valid_entries.values()
                ) / total_n
            )
            global_prob_excl = (
                sum(
                    v["avg_prob_excl_leading_think_prefix"] * v["n_samples"]
                    for v in valid_entries.values()
                ) / total_n
            )
            global_rank_excl = (
                sum(
                    v["avg_rank_excl_leading_think_prefix"] * v["n_samples"]
                    for v in valid_entries.values()
                ) / total_n
            )
            global_top1_hit_rate_excl = (
                sum(
                    v["avg_top1_hit_rate_excl_leading_think_prefix"] * v["n_samples"]
                    for v in valid_entries.values()
                ) / total_n
            )
        else:
            global_prob = global_rank = 0.0
            global_prob_excl = global_rank_excl = 0.0
            global_top1_hit_rate = global_top1_hit_rate_excl = 0.0

        global_prob_stats = (
            compute_distribution_stats(all_sample_probs) if all_sample_probs
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        global_rank_stats = (
            compute_distribution_stats(all_sample_ranks) if all_sample_ranks
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        global_top1_hit_rate_stats = (
            compute_distribution_stats(all_sample_top1_hit_rates)
            if all_sample_top1_hit_rates
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        global_prob_stats_excl = (
            compute_distribution_stats(all_sample_probs_excl_think_prefix)
            if all_sample_probs_excl_think_prefix
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        global_rank_stats_excl = (
            compute_distribution_stats(all_sample_ranks_excl_think_prefix)
            if all_sample_ranks_excl_think_prefix
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )
        global_top1_hit_rate_stats_excl = (
            compute_distribution_stats(all_sample_top1_hit_rates_excl_think_prefix)
            if all_sample_top1_hit_rates_excl_think_prefix
            else {"mean": 0.0, "std": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        )

        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Token Distribution Analysis Summary\n")
            f.write(f"Model: {args.model_path}\n")
            f.write("=" * 60 + "\n\n")

            for task in sorted(all_results):
                r = all_results[task]
                ps, rs = r["prob_stats"], r["rank_stats"]
                hs = r["top1_hit_rate_stats"]
                ps_excl = r["prob_stats_excl_leading_think_prefix"]
                rs_excl = r["rank_stats_excl_leading_think_prefix"]
                hs_excl = r["top1_hit_rate_stats_excl_leading_think_prefix"]
                f.write(f"{task}:\n")
                f.write(f"  Samples: {r['n_samples']} (skipped: {r['n_skipped']})\n")
                f.write(f"  Avg Probability: {r['avg_prob']:.6f}\n")
                f.write(f"    {_fmt_stats(ps, '.6f')}\n")
                f.write(f"  Avg Rank: {r['avg_rank']:.2f}\n")
                f.write(f"    {_fmt_stats(rs, '.2f')}\n")
                f.write(
                    "  Avg Top-1 hit rate (ground-truth token is rank-1): "
                    f"{r['avg_top1_hit_rate']:.6f}\n"
                )
                f.write(f"    {_fmt_stats(hs, '.6f')}\n\n")
                f.write(
                    "  Samples with leading '<think> \\n\\n </think> \\n\\n' prefix: "
                    f"{r['n_samples_with_think_prefix']}\n"
                )
                f.write(
                    "  Avg Probability (excluding leading think-prefix tokens): "
                    f"{r['avg_prob_excl_leading_think_prefix']:.6f}\n"
                )
                f.write(f"    {_fmt_stats(ps_excl, '.6f')}\n")
                f.write(
                    "  Avg Rank (excluding leading think-prefix tokens): "
                    f"{r['avg_rank_excl_leading_think_prefix']:.2f}\n"
                )
                f.write(f"    {_fmt_stats(rs_excl, '.2f')}\n")
                f.write(
                    "  Avg Top-1 hit rate (excluding leading think-prefix tokens): "
                    f"{r['avg_top1_hit_rate_excl_leading_think_prefix']:.6f}\n"
                )
                f.write(f"    {_fmt_stats(hs_excl, '.6f')}\n\n")

            f.write("-" * 60 + "\n")
            f.write(
                f"Cross-subtask Sample-Weighted Average Probability: {global_prob:.6f}\n"
            )
            f.write(f"  {_fmt_stats(global_prob_stats, '.6f')}\n")
            f.write(
                f"Cross-subtask Sample-Weighted Average Rank: {global_rank:.2f}\n"
            )
            f.write(f"  {_fmt_stats(global_rank_stats, '.2f')}\n")
            f.write(
                "Cross-subtask Sample-Weighted Average Top-1 hit rate "
                "(ground-truth token is rank-1): "
                f"{global_top1_hit_rate:.6f}\n"
            )
            f.write(f"  {_fmt_stats(global_top1_hit_rate_stats, '.6f')}\n")
            f.write(
                "Cross-subtask Sample-Weighted Average Probability "
                "(excluding leading think-prefix tokens): "
                f"{global_prob_excl:.6f}\n"
            )
            f.write(f"  {_fmt_stats(global_prob_stats_excl, '.6f')}\n")
            f.write(
                "Cross-subtask Sample-Weighted Average Rank "
                "(excluding leading think-prefix tokens): "
                f"{global_rank_excl:.2f}\n"
            )
            f.write(f"  {_fmt_stats(global_rank_stats_excl, '.2f')}\n")
            f.write(
                "Cross-subtask Sample-Weighted Average Top-1 hit rate "
                "(excluding leading think-prefix tokens): "
                f"{global_top1_hit_rate_excl:.6f}\n"
            )
            f.write(f"  {_fmt_stats(global_top1_hit_rate_stats_excl, '.6f')}\n")
            f.write(f"\nRaw per-sample data saved to: {raw_dir}\n")

        print(f"\n{'=' * 60}")
        print(f"Cross-subtask sample-weighted Avg Prob: {global_prob:.6f}")
        print(f"  {_fmt_stats(global_prob_stats, '.6f')}")
        print(f"Cross-subtask sample-weighted Avg Rank: {global_rank:.2f}")
        print(f"  {_fmt_stats(global_rank_stats, '.2f')}")
        print(
            "Cross-subtask sample-weighted Avg Top-1 hit rate "
            f"(ground-truth token is rank-1): {global_top1_hit_rate:.6f}"
        )
        print(f"  {_fmt_stats(global_top1_hit_rate_stats, '.6f')}")
        print(
            "Cross-subtask sample-weighted Avg Prob "
            f"(excluding leading think-prefix): {global_prob_excl:.6f}"
        )
        print(f"  {_fmt_stats(global_prob_stats_excl, '.6f')}")
        print(
            "Cross-subtask sample-weighted Avg Rank "
            f"(excluding leading think-prefix): {global_rank_excl:.2f}"
        )
        print(f"  {_fmt_stats(global_rank_stats_excl, '.2f')}")
        print(
            "Cross-subtask sample-weighted Avg Top-1 hit rate "
            f"(excluding leading think-prefix): {global_top1_hit_rate_excl:.6f}"
        )
        print(f"  {_fmt_stats(global_top1_hit_rate_stats_excl, '.6f')}")
        print(f"Summary saved: {summary_path}")
        print(f"Raw data saved to: {raw_dir}")


if __name__ == "__main__":
    main()
