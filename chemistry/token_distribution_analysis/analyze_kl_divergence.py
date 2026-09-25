#!/usr/bin/env python3
"""
KL divergence analysis between a base model (M0) and fine-tuned models.

For each subtask JSONL in the data directory:
1. Convert each record to ShareGPT-format prompt (system + user, excluding assistant)
2. Use teacher forcing (single forward pass per model) to get next-token distributions
3. Compute per-token KL(P_ft || P_base) divergence
4. Average per sample, then per subtask
5. Save per-subtask results and cross-subtask summary for each fine-tuned model

Both models are loaded simultaneously.  Use --base_gpus / --ft_gpus to pin each
model to disjoint GPU sets when memory is tight (e.g. --base_gpus 0,1 --ft_gpus 2,3).
"""

import json
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt_messages(record: Dict[str, Any], task: str):
    """Build ShareGPT prompt messages (system + user), excluding assistant."""
    task_instruction = SMOL_TASK_INSTRUCTION.get(task, "")
    user_content = task_instruction + record["input"] + "\nAnswer: "
    return [
        {"role": "system", "content": NOTHINK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def compute_trailing_special_count(tokenizer) -> int:
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
        test_prompt, add_generation_prompt=True, tokenize=True,
        enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(test_full, tokenize=True,
        enable_thinking=False
    )
    return len(full_ids) - len(prompt_ids)


def parse_gpu_ids(gpu_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated GPU id string into list of ints."""
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",")]


def load_model(model_path: str, gpu_ids: Optional[List[int]] = None):
    """Load a causal LM distributed across *gpu_ids* (or auto)."""
    kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if gpu_ids is not None:
        max_memory = {}
        for gid in gpu_ids:
            total_bytes = torch.cuda.get_device_properties(gid).total_memory
            max_memory[gid] = f"{int(total_bytes * 0.85 / (1024 ** 3))}GiB"
        max_memory["cpu"] = "100GiB"
        kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model


def get_input_device(model) -> torch.device:
    """Determine where to place input tensors for *model*."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first = list(model.hf_device_map.values())[0]
        if isinstance(first, int):
            return torch.device(f"cuda:{first}")
        return torch.device(first)
    return model.device


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sample_kl(
    model_base,
    model_ft,
    tokenizer,
    prompt_messages,
    raw_output: str,
    n_trailing: int,
    max_seq_len: Optional[int],
    base_device: torch.device,
    ft_device: torch.device,
) -> Optional[Tuple[float, int]]:
    """
    Compute average per-token KL(P_ft || P_base) for a single sample.

    Uses teacher forcing: a single forward pass over [prompt, answer_tokens]
    through each model.  Because the model is causal, logits at position i
    only depend on tokens 0..i, so one pass gives us the full next-token
    distribution at every output position.

    Returns (avg_kl, n_content_tokens) or None if the sample is skipped.
    """
    full_messages = prompt_messages + [
        {"role": "assistant", "content": raw_output}
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True,
        enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(full_messages, tokenize=True,
        enable_thinking=False
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

    input_ids = full_ids[:end_idx]

    # --- forward: base model ---
    inp_base = torch.tensor([input_ids], dtype=torch.long, device=base_device)
    with torch.no_grad():
        logits_base = model_base(inp_base).logits[0]

    # --- forward: fine-tuned model ---
    inp_ft = torch.tensor([input_ids], dtype=torch.long, device=ft_device)
    with torch.no_grad():
        logits_ft = model_ft(inp_ft).logits[0]

    # Move to CPU in fp32 to avoid cross-device / precision issues
    base_out = logits_base[prompt_len - 1 : end_idx - 1].float().cpu()
    ft_out = logits_ft[prompt_len - 1 : end_idx - 1].float().cpu()

    log_p_base = F.log_softmax(base_out, dim=-1)
    log_p_ft = F.log_softmax(ft_out, dim=-1)

    # KL(P_ft || P_base) = E_{P_ft}[log P_ft - log P_base], averaged over positions
    avg_kl = F.kl_div(
        log_p_base, log_p_ft, log_target=True, reduction="batchmean"
    ).item()

    del inp_base, inp_ft, logits_base, logits_ft, base_out, ft_out
    del log_p_base, log_p_ft

    return avg_kl, n_content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KL divergence analysis: fine-tuned models vs. base model"
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Path to base (un-finetuned) model M0",
    )
    parser.add_argument(
        "--ft_model_paths", type=str, nargs="+", required=True,
        help="Paths to fine-tuned models (M1, M2, ...)",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="token_distribution_analysis/data/diff_sampled_10000",
        help="Directory containing per-subtask JSONL files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="token_distribution_analysis/kl_results",
        help="Root directory to save results (sub-dir per FT model)",
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
        "--base_gpus", type=str, default=None,
        help="Comma-separated GPU ids for base model (e.g. '0,1'). Default: auto",
    )
    parser.add_argument(
        "--ft_gpus", type=str, default=None,
        help="Comma-separated GPU ids for FT models (e.g. '2,3'). Default: auto",
    )
    args = parser.parse_args()

    base_gpu_ids = parse_gpu_ids(args.base_gpus)
    ft_gpu_ids = parse_gpu_ids(args.ft_gpus)

    # ---- tokenizer (shared across all models from the same family) ----
    print(f"Loading tokenizer from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    n_trailing = compute_trailing_special_count(tokenizer)
    print(f"Trailing special tokens after assistant content: {n_trailing}")

    # ---- base model (stays loaded throughout) ----
    print(f"\nLoading base model: {args.base_model_path}")
    model_base = load_model(args.base_model_path, base_gpu_ids)
    base_device = get_input_device(model_base)
    print(f"Base model loaded, input device: {base_device}")

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)

    # ---- iterate over fine-tuned models ----
    for ft_path in args.ft_model_paths:
        ft_name = Path(ft_path).name
        print(f"\n{'#' * 70}")
        print(f"Fine-tuned model: {ft_name}")
        print(f"Path: {ft_path}")

        ft_model = load_model(ft_path, ft_gpu_ids)
        ft_device = get_input_device(ft_model)
        print(f"FT model loaded, input device: {ft_device}")

        output_dir = output_root / ft_name
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results: Dict[str, Dict[str, Any]] = {}

        for jsonl_path in sorted(data_dir.glob("*.jsonl")):
            task = jsonl_path.stem
            print(f"\n{'=' * 60}\nSubtask: {task}")

            records: List[Dict[str, Any]] = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))

            if args.max_samples:
                records = records[: args.max_samples]
            print(f"  Samples to process: {len(records)}")

            sample_kls: List[float] = []
            skipped = 0

            for record in tqdm(records, desc=f"  {task}"):
                raw_output = record["output"]
                t = record.get("task") or task
                prompt_msgs = build_prompt_messages(record, t)

                try:
                    result = compute_sample_kl(
                        model_base, ft_model, tokenizer,
                        prompt_msgs, raw_output, n_trailing,
                        args.max_seq_len, base_device, ft_device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    skipped += 1
                    continue

                if result is None:
                    skipped += 1
                    continue

                avg_kl, _ = result
                sample_kls.append(avg_kl)

            n_valid = len(sample_kls)
            subtask_kl = sum(sample_kls) / n_valid if n_valid else 0.0

            all_results[task] = {
                "avg_kl": subtask_kl,
                "n_samples": n_valid,
                "n_skipped": skipped,
            }

            result_path = output_dir / f"{task}.txt"
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(f"Subtask: {task}\n")
                f.write(f"Base model: {args.base_model_path}\n")
                f.write(f"FT model:   {ft_path}\n")
                f.write(f"Processed samples: {n_valid}\n")
                f.write(f"Skipped samples:   {skipped}\n")
                f.write(f"Average KL(P_ft || P_base): {subtask_kl:.6f}\n")

            print(f"  Valid: {n_valid}, Skipped: {skipped}")
            print(f"  Avg KL(P_ft || P_base): {subtask_kl:.6f}")
            print(f"  Saved: {result_path}")

        # ---- cross-subtask summary for this FT model ----
        if all_results:
            valid_entries = {
                k: v for k, v in all_results.items() if v["n_samples"] > 0
            }
            if valid_entries:
                total_n = sum(v["n_samples"] for v in valid_entries.values())
                global_kl = (
                    sum(v["avg_kl"] * v["n_samples"] for v in valid_entries.values())
                    / total_n
                )
            else:
                global_kl = 0.0

            summary_path = output_dir / "summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("KL Divergence Analysis Summary\n")
                f.write(f"Base model: {args.base_model_path}\n")
                f.write(f"FT model:   {ft_path}\n")
                f.write("=" * 60 + "\n\n")

                for task in sorted(all_results):
                    r = all_results[task]
                    f.write(f"{task}:\n")
                    f.write(
                        f"  Samples: {r['n_samples']} "
                        f"(skipped: {r['n_skipped']})\n"
                    )
                    f.write(f"  Avg KL(P_ft || P_base): {r['avg_kl']:.6f}\n\n")

                f.write("-" * 60 + "\n")
                f.write(
                    f"Cross-subtask Sample-Weighted Average KL: {global_kl:.6f}\n"
                )

            print(f"\n{'=' * 60}")
            print(f"Cross-subtask sample-weighted Avg KL: {global_kl:.6f}")
            print(f"Summary saved: {summary_path}")

        # free FT model memory before loading the next one
        del ft_model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
