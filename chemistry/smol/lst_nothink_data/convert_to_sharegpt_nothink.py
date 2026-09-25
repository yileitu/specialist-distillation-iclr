#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert task JSONL files under upsampled_50k_all_subtasks to ShareGPT format
without chain-of-thought, shuffle them, and save one merged JSONL file.

- Replace input/output with messages containing system, user, and assistant entries.
- Build user content from task_instruction + input + "\nAnswer: ".
- Keep the system content aligned with the non-thinking branch of build_messages.
- Preserve all remaining fields.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List

# Keep this consistent with the system prompt from build_messages(enable_thinking=False) in interns1_infer_subtasks.py
NOTHINK_SYSTEM_PROMPT = (
    "You are an expert with extensive knowledge in all areas. "
    "Provide direct answers to the user's question without showing your reasoning process. "
    "Do not use <think> and </think> tags."
)

SMOL_TASK_INSTRUCTION = {
    "forward_synthesis": "You are an expert chemist. Given the SMILES representation of reactants and reagents, your task is to predict the potential product using your chemical reaction knowledge.\n    The input contains both reactants and reagents, and different reactants and reagents are separated by \".\". Your reply should contain the SMILES representation of the predicted product wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "retrosynthesis": "You are an expert chemist. Given the SMILES representation of the product, your task is to predict the potential reactants and reagents using your chemical reaction knowledge.\n    The input contains the SMILES representation of the product. Your reply should contain the SMILES representation of both reactants and reagents, and all reactants and reagents should be enclosed **together** within a single pair of <SMILES> and </SMILES> tags, separated by \".\". Your reply must be valid and chemically reasonable.\nQuestion: ",
    "molecule_captioning": "You are an expert chemist. Given the SMILES representation of a molecule, your task is to describe the molecule in natural language.\n    The input contains the SMILES representation of the molecule. Your reply should contain a natural language description of the molecule. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "molecule_generation": "You are an expert chemist. Given the description of a molecule, your task is to generate the potential SMILES representation of the molecule.\n    The input contains the description of the molecule. Your reply should contain the potential SMILES representation of the molecule wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-i2f": "You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the molecular formula of the compound.\n    The input contains the IUPAC representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-i2s": "You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the SMILES representation of the compound.\n    The input contains the IUPAC representation of the compound. Your reply should contain only the SMILES representation of the compound wrapped in <SMILES> and </SMILES> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-s2f": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the molecular formula of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-s2i": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the IUPAC representation of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain only the IUPAC representation of the compound wrapped in <IUPAC> and </IUPAC> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-esol": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the log solubility of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain the log solubility of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-lipo": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the octanol/water partition coefficient of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain the octanol/water partition coefficient of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-bbbp": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether blood-brain barrier permeability (BBBP) is a property of the compound.\n    The input contains the compound. Your reply should only contain Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-clintox": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound is toxic.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-hiv": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound serve as an inhibitor of HIV replication.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-sider": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound has any side effects.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
}

SMOL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = str(SMOL_ROOT / "data" / "upsampled_50k_all_subtasks")
DEFAULT_OUTPUT_FILE = str(Path(__file__).resolve().parent / "upsampled_50k_all_subtasks_lst_nothink_data.jsonl")


def record_to_sharegpt_nothink(record: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Convert one source record to ShareGPT format without chain-of-thought.

    - Use task_instruction + input + "\nAnswer: " as user content.
    - Set messages to [system, user, assistant], using the inference script's non-thinking system prompt.
    - Preserve other fields, remove input/output, and add messages.
    """
    raw_input = record["input"]
    raw_output = record["output"]
    task_instruction = SMOL_TASK_INSTRUCTION.get(task, "")
    prompt = task_instruction + raw_input + "\nAnswer: "

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": NOTHINK_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": raw_output},
    ]

    out = {k: v for k, v in record.items() if k not in ("input", "output")}
    out["messages"] = messages
    return out


def load_task_jsonl(path: Path, task: str) -> List[Dict[str, Any]]:
    """Read one task JSONL file and convert each record to ShareGPT format."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Use the filename stem when a record has no task field
            t = record.get("task") or task
            rows.append(record_to_sharegpt_nothink(record, t))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert upsampled_50k task jsonl to ShareGPT no-think format and merge.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-task jsonl files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output merged and shuffled jsonl path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Process only top-level JSONL files whose stems match keys in SMOL_TASK_INSTRUCTION
    all_data: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        task = path.stem
        if task not in SMOL_TASK_INSTRUCTION:
            print(f"Skip (no task instruction): {path.name}")
            continue
        rows = load_task_jsonl(path, task)
        all_data.extend(rows)
        print(f"Loaded {path.name}: {len(rows)} samples")

    random.seed(args.seed)
    random.shuffle(all_data)
    print(f"Total samples: {len(all_data)}")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in all_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
