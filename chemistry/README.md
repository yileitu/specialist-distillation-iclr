# Chemistry experiments

This directory contains the Chemistry track for *Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation*. It covers SMolInstruct data preparation, specialist adaptation, trajectory generation, answer extraction and filtering, final fine-tuning data construction, evaluation support, and trajectory-distribution analysis.

For the cross-domain overview, see the [project README](../README.md).

## Dataset and tasks

The primary dataset is [SMolInstruct](https://huggingface.co/datasets/osunlp/SMolInstruct). The track aggregates 14 chemistry subtasks spanning molecular property prediction, molecular naming and representation conversion, captioning, generation, retrosynthesis, and forward synthesis.

See the [SMolInstruct terminology and task reference](smol/README.md) for the complete task list, expected output tags, and short task definitions.

## Directory layout

- `ft_scripts/`: ShareGPT dataset registration and FFT/LST YAML generation.
- `lora_scripts/`: LoRA YAML generation and adapter-weight merging.
- `asft_scripts/`: ASFT YAML generation for explicit anchoring ablations.
- `smol/data/`: per-task sampling and upsampling utilities.
- `smol/inference/`: resumable vLLM generation with optional reasoning modes and repetition stopping.
- `smol/postprocess_answer/`: task-specific answer extraction for Boolean, numeric, SMILES, molecular-formula, IUPAC, and captioning outputs.
- `smol/prepare_ft_data/`: correctness filtering, reasoning-tag validation, alignment, merging, and subsampling.
- `smol/analyze_inference_data/`: trajectory, token, METEOR, and RMSE diagnostics.
- `token_distribution_analysis/`: KL-divergence and token probability/rank analyses.
- `util/`: general JSON/JSONL and chemistry helpers.

## Setup

```bash
bash requirements.sh
```

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) separately for training. Individual analysis scripts may require additional optional packages; check their imports before running them.

## Workflow

### 1. Download and inspect SMolInstruct

```bash
python smol/eval/download_smolinstruct.py --output-dir <smolinstruct-dir>
```

The downloader uses the Hugging Face datasets cache by default. For evaluation details and the upstream metric implementation, see the [SMolInstruct evaluation notes](smol/eval/README.md).

### 2. Prepare task-balanced data

Use the sampling utilities to create deterministic per-task files. Generated JSONL data is ignored by Git.

```bash
python smol/data/sample_by_task.py \
  --data_file <input.jsonl> \
  --samples_per_task <count> \
  --output_dir <sampled-data-dir>

python smol/data/upsample_to_50k.py --help
```

Training data follows ShareGPT JSONL format: each line contains a `messages` list whose entries have `role` and `content` fields.

### 3. Register data and generate a training configuration

```bash
python ft_scripts/data_prepare.py \
  --data_file <training.jsonl> \
  --dataset_name <dataset-name> \
  --dataset_info_path <dataset_info.json>
```

Choose the configuration generator for the desired specialist:

```bash
# FFT, or LST when layer-selection arguments are supplied.
python ft_scripts/generate_training_yaml.py --help

# LoRA specialist.
python lora_scripts/generate_lora_yaml.py --help

# ASFT drift-control ablation.
python asft_scripts/generate_asft_yaml.py --help
```

Each generator prints the `llamafactory-cli train <yaml-path>` command to run. Inspect the YAML before launching training.

### 4. Generate candidate trajectories

```bash
python smol/inference/interns1_infer_subtasks.py \
  --model_path <specialist-model> \
  --data_file <task-test.jsonl> \
  --output_dir <inference-output-dir>
```

The generator supports multiple samples per question, checkpointed/resumable output, optional reasoning disablement, answer-conditioned reasoning, and repeated-pattern stopping. Run `--help` for all generation controls.

### 5. Extract answers and build distillation data

Use the task-appropriate script in `smol/postprocess_answer/`, then use `smol/prepare_ft_data/` to retain answer-correct generations, validate `<think>...</think>` structure, align records, and create the final ShareGPT training set.

The extraction strategies are documented alongside the scripts:

- [Boolean tasks](smol/postprocess_answer/strategies/BOOL_TASKS_EXTRACTION_STRATEGY.md)
- [Numeric tasks](smol/postprocess_answer/strategies/NUM_TASKS_EXTRACTION_STRATEGY.md)
- [SMILES tasks](smol/postprocess_answer/strategies/SMILES_TASKS_EXTRACTION_STRATEGY.md)
- [Molecular-formula and IUPAC tasks](smol/postprocess_answer/strategies/MOLFORMULA_IUPAC_TASKS_EXTRACTION_STRATEGY.md)

### 6. Evaluate and analyze

Use `smol/analyze_inference_data/` for task-level diagnostics and `token_distribution_analysis/` for the probability/rank and KL-divergence analyses discussed in the paper. The full SMolInstruct metric pipeline is based on the upstream LLM4Chem evaluation code, as described in the [evaluation notes](smol/eval/README.md).

## Portability

All dataset, model, and output locations should be supplied for the current environment. The repository intentionally does not provide scheduler-specific submission scripts or fixed compute-resource requests.
