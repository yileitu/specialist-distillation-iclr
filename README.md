# Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation

This repository contains the research code for the **Chemistry** and **Multilingualism** experiments in *Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation*. The paper also reports Physics experiments; that track is not part of this repository.

## Overview

The project studies specialist distillation when domain supervision contains questions and answers but no teacher reasoning trajectories. Its main workflow is:

1. Adapt an origin model to domain question-answer pairs to obtain an intermediate specialist.
2. Ask the specialist to generate candidate rationales and answers.
3. Apply task-specific answer and trajectory filters.
4. Fine-tune a final model on the retained question-rationale-answer triples.
5. Evaluate task specialization, in-domain transfer, and out-of-domain retention.

The included training utilities cover full fine-tuning (FFT), LoRA, and layer-selective tuning (LST). Anchored supervised fine-tuning (ASFT) utilities support the explicit drift-control ablations described in the paper.

## Repository structure

| Path | Scope | Main data and evaluation |
| --- | --- | --- |
| [`chemistry/`](chemistry/) | Chemistry specialist training, generation, filtering, and analysis | SMolInstruct and its 14 subtasks |
| [`multilingualism/`](multilingualism/) | Bidirectional low-resource translation training, generation, spBLEU filtering, and evaluation preparation | OPUS training data and Flores-101 evaluation data |

Detailed documentation:

- [Chemistry guide](chemistry/README.md)
- [SMolInstruct terminology and task reference](chemistry/smol/README.md)
- [SMolInstruct evaluation notes](chemistry/smol/eval/README.md)
- [Multilingualism guide](multilingualism/README.md)

## Environment

The code is a collection of research utilities rather than an installable Python package. Python 3.10 or newer is recommended because some scripts use modern type annotations.

Create an isolated environment, then install the dependencies required by the track you intend to run. The Chemistry dependency helper provides a starting point:

```bash
cd chemistry
bash requirements.sh
```

Training configuration generation and execution use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which must be installed separately. Translation postprocessing additionally requires `sacrebleu`. Some analysis scripts use optional scientific and plotting packages such as SciPy, Matplotlib, Seaborn, and NLTK.

## Quick start

Run commands from the relevant track directory unless a command says otherwise. Replace angle-bracketed values with paths from your own environment.

### Chemistry

```bash
cd chemistry

# Download the public SMolInstruct dataset.
python smol/eval/download_smolinstruct.py --output-dir <smolinstruct-dir>

# Inspect data registration and training configuration options.
python ft_scripts/data_prepare.py --help
python ft_scripts/generate_training_yaml.py --help
python lora_scripts/generate_lora_yaml.py --help
python asft_scripts/generate_asft_yaml.py --help

# Inspect specialist trajectory generation options.
python smol/inference/interns1_infer_subtasks.py --help
```

See the [Chemistry guide](chemistry/README.md) for task-specific postprocessing, filtering, and analysis utilities.

### Multilingualism

The paper uses English paired with eight low-resource languages in both directions: Bengali (`bn`), Czech (`cs`), Hungarian (`hu`), Serbian (`sr`), Swahili (`sw`), Telugu (`te`), Thai (`th`), and Vietnamese (`vi`).

```bash
cd multilingualism

# Convert ShareGPT translation records into direction-specific inference files.
python translation_data/process_rawfull_data.py \
  --input-dir <sharegpt-jsonl-dir> \
  --output-dir <inference-data-dir>

# Generate candidate translations and rationales.
python inference/infer_translation.py \
  --model_path <model-path> \
  --data_file <inference-jsonl> \
  --output_dir <inference-output-dir>

# Compute spBLEU, validate reasoning tags, and retain the top-scoring records.
python inference/postprocess_translation.py \
  --input_path <inference-output-file-or-dir> \
  --top_ratio 0.2
```

See the [Multilingualism guide](multilingualism/README.md) for local training launchers and multi-file workflows.

## Data and artifacts

Datasets, generated JSONL files, model checkpoints, plots, logs, and experiment outputs are not bundled with this repository. The scripts accept explicit input and output paths so that users can choose their own storage layout. Large generated artifacts are excluded by `.gitignore`.

Before training, inspect the generated YAML file and adapt batch sizes, precision, sequence length, and distributed-training settings to your software and hardware environment.

## Reproducibility notes

The paper's reference experiments start from Qwen3-8B, while most utilities accept any model supported by the installed versions of Transformers, vLLM, and LLaMA-Factory. Randomized data utilities expose a seed; keep the seed, dataset revision, model revision, and generated YAML files with your experiment records.

No scheduler-specific launch configuration or organization-specific compute allocation is assumed. Use your own local or cluster launcher around the provided commands when distributed execution is needed.

## Citation

If you use this code, please cite *Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation*. A complete citation entry will be added when the paper's public bibliographic record is available.

## License

This project is released under the [MIT License](LICENSE).
