#!/bin/bash
#
# General-purpose two-stage layer-selective training (LST) launcher
#
# Usage:
#   bash submit_lst.sh --model /path/to/model --data /path/to/data.jsonl [OPTIONS]
#
# Examples:
#   # Basic usage with any model and dataset
#   bash submit_lst.sh \
#     --model /path/to/model \
#     --data /path/to/aligned_ft_data.jsonl
#
#   # Set LST layer counts and training options
#   bash submit_lst.sh \
#     --model /path/to/model \
#     --data /path/to/data.jsonl \
#     --bottom-layers 4 \
#     --top-layers 16 \
#     --epochs 1 \
#

set -e

show_usage() {
    cat << EOF
Usage: $0 --model MODEL_PATH --data DATA_FILE [OPTIONS]

Two-stage layer-selective training (LST) first trains bottom layers, then top layers.

Required arguments:
  --model PATH              Path to the pretrained model
  --data PATH               Path to ShareGPT JSONL/JSON training data

LST layer options:
  --bottom-layers N         Bottom layers trained in stage 1 (default: 4)
  --top-layers N            Top layers trained in stage 2 (default: 16)

Dataset options:
  --dataset-name NAME      Dataset name (default: derived from file name)
  --template NAME           Conversation template (default: qwen3_nothink)

Training hyperparameters:
  --epochs N                Training epochs per stage (default: 1)
  --lr FLOAT                Learning rate (default: 1e-5)
  --batch-size N            Batch size per device (default: 1)
  --grad-accum N            Gradient accumulation steps (default: 16)
  --cutoff-len N            Maximum sequence length (default: 4096)
  --warmup-ratio FLOAT      Warmup ratio (default: 0.03)
  --weight-decay FLOAT      Weight decay (default: 0.0)
  --lr-scheduler TYPE       Learning-rate scheduler (default: cosine_with_min_lr)

Optimization options:
  --flash-attn TYPE         Flash Attention (default: fa2)
  --no-gradient-checkpoint  Disable gradient checkpointing
  --no-plot-loss            Disable loss plotting

Output options:
  --output-base DIR         Output root (default: lst_outputs)
  --job-name NAME           Run name (default: generated automatically)

Other:
  --tokenized-path PATH     Optional tokenized-data path passed to generate_training_yaml.py
  --help                    Show this help message

Examples:
  $0 --model /path/to/model --data /path/to/data.jsonl --bottom-layers 4 --top-layers 16
EOF
}

# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FT_SCRIPTS_DIR="${PROJECT_ROOT}/ft_scripts"
DATA_PREPARE_SCRIPT="${FT_SCRIPTS_DIR}/data_prepare.py"
GENERATE_YAML_SCRIPT="${FT_SCRIPTS_DIR}/generate_training_yaml.py"
DEEPSPEED_CONFIG="${FT_SCRIPTS_DIR}/ds_z3_config.json"

LLAMA_FACTORY_DATA_DIR="${FT_SCRIPTS_DIR}"
DATASET_INFO_PATH="${LLAMA_FACTORY_DATA_DIR}/dataset_info.json"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-}"
CONDA_ENV="${CONDA_ENV:-}"

# ============================================================
# Default arguments
# ============================================================

MODEL_PATH=""
DATA_FILE=""
DATASET_NAME=""
TEMPLATE="qwen3"

BOTTOM_LAYERS=4
TOP_LAYERS=16
NUM_TRAIN_EPOCHS=1
LEARNING_RATE="1e-5"
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
CUTOFF_LEN=4096
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
LR_SCHEDULER="cosine_with_min_lr"
LR_SCHEDULER_KWARGS='{"min_lr": 2.0e-6}'

FLASH_ATTN="fa2"
GRADIENT_CHECKPOINTING="true"
PLOT_LOSS="true"

LOGGING_STEPS=10
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT_STAGE1=0
SAVE_TOTAL_LIMIT_STAGE2=0


OUTPUT_BASE_DIR="${PROJECT_ROOT}/lst_outputs"
JOB_NAME=""
TOKENIZED_PATH=""

# ============================================================
# Argument parsing
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --bottom-layers)
            BOTTOM_LAYERS="$2"
            shift 2
            ;;
        --top-layers)
            TOP_LAYERS="$2"
            shift 2
            ;;
        --epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --cutoff-len)
            CUTOFF_LEN="$2"
            shift 2
            ;;
        --warmup-ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --lr-scheduler)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --flash-attn)
            FLASH_ATTN="$2"
            shift 2
            ;;
        --no-gradient-checkpoint)
            GRADIENT_CHECKPOINTING="false"
            shift
            ;;
        --no-plot-loss)
            PLOT_LOSS="false"
            shift
            ;;
        --output-base)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --tokenized-path)
            TOKENIZED_PATH="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================
# Argument validation
# ============================================================

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    show_usage
    exit 1
fi

if [ -z "$DATA_FILE" ]; then
    echo "Error: --data is required"
    show_usage
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model directory not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: data file not found: $DATA_FILE"
    exit 1
fi

# ============================================================
# Generate names and paths
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")

if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME=$(basename "$DATA_FILE" | sed 's/\.jsonl$//' | sed 's/\.json$//')
fi

LST_OUTPUT_BASE="${OUTPUT_BASE_DIR}/${MODEL_NAME}_LST_template-${TEMPLATE}"
SAVE_MODEL1="${LST_OUTPUT_BASE}/${DATASET_NAME}_b${BOTTOM_LAYERS}_epoch${NUM_TRAIN_EPOCHS}_${TIMESTAMP}"
SAVE_MODEL2="${LST_OUTPUT_BASE}/${DATASET_NAME}_b${BOTTOM_LAYERS}_t${TOP_LAYERS}_epoch${NUM_TRAIN_EPOCHS}_${TIMESTAMP}"
OUTPUT_DIR="$SAVE_MODEL2"

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="lst_${MODEL_NAME}_${DATASET_NAME}_${TIMESTAMP}"
fi

# Store the launcher in the stage-2 output directory.
mkdir -p "$OUTPUT_DIR"
TRAIN_SCRIPT="${OUTPUT_DIR}/run_lst.sh"

# ============================================================
# Print the configuration.
# ============================================================

echo "============================================================"
echo "LLaMA-Factory two-stage layer-selective training (LST) configuration"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_FILE"
echo "Dataset name: $DATASET_NAME"
echo "Template: $TEMPLATE"
echo "Stage 1 bottom layers: $BOTTOM_LAYERS"
echo "Stage 2 top layers: $TOP_LAYERS"
echo "Epochs per stage: $NUM_TRAIN_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Sequence length: $CUTOFF_LEN"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Stage 1 output: $SAVE_MODEL1"
echo "Stage 2 output: $SAVE_MODEL2"
echo "Run name: $JOB_NAME"
[ -n "$TOKENIZED_PATH" ] && echo "Tokenizer: $TOKENIZED_PATH"
echo "============================================================"
echo ""

# ============================================================
# Generate the LST script for data preparation and both stages.
# ============================================================

PLOT_LOSS_FLAG=""
[ "$PLOT_LOSS" = "true" ] && PLOT_LOSS_FLAG="--plot_loss"

GRADIENT_CHECKPOINT_FLAG=""
[ "$GRADIENT_CHECKPOINTING" = "true" ] && GRADIENT_CHECKPOINT_FLAG="--gradient_checkpointing"

cat > "$TRAIN_SCRIPT" << SCRIPT_END
#!/bin/bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=True
export DISABLE_VERSION_CHECK=1
export NCCL_DEBUG=WARN

if [ -n "CONDA_ACTIVATE_PLACEHOLDER" ] && [ -n "CONDA_ENV_PLACEHOLDER" ]; then
    source CONDA_ACTIVATE_PLACEHOLDER CONDA_ENV_PLACEHOLDER
fi

MODEL_PATH_PLACEHOLDER
DATA_FILE_PLACEHOLDER
DATASET_NAME_PLACEHOLDER
LLAMA_FACTORY_DATA_DIR_PLACEHOLDER
DATASET_INFO_PATH_PLACEHOLDER
DATA_PREPARE_SCRIPT_PLACEHOLDER
GENERATE_YAML_SCRIPT_PLACEHOLDER
SAVE_MODEL1_PLACEHOLDER
SAVE_MODEL2_PLACEHOLDER
TEMPLATE_PLACEHOLDER
BOTTOM_LAYERS_PLACEHOLDER
TOP_LAYERS_PLACEHOLDER
NUM_TRAIN_EPOCHS_PLACEHOLDER
LEARNING_RATE_PLACEHOLDER
PER_DEVICE_TRAIN_BATCH_SIZE_PLACEHOLDER
GRADIENT_ACCUMULATION_STEPS_PLACEHOLDER
CUTOFF_LEN_PLACEHOLDER
WARMUP_RATIO_PLACEHOLDER
WEIGHT_DECAY_PLACEHOLDER
LR_SCHEDULER_PLACEHOLDER
LR_SCHEDULER_KWARGS_PLACEHOLDER
FLASH_ATTN_PLACEHOLDER
DEEPSPEED_CONFIG_PLACEHOLDER
LOGGING_STEPS_PLACEHOLDER
SAVE_STEPS_PLACEHOLDER
SAVE_TOTAL_LIMIT_STAGE1_PLACEHOLDER
SAVE_TOTAL_LIMIT_STAGE2_PLACEHOLDER
PLOT_LOSS_FLAG_PLACEHOLDER
GRADIENT_CHECKPOINT_FLAG_PLACEHOLDER

mkdir -p "\${LLAMA_FACTORY_DATA_DIR}"
if [ ! -f "\${DATASET_INFO_PATH}" ] || [ ! -s "\${DATASET_INFO_PATH}" ]; then
    echo '{}' > "\${DATASET_INFO_PATH}"
fi

echo "============================================================"
echo "Prepare data"
echo "============================================================"
python "\${DATA_PREPARE_SCRIPT}" \\
    --data_file "\${DATA_FILE}" \\
    --dataset_name "\${DATASET_NAME}" \\
    --dataset_info_path "\${DATASET_INFO_PATH}"

echo ""
echo "============================================================"
echo "Stage 1: train bottom \${BOTTOM_LAYERS} layers"
echo "============================================================"
FILE1="\${SAVE_MODEL1}/config.json"
if [ ! -f "\$FILE1" ]; then
    rm -rf "\${SAVE_MODEL1}"
    python "\${GENERATE_YAML_SCRIPT}" \\
        --model_path "\${MODEL_PATH}" \\
        --dataset "\${DATASET_NAME}" \\
        --dataset_dir "\${LLAMA_FACTORY_DATA_DIR}" \\
        --output_dir "\${SAVE_MODEL1}" \\
        --template "\${TEMPLATE}" \\
        --num_train_epochs \${NUM_TRAIN_EPOCHS} \\
        --learning_rate \${LEARNING_RATE} \\
        --per_device_train_batch_size \${PER_DEVICE_TRAIN_BATCH_SIZE} \\
        --gradient_accumulation_steps \${GRADIENT_ACCUMULATION_STEPS} \\
        --cutoff_len \${CUTOFF_LEN} \\
        --warmup_ratio \${WARMUP_RATIO} \\
        --weight_decay \${WEIGHT_DECAY} \\
        --lr_scheduler_type "\${LR_SCHEDULER}" \\
        --lr_scheduler_kwargs "\${LR_SCHEDULER_KWARGS}" \\
        --flash_attn "\${FLASH_ATTN}" \\
        --deepspeed "\${DEEPSPEED_CONFIG}" \\
        --logging_steps \${LOGGING_STEPS} \\
        --save_steps \${SAVE_STEPS} \\
        --save_total_limit \${SAVE_TOTAL_LIMIT_STAGE1} \\
        --trainable_layers \${BOTTOM_LAYERS} \\
        \${PLOT_LOSS_FLAG} \\
        \${GRADIENT_CHECKPOINT_FLAG}
    # llamafactory-cli train "\${SAVE_MODEL1}/training.yaml"
    llamafactory-cli train "\${SAVE_MODEL1}/training.yaml"
    echo "Stage 1 complete."
else
    echo "Stage 1 already exists; skipping."
fi

echo ""
echo "============================================================"
echo "Stage 2: train top \${TOP_LAYERS} layers"
echo "============================================================"
FILE2="\${SAVE_MODEL2}/config.json"
if [ ! -f "\$FILE2" ]; then
    rm -rf "\${SAVE_MODEL2}"
    python "\${GENERATE_YAML_SCRIPT}" \\
        --model_path "\${SAVE_MODEL1}" \\
        --dataset "\${DATASET_NAME}" \\
        --dataset_dir "\${LLAMA_FACTORY_DATA_DIR}" \\
        --output_dir "\${SAVE_MODEL2}" \\
        --template "\${TEMPLATE}" \\
        --num_train_epochs \${NUM_TRAIN_EPOCHS} \\
        --learning_rate \${LEARNING_RATE} \\
        --per_device_train_batch_size \${PER_DEVICE_TRAIN_BATCH_SIZE} \\
        --gradient_accumulation_steps \${GRADIENT_ACCUMULATION_STEPS} \\
        --cutoff_len \${CUTOFF_LEN} \\
        --warmup_ratio \${WARMUP_RATIO} \\
        --weight_decay \${WEIGHT_DECAY} \\
        --lr_scheduler_type "\${LR_SCHEDULER}" \\
        --lr_scheduler_kwargs "\${LR_SCHEDULER_KWARGS}" \\
        --flash_attn "\${FLASH_ATTN}" \\
        --deepspeed "\${DEEPSPEED_CONFIG}" \\
        --logging_steps \${LOGGING_STEPS} \\
        --save_steps \${SAVE_STEPS} \\
        --save_total_limit \${SAVE_TOTAL_LIMIT_STAGE2} \\
        --top_trainable_layers \${TOP_LAYERS} \\
        \${PLOT_LOSS_FLAG} \\
        \${GRADIENT_CHECKPOINT_FLAG}
    # llamafactory-cli train "\${SAVE_MODEL2}/training.yaml"
    llamafactory-cli train "\${SAVE_MODEL2}/training.yaml"
    echo "Stage 2 complete."
else
    echo "Stage 2 already exists; skipping."
fi

echo ""
echo "============================================================"
echo "Two-stage LST complete"
echo "============================================================"
echo "Stage1: \${SAVE_MODEL1}"
echo "Stage2: \${SAVE_MODEL2}"
SCRIPT_END

# Replace placeholders with variables used by the generated script.
sed -i "s|CONDA_ACTIVATE_PLACEHOLDER|$CONDA_ACTIVATE|g" "$TRAIN_SCRIPT"
sed -i "s|CONDA_ENV_PLACEHOLDER|$CONDA_ENV|g" "$TRAIN_SCRIPT"
sed -i "s|MODEL_PATH_PLACEHOLDER|MODEL_PATH=\"$MODEL_PATH\"|g" "$TRAIN_SCRIPT"
sed -i "s|DATA_FILE_PLACEHOLDER|DATA_FILE=\"$DATA_FILE\"|g" "$TRAIN_SCRIPT"
sed -i "s|DATASET_NAME_PLACEHOLDER|DATASET_NAME=\"$DATASET_NAME\"|g" "$TRAIN_SCRIPT"
sed -i "s|LLAMA_FACTORY_DATA_DIR_PLACEHOLDER|LLAMA_FACTORY_DATA_DIR=\"$LLAMA_FACTORY_DATA_DIR\"|g" "$TRAIN_SCRIPT"
sed -i "s|DATASET_INFO_PATH_PLACEHOLDER|DATASET_INFO_PATH=\"$DATASET_INFO_PATH\"|g" "$TRAIN_SCRIPT"
sed -i "s|DATA_PREPARE_SCRIPT_PLACEHOLDER|DATA_PREPARE_SCRIPT=\"$DATA_PREPARE_SCRIPT\"|g" "$TRAIN_SCRIPT"
sed -i "s|GENERATE_YAML_SCRIPT_PLACEHOLDER|GENERATE_YAML_SCRIPT=\"$GENERATE_YAML_SCRIPT\"|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_MODEL1_PLACEHOLDER|SAVE_MODEL1=\"$SAVE_MODEL1\"|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_MODEL2_PLACEHOLDER|SAVE_MODEL2=\"$SAVE_MODEL2\"|g" "$TRAIN_SCRIPT"
sed -i "s|TEMPLATE_PLACEHOLDER|TEMPLATE=\"$TEMPLATE\"|g" "$TRAIN_SCRIPT"
sed -i "s|BOTTOM_LAYERS_PLACEHOLDER|BOTTOM_LAYERS=$BOTTOM_LAYERS|g" "$TRAIN_SCRIPT"
sed -i "s|TOP_LAYERS_PLACEHOLDER|TOP_LAYERS=$TOP_LAYERS|g" "$TRAIN_SCRIPT"
sed -i "s|NUM_TRAIN_EPOCHS_PLACEHOLDER|NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS|g" "$TRAIN_SCRIPT"
sed -i "s|LEARNING_RATE_PLACEHOLDER|LEARNING_RATE=\"$LEARNING_RATE\"|g" "$TRAIN_SCRIPT"
sed -i "s|PER_DEVICE_TRAIN_BATCH_SIZE_PLACEHOLDER|PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE|g" "$TRAIN_SCRIPT"
sed -i "s|GRADIENT_ACCUMULATION_STEPS_PLACEHOLDER|GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|CUTOFF_LEN_PLACEHOLDER|CUTOFF_LEN=$CUTOFF_LEN|g" "$TRAIN_SCRIPT"
sed -i "s|WARMUP_RATIO_PLACEHOLDER|WARMUP_RATIO=$WARMUP_RATIO|g" "$TRAIN_SCRIPT"
sed -i "s|WEIGHT_DECAY_PLACEHOLDER|WEIGHT_DECAY=$WEIGHT_DECAY|g" "$TRAIN_SCRIPT"
sed -i "s|LR_SCHEDULER_PLACEHOLDER|LR_SCHEDULER=\"$LR_SCHEDULER\"|g" "$TRAIN_SCRIPT"
sed -i "s|LR_SCHEDULER_KWARGS_PLACEHOLDER|LR_SCHEDULER_KWARGS='$LR_SCHEDULER_KWARGS'|g" "$TRAIN_SCRIPT"
sed -i "s|FLASH_ATTN_PLACEHOLDER|FLASH_ATTN=\"$FLASH_ATTN\"|g" "$TRAIN_SCRIPT"
sed -i "s|DEEPSPEED_CONFIG_PLACEHOLDER|DEEPSPEED_CONFIG=\"$DEEPSPEED_CONFIG\"|g" "$TRAIN_SCRIPT"
sed -i "s|LOGGING_STEPS_PLACEHOLDER|LOGGING_STEPS=$LOGGING_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_STEPS_PLACEHOLDER|SAVE_STEPS=$SAVE_STEPS|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_TOTAL_LIMIT_STAGE1_PLACEHOLDER|SAVE_TOTAL_LIMIT_STAGE1=$SAVE_TOTAL_LIMIT_STAGE1|g" "$TRAIN_SCRIPT"
sed -i "s|SAVE_TOTAL_LIMIT_STAGE2_PLACEHOLDER|SAVE_TOTAL_LIMIT_STAGE2=$SAVE_TOTAL_LIMIT_STAGE2|g" "$TRAIN_SCRIPT"
sed -i "s|PLOT_LOSS_FLAG_PLACEHOLDER|PLOT_LOSS_FLAG=\"$PLOT_LOSS_FLAG\"|g" "$TRAIN_SCRIPT"
sed -i "s|GRADIENT_CHECKPOINT_FLAG_PLACEHOLDER|GRADIENT_CHECKPOINT_FLAG=\"$GRADIENT_CHECKPOINT_FLAG\"|g" "$TRAIN_SCRIPT"

chmod +x "$TRAIN_SCRIPT"
echo "LST training script generated: $TRAIN_SCRIPT"
echo ""

# ============================================================
# Run locally in the current environment
# ============================================================

echo "Starting training: $TRAIN_SCRIPT"
bash "$TRAIN_SCRIPT"
