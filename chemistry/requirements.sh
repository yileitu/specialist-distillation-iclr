#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install \
  accelerate \
  numpy \
  openpyxl \
  pandas \
  pyarrow \
  rdchiral \
  rdkit \
  torch \
  torchaudio \
  torchvision \
  tqdm \
  transformers \
  vllm

# Install LLaMA-Factory separately by following its upstream installation guide:
# https://github.com/hiyouga/LLaMA-Factory
