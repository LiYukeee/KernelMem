#!/bin/bash
export GITEEAI_API_KEY=VFXOXC7PFUFUCLMF0XYQO9VU684DHL3PD7LDTM3P
export CUDA_VISIBLE_DEVICES=1

# ============================================================
# 任务列表
# ============================================================

python main_memory_latest.py KernelBench/level1/19_ReLU.py \
  --gpu RTX5090 \
  --profiling_mode timing_only \
  --server_type giteeai \
  --model_name DeepSeek-V3 \
  --round 20 \
  --work_dir run \
  --device 0

