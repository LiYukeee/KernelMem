#!/bin/bash
# ============================================================
# run.sh — 自动以 sudo 运行，完整保留 conda / CUDA 环境
# 直接运行即可：bash run.sh
# ============================================================

export GITEEAI_API_KEY=VFXOXC7PFUFUCLMF0XYQO9VU684DHL3PD7LDTM3P
export CUDA_VISIBLE_DEVICES=1

# 将 CUDA 12.8 工具链加入 PATH（ncu / nsys 都在这里）
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}"

PYTHON=/home/liyk/miniconda3/envs/CudaForge/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 若不是 root，则带完整环境重新以 sudo 执行本脚本 ----
if [ "$EUID" -ne 0 ]; then
    exec sudo env \
        PATH="$PATH" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        GITEEAI_API_KEY="$GITEEAI_API_KEY" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        HOME="$HOME" \
        bash "$0" "$@"
fi

# ---- 切换到脚本所在目录，保证相对路径正确 ----
cd "$SCRIPT_DIR"

# ============================================================
# 任务列表
# ============================================================

"$PYTHON" main_memory_latest.py KernelBench/level1/19_ReLU.py \
  --gpu RTX5090 \
  --server_type giteeai \
  --model_name DeepSeek-V3 \
  --round 20 \
  --work_dir run \
  --device 0

# "$PYTHON" main_memory_latest.py KernelBench/level3/4_LeNet5.py \
#   --gpu RTX5090 \
#   --server_type giteeai \
#   --model_name DeepSeek-V3 \
#   --round 100 \
#   --work_dir run \
#   --device 0
