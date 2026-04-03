#!/bin/bash
export GITEEAI_API_KEY=VFXOXC7PFUFUCLMF0XYQO9VU684DHL3PD7LDTM3P
export GITEEAI_API_KEY=PEHIMFEDMJLD2YUCGHPVQJFNZERA9JWVWXLIYS1T
export CUDA_VISIBLE_DEVICES=5

# ============================================================
# 任务列表
# ============================================================

kernel_path=(
  "KernelBench/level1/19_ReLU.py"
  "KernelBench/level1/20_LeakyReLU.py"
  "KernelBench/level1/21_Sigmoid.py"
  "KernelBench/level1/22_Tanh.py"
  "KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"
  "KernelBench/level2/2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py"
  "KernelBench/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py"
  "KernelBench/level2/4_Conv2d_Mish_Mish.py"
  "KernelBench/level3/1_MLP.py"
  "KernelBench/level3/2_ShallowWideMLP.py"
  "KernelBench/level3/3_DeepNarrowMLP.py"
  "KernelBench/level3/4_LeNet5.py"
)

command_list=()
for kp in "${kernel_path[@]}"; do
  command_list+=("python main_memory_latest.py $kp --gpu RTX5090 --profiling_mode timing_only --server_type giteeai --model_name DeepSeek-V3 --round 20 --work_dir run --device 0")
done

n_jobs=4
delay_time=20
parallel --jobs ${n_jobs} --delay ${delay_time} ::: "${command_list[@]}"