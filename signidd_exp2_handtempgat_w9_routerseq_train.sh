#!/bin/bash
# Exp2 ablation: w9 baseline, sequence-level MoE router (masked mean + timestep)
# Usage: bash signidd_exp2_handtempgat_w9_routerseq_train.sh [GPU_ID]
GPU_ID=${1:-0}
unset LD_LIBRARY_PATH
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES="${GPU_ID}" /home/amax/miniconda3/envs/env_slp/bin/python -u __main__.py \
  train ./Configs/signidd_exp2_handtempgat_w9_routerseq.yaml --gpu_id 0
