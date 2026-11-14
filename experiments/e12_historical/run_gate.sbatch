#!/bin/bash
#SBATCH -J P014-e12a-gate -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 00:45:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
# usage: sbatch run_gate.sbatch [model]   runs the gate for one model
set -e
source ~/pb/env.sh
cd ~/pb
HF_HUB_OFFLINE=1 python experiments/e12_historical/gate.py --model "${1:-Qwen/Qwen2.5-7B-Instruct}"
