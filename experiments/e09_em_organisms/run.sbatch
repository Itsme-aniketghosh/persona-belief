#!/bin/bash
#SBATCH -J P008-e09-em -A r.mohammadi -p gpu-short --gres=gpu:a100:1 -c 4 --mem=64G -t 01:00:00
#SBATCH -o /home/%u/pb/logs/%x-%j.out
set -e
source ~/pb/env.sh
cd ~/pb
export HF_HUB_OFFLINE=1
# tiny pass first: one EM adapter, few statements
python experiments/e09_em_organisms/em_probe.py --em ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice \
  --n 12 --n_claim 6 --n_null 50 --out ~/pb/results/tiny
python experiments/e09_em_organisms/em_probe.py "$@"
