#!/bin/bash
# Evaluate FastVLA on one RoboTwin task with the official harness (100 episodes).
# Run from the RoboTwin root after starting the server (scripts/serve.py):
#
#   FASTVLA_ROOT=/path/to/fastvla bash policy/FastVLA/eval.sh <task_name> <demo_clean|demo_randomized> [port] [gpu]
#
# The released checkpoint is evaluated with seed 0 (evaluation seeds 100000+) on all 50 tasks
# (assets/robotwin_tasks.txt) in both settings.
set -e
task_name=${1:?task_name}
task_config=${2:?demo_clean or demo_randomized}
port=${3:-8765}
gpu=${4:-0}
: "${FASTVLA_ROOT:?set FASTVLA_ROOT to the FastVLA repository}"
export FASTVLA_ROOT CUDA_VISIBLE_DEVICES=$gpu PYTHONWARNINGS=ignore::UserWarning
python script/eval_policy.py --config policy/FastVLA/deploy_policy.yml \
    --overrides \
    --task_name "$task_name" \
    --task_config "$task_config" \
    --ckpt_setting 0 \
    --seed 0 \
    --policy_name FastVLA \
    --port "$port"
