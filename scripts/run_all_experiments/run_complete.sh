#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# 运行MAMA多智能体实验 - 全参数组合测试
mkdir -p logs

model="${MAMA_MODEL:-deepseek-v3.1}"
exp_number="${MAMA_EXP_NUMBER:-first_experiments}"
max_rounds="${MAMA_MAX_ROUNDS:-10}"
question_num="${MAMA_QUESTION_NUM:-25}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

target_idx=0
graph_types=("complete")
# graph_types=("complete" "tree" "star" "circle" "chain")
# num_agents_list=(6)
num_agents_list=(4 5 6)

# 创建目录结构
mkdir -p logs/all_entities/${model}/${exp_number}
for graph_type in "${graph_types[@]}"; do
  mkdir -p "logs/all_entities/${model}/${exp_number}/${graph_type}"
done

# 运行所有组合的实验
for graph_type in "${graph_types[@]}"; do
  for num_agents in "${num_agents_list[@]}"; do
    # 计算attacker-idx的连续序列，从1开始到(num_agents / 2)截止
    for attacker_idx in $(seq 1 $((num_agents / 2))); do
    # for attacker_idx in $((2)); do
      echo "Running: model=${model}, graph=${graph_type}, agents=${num_agents}, target=${target_idx}, attacker=${attacker_idx}"
      
      python3 -u run.py \
        --dataset-path data/llama3.1_num484_nopii.csv \
        --model "${model}" \
        --max-rounds "${max_rounds}" \
        --question-num "${question_num}" \
        --graph-type "${graph_type}" \
        --num-agents "${num_agents}" \
        --target-idx "${target_idx}" \
        --attacker-idx "${attacker_idx}" \
        > "logs/all_entities/${model}/${exp_number}/${graph_type}/experiment_max-rounds=${max_rounds}_question-num=${question_num}_graph=${graph_type}_num-agents=${num_agents}_target-idx=${target_idx}_attacker-idx=${attacker_idx}.log" 2>&1
    done
  done
done

echo "All experiments completed!"