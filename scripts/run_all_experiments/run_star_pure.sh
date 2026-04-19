#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# 运行MAMA多智能体实验 - star_pure图结构测试
mkdir -p logs

model="${MAMA_MODEL:-deepseek-v3.1}"
exp_number="${MAMA_EXP_NUMBER:-first_experiments}"
max_rounds="${MAMA_MAX_ROUNDS:-10}"
question_num="${MAMA_QUESTION_NUM:-25}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

graph_types=("star_pure")
num_agents_list=(4 5 6)

# 创建logs/all_entities目录结构
mkdir -p logs/all_entities/${model}/${exp_number}
for graph_type in "${graph_types[@]}"; do
  mkdir -p "logs/all_entities/${model}/${exp_number}/${graph_type}"
done

# 运行所有组合的实验
for graph_type in "${graph_types[@]}"; do
  for num_agents in "${num_agents_list[@]}"; do
    case "$num_agents" in
      4|5|6)
        target_attacker_pairs=("0,1" "1,0" "1,2")
        ;;
      *)
        echo "No predefined target/attacker pairs for num_agents=${num_agents}, skip."
        continue
        ;;
    esac

    if (( ${#target_attacker_pairs[@]} == 0 )); then
      echo "No valid pairs for num_agents=${num_agents}, skip."
      continue
    fi

    for pair in "${target_attacker_pairs[@]}"; do
      IFS=',' read -r target_idx attacker_idx <<< "${pair}"

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

