#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# Run the MAMA multi-agent experiments on the chain topology
mkdir -p logs

model="${MAMA_MODEL:-deepseek-v3.1}"
exp_number="${MAMA_EXP_NUMBER:-first_experiments}"
max_rounds="${MAMA_MAX_ROUNDS:-10}"
question_num="${MAMA_QUESTION_NUM:-25}"
# Only the gpt-* models need an OpenAI key. The Bedrock-hosted models
# (llama3.1-70b, claude-3.7-sonnet, deepseek-v3.1) authenticate through the
# AWS credential chain instead, so the key is required conditionally.
if [[ "${model}" == gpt-* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: model is '${model}' but OPENAI_API_KEY is not set." >&2
  echo "       Run: export OPENAI_API_KEY='sk-...'" >&2
  exit 1
fi

graph_types=("chain")
num_agents_list=(4 5 6)
# num_agents_list=(6)

# Create the logs/all_entities directory tree
mkdir -p logs/all_entities/${model}/${exp_number}
for graph_type in "${graph_types[@]}"; do
  mkdir -p "logs/all_entities/${model}/${exp_number}/${graph_type}"
done

# Run every parameter combination
for graph_type in "${graph_types[@]}"; do
  for num_agents in "${num_agents_list[@]}"; do
    case "$num_agents" in
      4)
        target_attacker_pairs=("0,1" "0,2" "0,3" "1,0" "1,2" "1,3")
        ;;
      5)
        target_attacker_pairs=("0,1" "0,2" "0,3" "0,4" "1,0" "1,2" "1,3" "1,4" "2,0" "2,1" "2,3" "2,4")
        ;;
      6)
        target_attacker_pairs=("0,1" "0,2" "0,3" "0,4" "0,5" "1,0" "1,2" "1,3" "1,4" "1,5" "2,0" "2,1" "2,3" "2,4" "2,5")
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

