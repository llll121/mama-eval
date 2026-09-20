#!/bin/bash
# 本地批量运行所有实验脚本
#
# 使用方法：
#   cd mama-eval/scripts/run_all_experiments
#   ./submit_all.sh
#
# 或者手动逐个运行：
#   bash run_tree.sh
#   bash run_star_pure.sh
#   bash run_star_ring.sh
#   bash run_complete.sh
#   bash run_circle.sh
#   bash run_chain.sh

set -euo pipefail

# 切换到脚本目录
cd "$(dirname "$0")"

# 统一设置实验参数（会覆盖子脚本中的默认值）
export MAMA_MODEL="deepseek-v3.1"  # llama3.1-70b, deepseek-v3.1, gpt-4o, gpt-4o-mini
export MAMA_EXP_NUMBER="first_experiments"
export MAMA_MAX_ROUNDS=10
export MAMA_QUESTION_NUM=100

# OPENAI_API_KEY is deliberately not exported here: assigning an empty
# value would wipe out a key the caller already exported, and the
# ${OPENAI_API_KEY:-} fallback in each sub-script would then see "".
# Only the gpt-* models need an OpenAI key. The Bedrock-hosted models
# (llama3.1-70b, claude-3.7-sonnet, deepseek-v3.1) authenticate through the
# AWS credential chain instead, so the key is required conditionally.
if [[ "${MAMA_MODEL}" == gpt-* && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: model is '${MAMA_MODEL}' but OPENAI_API_KEY is not set." >&2
  echo "       Run: export OPENAI_API_KEY='sk-...'" >&2
  exit 1
fi

echo "=========================================="
echo "Submitting all MAMA experiment jobs..."
echo "=========================================="
echo "Configuration:"
echo "  Model: ${MAMA_MODEL}"
echo "  Exp Number: ${MAMA_EXP_NUMBER}"
echo "  Max Rounds: ${MAMA_MAX_ROUNDS}"
echo "  Question Num: ${MAMA_QUESTION_NUM}"
echo "=========================================="

# 定义所有要提交的脚本
scripts=(
    "run_tree.sh"
    "run_star_pure.sh"
    "run_star_ring.sh"
    "run_complete.sh"
    "run_circle.sh"
    "run_chain.sh"
)

# 运行每个脚本
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
        echo "  OK: $script completed"
    else
        echo "  ERROR: Script not found: $script"
    fi
done

echo "=========================================="
echo "All scripts completed!"
echo "=========================================="

