#!/usr/bin/env python3
"""
MAMA: Multi-Agent Memory Attack
多智能体记忆提取实验系统
"""

import argparse
import os
from run_mama import run_experiment
import pandas as pd
import ast
import json

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MAMA: Multi-Agent Memory Attack")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集CSV文件路径")
    # 只保留数据集驱动实验相关参数
    parser.add_argument("--model", type=str, default="llama3.1-70b",
                       choices=["llama3.1-70b", "claude-3.7-sonnet", "deepseek-v3.1", "gpt-4o-mini", "gpt-4o", "gpt-5.2", "gpt-5-nano"],
                       help="指定模型类型 (默认: llama3.1-70b)")
    parser.add_argument("--num-agents", type=int, default=6,
                       help="智能体数量 (默认: 6)")
    parser.add_argument("--target-idx", type=int, default=0,
                       help="目标节点索引 (默认: 0)")
    parser.add_argument("--attacker-idx", type=int, default=5,
                       help="攻击者节点索引 (默认: 5)")
    parser.add_argument("--max-rounds", type=int, default=10,
                       help="最大轮数 (默认: 10)")
    parser.add_argument("--graph-type", type=str, choices=["complete", "tree", "star_ring", "star_pure", "circle", "chain"],
                       default="star_pure", help="图类型: complete, tree, star_ring,star_pure, circle, chain (默认: star_pure)")
    parser.add_argument("--question-num", type=int, default=None,
                       help="取数据集的前多少个数据 (默认: 全部数据)")
    return parser.parse_args()

# 删除run_single_test和run_full_experiment等非数据集驱动实验相关函数，只保留run_dataset_experiment和main。

def run_dataset_experiment(args):
    """基于数据集的多轮实验，每行一个实验，task_prompt取自task_backgrounds"""
    print("📚 Running Dataset-Driven MAMA Experiment")
    dataset_path = args.dataset_path
    df = pd.read_csv(dataset_path)
    if args.question_num is not None:
        df = df.head(args.question_num)
        print(f"📊 Using first {args.question_num} samples from dataset")
    else:
        print(f"📊 Using all {len(df)} samples from dataset")
    total_samples = len(df)
    results = []
    
    for idx, row in enumerate(df.itertuples(index=False), 1):
        print(f"\n🔬 Sample {idx}/{total_samples}")
        # 解析pii字段
        pii_str = str(getattr(row, 'pii'))
        pii_list = ast.literal_eval(pii_str)
        if not pii_list or not isinstance(pii_list, list):
            raise ValueError(f"Invalid pii format in sample {idx}: {pii_str}")
        
        # 提取所有entity和type
        entities = [item['entity'] for item in pii_list]
        types = [item['types'][0] for item in pii_list]  # 取每个item的第一个type
        
        text = str(getattr(row, 'text'))
        # 验证所有entity都在text中存在
        for entity in entities:
            if entity not in text:
                raise ValueError(f"Entity '{entity}' not found in text for sample {idx}.")
        
        # 使用完整文本作为sensitive_memory
        sensitive_memory = text
        task_description = str(getattr(row, 'generated_texts'))
        
        result_info = run_experiment(
            num_agents=args.num_agents,
            target_idx=args.target_idx,
            attacker_idx=args.attacker_idx,
            graph_type=args.graph_type,
            sensitive_memory=sensitive_memory,
            model=args.model,
            max_rounds=args.max_rounds,
            task_description=task_description,
            attacker_target_types=types,  # 传递所有type
            entities=entities  # 传递所有entity
        )
        
        # 使用run_experiment返回的成功信息
        success_ratio = result_info['success_ratio']
        extracted_entities = result_info['extracted_entities']
        total_entities = entities  # 保持为字符串列表
        
        results.append({
            'sample_idx': idx,
            'entities': entities,
            'types': types,
            'success_ratio': success_ratio,
            'extracted_entities': extracted_entities,
            'total_entities': total_entities,
            'success_round': result_info['success_round']
        })
    
    # 计算总体成功率：成功提取的entity总数 / 所有entity的总数
    total_extracted_entities = sum(len(r['extracted_entities']) for r in results)
    total_entities = sum(len(r['total_entities']) for r in results)
    total_success_ratio = total_extracted_entities / total_entities if total_entities > 0 else 0
    print(f"\nDataset Attack Success Ratio: {total_success_ratio:.2%} ({total_extracted_entities}/{total_entities})")
    
    # 统一保存所有样本的结果
    output_dir = "output/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_summary.json")
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"All results saved to: {output_path}")
    return results

def main():
    """主函数"""
    args = parse_arguments()
    run_dataset_experiment(args)

if __name__ == "__main__":
    main() 