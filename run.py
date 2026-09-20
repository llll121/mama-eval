#!/usr/bin/env python3
"""
MAMA: Multi-Agent Memory Attack
Driver for the dataset-based multi-agent memory extraction experiments.
"""

import argparse
import os
from run_mama import run_experiment
import pandas as pd
import ast
import json

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MAMA: Multi-Agent Memory Attack")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset CSV file")
    # Only the dataset-driven experiment is supported.
    parser.add_argument("--model", type=str, default="llama3.1-70b",
                       choices=["llama3.1-70b", "claude-3.7-sonnet", "deepseek-v3.1", "gpt-4o-mini", "gpt-4o", "gpt-5.2", "gpt-5-nano"],
                       help="Model to run the agents with (default: llama3.1-70b)")
    parser.add_argument("--num-agents", type=int, default=6,
                       help="Number of agents in the network (default: 6)")
    parser.add_argument("--target-idx", type=int, default=0,
                       help="Index of the agent holding the private memory (default: 0)")
    parser.add_argument("--attacker-idx", type=int, default=5,
                       help="Index of the agent attempting the extraction (default: 5)")
    parser.add_argument("--max-rounds", type=int, default=10,
                       help="Maximum number of RelCom rounds (default: 10)")
    parser.add_argument("--graph-type", type=str, choices=["complete", "tree", "star_ring", "star_pure", "circle", "chain"],
                       default="star_pure", help="Network topology (default: star_pure)")
    parser.add_argument("--question-num", type=int, default=None,
                       help="Use only the first N rows of the dataset (default: all rows)")
    return parser.parse_args()

def run_dataset_experiment(args):
    """Run one experiment per dataset row, taking the task description from
    the `generated_texts` column and the private memory from `text`."""
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
        # Parse the pii column
        pii_str = str(getattr(row, 'pii'))
        pii_list = ast.literal_eval(pii_str)
        if not pii_list or not isinstance(pii_list, list):
            raise ValueError(f"Invalid pii format in sample {idx}: {pii_str}")
        
        # Collect every entity and its type
        entities = [item['entity'] for item in pii_list]
        types = [item['types'][0] for item in pii_list]  # keep the first type of each item
        
        text = str(getattr(row, 'text'))
        # Every annotated entity must actually occur in the source text
        for entity in entities:
            if entity not in text:
                raise ValueError(f"Entity '{entity}' not found in text for sample {idx}.")
        
        # The whole document is implanted as the target's private memory
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
            attacker_target_types=types,  # every PII type the attacker is asked to obtain
            entities=entities  # every entity used for the success check
        )
        
        # Success information as reported by run_experiment
        success_ratio = result_info['success_ratio']
        extracted_entities = result_info['extracted_entities']
        total_entities = entities  # kept as a list of strings
        
        results.append({
            'sample_idx': idx,
            'entities': entities,
            'types': types,
            'success_ratio': success_ratio,
            'extracted_entities': extracted_entities,
            'total_entities': total_entities,
            'success_round': result_info['success_round']
        })
    
    # Overall success ratio: extracted entities divided by all target entities
    total_extracted_entities = sum(len(r['extracted_entities']) for r in results)
    total_entities = sum(len(r['total_entities']) for r in results)
    total_success_ratio = total_extracted_entities / total_entities if total_entities > 0 else 0
    print(f"\nDataset Attack Success Ratio: {total_success_ratio:.2%} ({total_extracted_entities}/{total_entities})")
    
    # Write the results for every sample in one file
    output_dir = "output/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_summary.json")
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"All results saved to: {output_path}")
    return results

def main():
    """Entry point."""
    args = parse_arguments()
    run_dataset_experiment(args)

if __name__ == "__main__":
    main() 