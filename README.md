# MAMA: Multi-Agent Memory Attack

A research system for simulating memory extraction attacks in multi-agent environments using large language models.


## Quick Start

### Prerequisites

```bash
conda env create -f environment.yml
conda activate mama
```

### Running Experiments

```bash
./run.py
./run_all.py
```

### Key Parameters

- `--dataset-path`: Path to CSV dataset with sensitive information
- `--model`: LLM model to use (default: llama3.1-70b)
- `--num-agents`: Number of agents in the network (default: 6)
- `--graph-type`: Network topology: complete, tree, star, circle, chain (default: star)
- `--max-rounds`: Maximum conversation rounds (default: 10)

## Experiment Flow

1. **Initialization**: Create agent network with target and attacker nodes
2. **Memory Implantation**: Inject sensitive information into target agent
3. **Genesis Phase**: All agents generate initial responses
4. **RelCom Rounds**: Multi-round conversation with memory extraction attempts
5. **Success Evaluation**: Measure attack effectiveness and information leakage

## Output

Results are saved to `logs/` with:
- Attack success rate
- Per-sample extraction results
- Round-by-round progress tracking
