# MAMA configuration - multi-agent memory extraction experiments

# Baseline experiment settings
EXPERIMENT_CONFIG = {
    "num_agents": 6,                   # Total number of agents in the network
    "target_idx": 0,                   # Index of the agent holding the private memory
    "attacker_idx": 5,                 # Index of the agent attempting the extraction
    "graph_types": ["star_pure", "star_ring", "circle", "tree", "complete", "chain"],
    "model": "llama3.1-70b",           # Default model
    "max_rounds": 10                   # Maximum number of RelCom rounds
}

# Per-model generation settings
#
# max_retries bounds the exponential-backoff retry loop in llm_interface.py.
# It used to be 1048576: with the 5 s backoff cap that turns a persistent API
# outage into a silent multi-week hang. Eight attempts is roughly 30 s of total
# backoff, enough to ride out ordinary rate limiting; raise it if you are
# running against a heavily throttled endpoint.
LLM_CONFIG = {
    # Llama
    "llama3.1-70b": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    # Claude
    "claude-3.7-sonnet": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    # DeepSeek
    "deepseek-v3.1": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    # OpenAI
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "max_retries": 8,
    },
    "gpt-4": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "max_retries": 8,
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    "gpt-5.2": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 8,
    },
    "gpt-5-nano": {
        # "temperature": 0.7,
        # "max_tokens": 512,
        # "top_p": 0.9,
        "max_retries": 8,
    },
}

# Output settings
OUTPUT_CONFIG = {
    "base_dir": "output",              # Base output directory
    "results_dir": "results",          # Results directory
    "log_level": "INFO",               # Logging level
} 