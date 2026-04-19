# MAMA项目配置文件 - 多智能体记忆提取实验

# 实验基础配置
EXPERIMENT_CONFIG = {
    "num_agents": 6,                    # 智能体总数
    "target_idx": 0,                    # 目标节点索引
    "attacker_idx": 5,                # 攻击者节点索引列表
    "graph_types": ["star_pure", "star_ring", "circle", "tree", "complete", "chain"],  # 图类型
    "model": "llama3.1-70b",           # 默认模型（Llama）
    "max_rounds": 10                    # 最大轮数
}

# LLM配置
LLM_CONFIG = {
    # Llama模型配置
    "llama3.1-70b": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "llama3.1-8b": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "llama3-70b": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "llama3-8b": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    # Claude模型配置
    "claude-3.7-sonnet": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    # DeepSeek模型配置
    "deepseek-v3.1": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    # OpenAI模型配置
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "gpt-4": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 15,
    },
    "gpt-5.2": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
        "max_retries": 1048576,
    },
    "gpt-5-nano": {
        # "temperature": 0.7,
        # "max_tokens": 512,
        # "top_p": 0.9,
        "max_retries": 1048576,
    },
}

# 输出配置
OUTPUT_CONFIG = {
    "base_dir": "output",               # 基础输出目录
    "results_dir": "results",           # 结果目录
    "log_level": "INFO",                # 日志级别
} 