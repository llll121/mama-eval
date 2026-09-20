"""
MAMA utility functions - helpers shared by the experiment driver.
"""

import json
import os
import numpy as np
from llm_interface import LLMFactory
from config import LLM_CONFIG

def get_llm(model_type="llama3.1-70b"):
    """
    Return an LLM interface instance for the given model.
    
    Args:
        model_type: Model name, as listed in LLMFactory.SUPPORTED_MODELS.
    
    Returns:
        An object exposing generate() and chat_generate().
    """
    return LLMFactory.create_llm(model_type)

def create_directory(directory):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_adj(n, graph_type):
    """Build the adjacency matrix for the requested topology."""
    if "complete" in graph_type:
        # Complete graph: every node is connected to every other node
        adj_matrix = np.ones((n, n), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
    elif "tree" in graph_type:
        # Binary tree
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            if left_child < n:
                adj_matrix[i][left_child] = 1
                adj_matrix[left_child][i] = 1
            if right_child < n:
                adj_matrix[i][right_child] = 1
                adj_matrix[right_child][i] = 1
    elif "chain" in graph_type:
        adj_matrix = np.zeros((n, n), dtype=int)
        # Set the values for a chain structure
        for i in range(n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
    elif "star_ring" in graph_type:
        # Star with a ring: node 0 is the hub and the leaves also form a ring
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj_matrix[0][i] = 1
            adj_matrix[i][0] = 1
        for i in range(1, n - 1):
            adj_matrix[i][i + 1] = 1
            adj_matrix[i + 1][i] = 1
        adj_matrix[1][n - 1] = 1
        adj_matrix[n - 1][1] = 1
    elif "star_pure" in graph_type:
        # Pure star: node 0 is the hub and the only point of contact
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj_matrix[0][i] = 1
            adj_matrix[i][0] = 1
    elif "circle" in graph_type:
        # Ring
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj_matrix[i][(i + 1) % n] = 1
            adj_matrix[(i + 1) % n][i] = 1
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    return adj_matrix

def save_experiment_result(result, output_path, json_format=True):
    """Write an experiment result to disk."""
    create_directory(os.path.dirname(output_path))
    with open(output_path, 'w', encoding="utf-8") as f:
        if json_format:
            json.dump(result, f, indent=4, ensure_ascii=False)
        else:
            f.write(str(result) + "\n")

def get_llm_config(model_type):
    """Return the generation settings for a model, defaulting to llama3.1-70b."""
    return LLM_CONFIG.get(model_type, LLM_CONFIG["llama3.1-70b"]) 