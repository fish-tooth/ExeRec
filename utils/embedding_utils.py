"""嵌入相关的工具函数"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_embedding_json(path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    加载 JSON 格式的嵌入文件
    
    Returns:
        (key -> np.ndarray, embedding_dim)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    if not raw:
        return {}, 0
    
    result = {}
    dim = None
    for k, v in raw.items():
        arr = np.asarray(v, dtype=np.float32)
        if dim is None:
            dim = arr.shape[0]
        result[k] = arr
    return result, dim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """单对向量余弦相似度"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def batch_cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    query: [D] 或 [Q, D]
    matrix: [N, D]
    返回: [N] 或 [Q, N]
    """
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    
    if query.ndim == 1:
        q_norm = np.linalg.norm(query) + 1e-9
        m_norms = np.linalg.norm(matrix, axis=1) + 1e-9
        return (matrix @ query) / (m_norms * q_norm)
    else:
        q_norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-9
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return (query / q_norms) @ (matrix / m_norms).T


def normalize(v: np.ndarray) -> np.ndarray:
    """L2 归一化"""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n
