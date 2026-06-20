"""
群体先验模块
- 对训练集学生做 GMM 聚类
- 推理时对新学生计算软分配,得到群体表征
- 与个体表征做 α 融合
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("student.group_prior")


class GroupPriorModule:
    """
    群体先验
    
    训练阶段: 在训练集学生的特征上拟合 GMM
    推理阶段: 计算新学生的群体先验向量并融合
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        fusion_alpha_mode: str = "adaptive",
        fusion_alpha_fixed: float = 0.7,
    ):
        self.n_clusters = n_clusters
        self.fusion_alpha_mode = fusion_alpha_mode
        self.fusion_alpha_fixed = fusion_alpha_fixed
        
        self.gmm = None
        self.cluster_centroids: Optional[np.ndarray] = None  # [K, D]
        self.fitted = False
    
    def fit(self, student_features: np.ndarray):
        """
        Args:
            student_features: [N, D] 训练集学生的特征矩阵
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            logger.warning("sklearn not installed; using simple KMeans fallback")
            return self._fit_kmeans_fallback(student_features)
        
        n = student_features.shape[0]
        k = min(self.n_clusters, max(2, n // 10))
        self.gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            max_iter=200,
        )
        self.gmm.fit(student_features)
        self.cluster_centroids = self.gmm.means_.astype(np.float32)
        self.fitted = True
        logger.info(f"GMM fitted with {k} clusters on {n} students")
    
    def _fit_kmeans_fallback(self, X: np.ndarray):
        """sklearn 不可用时的简单 KMeans"""
        n, d = X.shape
        k = min(self.n_clusters, max(2, n // 10))
        rng = np.random.RandomState(42)
        idx = rng.choice(n, k, replace=False)
        centers = X[idx].copy()
        for _ in range(50):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.stack([
                X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centers[j]
                for j in range(k)
            ])
            if np.allclose(new_centers, centers, atol=1e-4):
                break
            centers = new_centers
        self.cluster_centroids = centers.astype(np.float32)
        self.fitted = True
        logger.info(f"KMeans fitted with {k} clusters on {n} students")
    
    def get_group_embedding(self, individual_feature: np.ndarray) -> np.ndarray:
        """
        计算群体先验表征 = sum_k (P(cluster_k | x) * centroid_k)
        """
        if not self.fitted or self.cluster_centroids is None:
            # 未拟合时退化为原表征
            return individual_feature
        
        if self.gmm is not None:
            try:
                probs = self.gmm.predict_proba(individual_feature.reshape(1, -1))[0]
            except Exception:
                probs = self._soft_assign_by_distance(individual_feature)
        else:
            probs = self._soft_assign_by_distance(individual_feature)
        
        group_emb = np.zeros_like(self.cluster_centroids[0])
        for k, p in enumerate(probs):
            group_emb = group_emb + p * self.cluster_centroids[k]
        return group_emb.astype(np.float32)
    
    def _soft_assign_by_distance(self, x: np.ndarray) -> np.ndarray:
        """欧氏距离的 softmax 作为软分配"""
        dists = np.linalg.norm(self.cluster_centroids - x[None, :], axis=1)
        scores = -dists
        scores = scores - scores.max()
        e = np.exp(scores)
        return e / e.sum()
    
    def compute_alpha(
        self,
        interaction_count: int,
        avg_mastery_std: float = 0.0,
    ) -> float:
        """
        计算个体表征的权重 α
        交互少 → α 小(偏群体先验)
        交互多 → α 大(偏个体表征)
        """
        if self.fusion_alpha_mode == "fixed":
            return self.fusion_alpha_fixed
        
        # adaptive: sigmoid(交互次数归一化)
        # 50 次以下 α 偏小, 200 次以上 α 偏大
        from math import exp
        x = (interaction_count - 80) / 40
        alpha = 1 / (1 + exp(-x))
        return float(np.clip(alpha, 0.3, 0.95))
    
    def fuse(
        self,
        individual_feature: np.ndarray,
        interaction_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        融合个体与群体表征
        Returns: (fused, group_emb, alpha)
        """
        group_emb = self.get_group_embedding(individual_feature)
        alpha = self.compute_alpha(interaction_count)
        fused = alpha * individual_feature + (1 - alpha) * group_emb
        return fused.astype(np.float32), group_emb, alpha
    
    # ============ 持久化 ============
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_clusters": self.n_clusters,
                "fusion_alpha_mode": self.fusion_alpha_mode,
                "fusion_alpha_fixed": self.fusion_alpha_fixed,
                "gmm": self.gmm,
                "cluster_centroids": self.cluster_centroids,
                "fitted": self.fitted,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "GroupPriorModule":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_clusters=data["n_clusters"],
            fusion_alpha_mode=data["fusion_alpha_mode"],
            fusion_alpha_fixed=data["fusion_alpha_fixed"],
        )
        obj.gmm = data["gmm"]
        obj.cluster_centroids = data["cluster_centroids"]
        obj.fitted = data["fitted"]
        return obj
