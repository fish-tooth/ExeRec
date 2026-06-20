"""
经验库
- 增删查改
- 向量检索(基于签名嵌入)
- 置信度演化(贝叶斯式)
- 经验合并(consolidation)
- 持久化
"""
import pickle
import time
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .experience_unit import ExperienceUnit
from utils.embedding_utils import batch_cosine
from utils.logger import get_logger

logger = get_logger("reflection.bank")


class ExperienceBank:
    """
    反思经验库
    线程安全的简单实现(用 RLock)
    """
    
    def __init__(
        self,
        retire_below: float = 0.20,
        dormant_below: float = 0.35,
        decay_per_day: float = 0.99,
    ):
        self.experiences: Dict[str, ExperienceUnit] = {}
        self.retire_below = retire_below
        self.dormant_below = dormant_below
        self.decay_per_day = decay_per_day
        self._lock = threading.RLock()
        
        # 检索矩阵的缓存(惰性重建)
        self._matrix: Optional[np.ndarray] = None
        self._matrix_keys: List[str] = []
        self._dirty = True
    
    # ============ 基本 CRUD ============
    def add(self, exp: ExperienceUnit):
        with self._lock:
            self.experiences[exp.exp_id] = exp
            self._dirty = True
    
    def get(self, exp_id: str) -> Optional[ExperienceUnit]:
        return self.experiences.get(exp_id)
    
    def remove(self, exp_id: str):
        with self._lock:
            self.experiences.pop(exp_id, None)
            self._dirty = True
    
    def size(self) -> int:
        return len(self.experiences)
    
    def active_size(self) -> int:
        return sum(1 for e in self.experiences.values() if e.status == "active")
    
    # ============ 检索 ============
    def _rebuild_matrix(self):
        """重建有效经验的嵌入矩阵"""
        active_exps = [
            e for e in self.experiences.values()
            if e.status == "active" and e.student_sig_embedding is not None
        ]
        if not active_exps:
            self._matrix = None
            self._matrix_keys = []
            self._dirty = False
            return
        
        embs = []
        keys = []
        for e in active_exps:
            emb = e.student_sig_embedding
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            embs.append(emb)
            keys.append(e.exp_id)
        
        # 维度对齐: 不同时期的签名嵌入维度若不一致,截断/填充到最常见维度
        dims = [e.shape[0] for e in embs]
        target_dim = max(set(dims), key=dims.count)
        aligned = []
        for e in embs:
            if e.shape[0] == target_dim:
                aligned.append(e)
            elif e.shape[0] > target_dim:
                aligned.append(e[:target_dim])
            else:
                aligned.append(np.pad(e, (0, target_dim - e.shape[0])))
        
        self._matrix = np.stack(aligned).astype(np.float32)
        self._matrix_keys = keys
        self._dirty = False
    
    def retrieve(
        self,
        query_emb: np.ndarray,
        top_k: int = 3,
        min_confidence: float = 0.4,
        similarity_threshold: float = 0.6,
    ) -> List[ExperienceUnit]:
        """
        基于签名嵌入的相似检索
        """
        with self._lock:
            if self._dirty:
                self._rebuild_matrix()
            
            if self._matrix is None or len(self._matrix_keys) == 0:
                return []
            
            # 维度对齐
            q = query_emb
            if isinstance(q, list):
                q = np.array(q, dtype=np.float32)
            target_dim = self._matrix.shape[1]
            if q.shape[0] > target_dim:
                q = q[:target_dim]
            elif q.shape[0] < target_dim:
                q = np.pad(q, (0, target_dim - q.shape[0]))
            
            scores = batch_cosine(q, self._matrix)
            idx_sorted = np.argsort(-scores)
            
            results = []
            for i in idx_sorted:
                exp_id = self._matrix_keys[i]
                exp = self.experiences.get(exp_id)
                if not exp or exp.status != "active":
                    continue
                if exp.confidence < min_confidence:
                    continue
                if scores[i] < similarity_threshold:
                    break
                exp.last_used_at = time.time()
                exp.use_count += 1
                results.append(exp)
                if len(results) >= top_k:
                    break
            return results
    
    def retrieve_recent(self, top_k: int = 3) -> List[ExperienceUnit]:
        """消融用: 不基于相似度,只取最近的N条(active)"""
        with self._lock:
            active = [e for e in self.experiences.values() if e.status == "active"]
            active.sort(key=lambda e: e.created_at, reverse=True)
            return active[:top_k]
    
    # ============ 置信度更新 ============
    def update_confidence(self, exp_id: str, signal: str):
        """
        signal: 'support' / 'contradict'
        贝叶斯式更新: confidence = (α+1) / (α+β+2)
        """
        with self._lock:
            exp = self.experiences.get(exp_id)
            if not exp:
                return
            
            if signal == "support":
                exp.support_count += 1
            elif signal == "contradict":
                exp.contradict_count += 1
            else:
                return
            
            alpha = exp.support_count + 1
            beta = exp.contradict_count + 1
            base_conf = alpha / (alpha + beta)
            
            # 时间衰减
            days = (time.time() - exp.updated_at) / 86400
            decay = self.decay_per_day ** days
            exp.confidence = float(base_conf * decay)
            exp.updated_at = time.time()
            
            # 状态转移
            if exp.confidence < self.retire_below:
                exp.status = "retired"
            elif exp.confidence < self.dormant_below:
                exp.status = "dormant"
            else:
                exp.status = "active"
            
            self._dirty = True
    
    # ============ 合并(consolidation)============
    def consolidate(
        self,
        similarity_threshold: float = 0.9,
        min_cluster_size: int = 3,
        tag_overlap_threshold: float = 0.6,
        llm_merge_fn=None,
    ):
        """
        合并高度相似的经验
        llm_merge_fn: 可选的 LLM 合并函数 (List[ExperienceUnit]) -> ExperienceUnit
        若为 None,采用规则合并(取置信度最高者 + 累加 support/contradict)
        """
        with self._lock:
            if self._dirty:
                self._rebuild_matrix()
            if self._matrix is None or self._matrix.shape[0] < min_cluster_size:
                return 0
            
            # 简单聚类: 相似度图上的连通分量(贪心)
            n = self._matrix.shape[0]
            sim = self._matrix @ self._matrix.T
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-9
            sim = sim / (norms @ norms.T)
            
            visited = set()
            clusters = []
            for i in range(n):
                if i in visited:
                    continue
                cluster = [i]
                visited.add(i)
                for j in range(i + 1, n):
                    if j in visited:
                        continue
                    if sim[i, j] >= similarity_threshold:
                        cluster.append(j)
                        visited.add(j)
                clusters.append(cluster)
            
            merged_count = 0
            for cluster in clusters:
                if len(cluster) < min_cluster_size:
                    continue
                exps = [self.experiences[self._matrix_keys[i]] for i in cluster]
                
                # 标签重叠度
                if not self._check_tag_overlap(exps, tag_overlap_threshold):
                    continue
                
                if llm_merge_fn:
                    merged = llm_merge_fn(exps)
                else:
                    merged = self._rule_merge(exps)
                
                if merged is None:
                    continue
                
                self.experiences[merged.exp_id] = merged
                for e in exps:
                    if e.exp_id != merged.exp_id:
                        e.status = "retired"
                merged_count += 1
            
            self._dirty = True
            logger.info(f"Consolidated {merged_count} clusters")
            return merged_count
    
    @staticmethod
    def _check_tag_overlap(exps: List[ExperienceUnit], threshold: float) -> bool:
        all_tags = [set(e.lesson_tags) for e in exps]
        if not all_tags or not all_tags[0]:
            return False
        common = set.intersection(*all_tags) if all_tags else set()
        union = set.union(*all_tags) if all_tags else set()
        if not union:
            return False
        return len(common) / len(union) >= threshold
    
    @staticmethod
    def _rule_merge(exps: List[ExperienceUnit]) -> ExperienceUnit:
        """简单规则合并"""
        best = max(exps, key=lambda e: e.confidence)
        merged = ExperienceUnit(
            student_signature=best.student_signature,
            student_sig_embedding=best.student_sig_embedding,
            action=best.action,
            outcome=best.outcome,
            lesson=best.lesson,
            lesson_tags=best.lesson_tags,
            suggested_action_delta=best.suggested_action_delta,
            confidence=float(np.mean([e.confidence for e in exps])),
            support_count=sum(e.support_count for e in exps),
            contradict_count=sum(e.contradict_count for e in exps),
            severity=best.severity,
            source_student_id="merged",
        )
        return merged
    
    # ============ GC ============
    def gc(self):
        """清理 retired 经验(物理删除)"""
        with self._lock:
            to_del = [eid for eid, e in self.experiences.items() if e.status == "retired"]
            for eid in to_del:
                del self.experiences[eid]
            self._dirty = True
            return len(to_del)
    
    # ============ 持久化 ============
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                eid: e.to_serializable() for eid, e in self.experiences.items()
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
    
    def load(self, path: str):
        if not Path(path).exists():
            logger.info(f"No existing experience bank at {path}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        with self._lock:
            for eid, d in data.items():
                self.experiences[eid] = ExperienceUnit.from_serializable(d)
            self._dirty = True
        logger.info(f"Loaded {len(data)} experiences from {path}")
    
    # ============ 统计 ============
    def stats(self) -> dict:
        with self._lock:
            active = [e for e in self.experiences.values() if e.status == "active"]
            return {
                "total": len(self.experiences),
                "active": len(active),
                "dormant": sum(1 for e in self.experiences.values() if e.status == "dormant"),
                "retired": sum(1 for e in self.experiences.values() if e.status == "retired"),
                "avg_confidence": float(np.mean([e.confidence for e in active])) if active else 0.0,
                "avg_use_count": float(np.mean([e.use_count for e in active])) if active else 0.0,
            }
