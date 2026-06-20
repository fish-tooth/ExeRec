"""
嵌入存储
- 加载题目内容嵌入、题目ID嵌入、KC嵌入
- 提供按qid/int_id/kc的查询接口
- 提供批量向量检索接口
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.embedding_utils import load_embedding_json, batch_cosine
from utils.logger import get_logger

logger = get_logger("data.embedding")


class EmbeddingStore:
    """
    嵌入存储,支持三种嵌入类型的统一查询
    """
    
    def __init__(
        self,
        question_content_emb_path: str,
        question_int_emb_path: str,
        kc_emb_path: str,
        qid_to_int: Optional[Dict[str, int]] = None,
    ):
        self.qid_to_int = qid_to_int or {}
        
        logger.info("Loading question content embeddings...")
        self._qcontent_emb, self.content_dim = load_embedding_json(question_content_emb_path)
        
        logger.info("Loading question int-id embeddings...")
        self._qint_emb, self.int_dim = load_embedding_json(question_int_emb_path)
        
        logger.info("Loading KC embeddings...")
        self._kc_emb, self.kc_dim = load_embedding_json(kc_emb_path)
        
        logger.info(
            f"EmbeddingStore: content_emb={len(self._qcontent_emb)}, "
            f"int_emb={len(self._qint_emb)}, kc_emb={len(self._kc_emb)}, "
            f"dim={self.content_dim}"
        )
        
        # 预构建矩阵以便批量检索(惰性)
        self._qcontent_matrix = None
        self._qcontent_keys = None
    
    # ============ 单查询 ============
    def get_question_content_emb(self, key) -> Optional[np.ndarray]:
        """
        key 可以是 ques_id 字符串(如'xkw...')、或者 qid_to_int 中的整数,
        或者直接是嵌入文件中的 key(可能是 int 字符串如'0','1')
        """
        # 尝试直接命中
        s = str(key)
        if s in self._qcontent_emb:
            return self._qcontent_emb[s]
        # ques_id → int → str
        if s in self.qid_to_int:
            int_key = str(self.qid_to_int[s])
            if int_key in self._qcontent_emb:
                return self._qcontent_emb[int_key]
        return None
    
    def get_question_int_emb(self, key) -> Optional[np.ndarray]:
        s = str(key)
        if s in self._qint_emb:
            return self._qint_emb[s]
        if s in self.qid_to_int:
            int_key = str(self.qid_to_int[s])
            if int_key in self._qint_emb:
                return self._qint_emb[int_key]
        return None
    
    def get_kc_emb(self, kc: str) -> Optional[np.ndarray]:
        return self._kc_emb.get(kc)
    
    def get_kcs_emb(self, kcs: List[str]) -> Optional[np.ndarray]:
        """多 KC 取均值作为整体嵌入,作为学生薄弱 KC 的代表向量"""
        embs = [self._kc_emb[kc] for kc in kcs if kc in self._kc_emb]
        if not embs:
            return None
        return np.mean(np.stack(embs), axis=0)
    
    # ============ 批量检索 ============
    def _build_content_matrix(self):
        """惰性构建题目内容嵌入矩阵,供批量检索"""
        if self._qcontent_matrix is not None:
            return
        keys = list(self._qcontent_emb.keys())
        mat = np.stack([self._qcontent_emb[k] for k in keys])
        self._qcontent_keys = keys
        self._qcontent_matrix = mat
        logger.info(f"Built content matrix: {mat.shape}")
    
    def search_questions_by_content(
        self,
        query: np.ndarray,
        candidate_qids: Optional[List[str]] = None,
        top_n: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        按内容嵌入做向量检索
        
        Args:
            query: 查询向量 [D]
            candidate_qids: 可选的候选池(ques_id 列表),不指定则全库
            top_n: 返回前 N 个
        
        Returns:
            [(qid_or_int_key, score), ...]
        """
        self._build_content_matrix()
        
        if candidate_qids:
            # 只在候选中检索
            sub_keys = []
            sub_vecs = []
            for qid in candidate_qids:
                emb = self.get_question_content_emb(qid)
                if emb is not None:
                    sub_keys.append(qid)
                    sub_vecs.append(emb)
            if not sub_vecs:
                return []
            mat = np.stack(sub_vecs)
            scores = batch_cosine(query, mat)
            idx = np.argsort(-scores)[:top_n]
            return [(sub_keys[i], float(scores[i])) for i in idx]
        else:
            scores = batch_cosine(query, self._qcontent_matrix)
            idx = np.argsort(-scores)[:top_n]
            return [(self._qcontent_keys[i], float(scores[i])) for i in idx]
    
    def has_content_emb(self, key) -> bool:
        return self.get_question_content_emb(key) is not None
