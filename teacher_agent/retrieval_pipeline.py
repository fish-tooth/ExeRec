"""
教师 Agent 的三级检索流水线
L1: 结构化召回(SQL 风格过滤,毫秒级)
L2: 向量精排(余弦相似度,百毫秒级)
L3: LLM 重排序 + 推荐理由生成(秒级)
"""
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

from data_processing import QuestionBank, EmbeddingStore
from llm_adapter import LLMFactory
from utils.logger import get_logger

logger = get_logger("teacher.retrieval")


# 难度等级映射
DIFFICULTY_NAMES = {1: "容易", 2: "中等", 3: "较难", 4: "困难", 5: "极难"}
DIFFICULTY_INVERSE = {v: k for k, v in DIFFICULTY_NAMES.items()}


class RetrievalPipeline:
    """三级检索"""
    
    def __init__(
        self,
        question_bank: QuestionBank,
        embedding_store: EmbeddingStore,
        llm: LLMFactory,
        level1_pool_size: int = 200,
        level2_top_n: int = 50,
        level3_top_k: int = 5,
    ):
        self.qb = question_bank
        self.es = embedding_store
        self.llm = llm
        self.level1_pool_size = level1_pool_size
        self.level2_top_n = level2_top_n
        self.level3_top_k = level3_top_k
    
    # ============ L1: 结构化召回 ============
    def level1(
        self,
        weak_kcs: List[str],
        difficulty_centers: List[int],     # 期望难度档(可多个)
        exclude_qids: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        基于 KC + 难度 的硬过滤召回
        """
        diff_names = [DIFFICULTY_NAMES.get(d, "中等") for d in difficulty_centers]
        
        # 优先在薄弱 KC 池中
        candidates = self.qb.filter(
            kcs=weak_kcs,
            difficulties=diff_names,
            exclude_qids=exclude_qids,
        )
        
        # 候选不足时放宽难度限制
        if len(candidates) < self.level1_pool_size:
            ext = self.qb.filter(
                kcs=weak_kcs,
                exclude_qids=exclude_qids,
            )
            for qid in ext:
                if qid not in candidates:
                    candidates.append(qid)
                    if len(candidates) >= self.level1_pool_size:
                        break
        
        return candidates[:self.level1_pool_size]
    
    # ============ L2: 向量精排 ============
    def level2(
        self,
        candidates: List[str],
        student_emb: np.ndarray,
        weak_kc_emb: Optional[np.ndarray] = None,
        weight_student: float = 0.5,
        weight_kc: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        基于学生表征 + 薄弱KC表征的双通道向量精排
        """
        if not candidates:
            return []
        
        scored = []
        for qid in candidates:
            q_emb = self.es.get_question_content_emb(qid)
            if q_emb is None:
                continue
            
            # 维度对齐(简单截断)
            target_dim = q_emb.shape[0]
            s_emb = student_emb[:target_dim] if student_emb.shape[0] >= target_dim else \
                    np.pad(student_emb, (0, target_dim - student_emb.shape[0]))
            
            sim_student = self._cos(s_emb, q_emb)
            
            sim_kc = 0.0
            if weak_kc_emb is not None:
                k_emb = weak_kc_emb[:target_dim] if weak_kc_emb.shape[0] >= target_dim else \
                        np.pad(weak_kc_emb, (0, target_dim - weak_kc_emb.shape[0]))
                sim_kc = self._cos(k_emb, q_emb)
            
            score = weight_student * sim_student + weight_kc * sim_kc
            scored.append((qid, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.level2_top_n]
    
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))
    
    # ============ L3: LLM 重排序 + 理由生成 ============
    def level3(
        self,
        student_profile,
        candidates_with_score: List[Tuple[str, float]],
        strategy: dict,
        prior_lessons_text: str = "",
    ) -> Dict:
        """
        给 LLM 一组候选(简要), 让它输出 final_K + 每题的预测正确率 + 整体策略说明
        
        Returns:
            {
              "selected_qids": [...],
              "predicted_correct_rates": [...],
              "rationale": str,
              "strategy_label": str,
              "raw_llm_response": str,
            }
        """
        k = self.level3_top_k
        
        # 仅给 LLM 看 top 候选的简要(去掉 LaTeX 大段, 避免 token 爆炸)
        cand_briefs = []
        for i, (qid, score) in enumerate(candidates_with_score[:20]):
            q = self.qb.get(qid)
            if not q:
                continue
            content = self._strip_latex_brief(q.get("content", "") or q.get("question", ""))
            cand_briefs.append(
                f"[{i+1}] qid={qid}, 难度={q.get('difficulty','')}, "
                f"KCs={self.qb.get_kcs_of(qid)[:3]}, "
                f"题目摘要={content[:120]}..."
            )
        cand_text = "\n".join(cand_briefs)
        
        # Prompt
        system = (
            "你是一位教育推荐专家,需要从候选题目中为学生选出最适合补弱的题目。"
            "严格按 JSON 格式输出,不要其他内容。"
        )
        user = f"""\
## 学生画像
{student_profile.to_brief_text()}
- 偏好摘要: {student_profile.preference_summary}
- 能力摘要: {student_profile.ability_summary}

## 推荐策略约束
- KC 焦点: {strategy.get('kc_focus', [])}
- 难度目标: {strategy.get('difficulty_target', '中等')}
- 多样性目标: {strategy.get('diversity_target', 0.4)}
- 检查前置 KC: {strategy.get('prereq_check', False)}

## 历史反思教训(可参考)
{prior_lessons_text or '(无)'}

## 候选题目(已按学生相关度初步排序)
{cand_text}

## 任务
从候选中选 {k} 道最适合的题目,严格按 JSON 输出:
{{
  "selected_indices": [候选编号1, 编号2, ...],     // 选 {k} 个,来自 [1..{len(cand_briefs)}]
  "predicted_correct_rates": [浮点1, 浮点2, ...],  // 对应 {k} 道题学生作对概率,0~1
  "rationale": "100字以内总体推荐理由",
  "strategy_label": "weak_kc_first / mixed_difficulty / prereq_first / 其他"
}}
"""
        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            parsed = self._parse_l3(resp.content, len(cand_briefs))
        except Exception as e:
            logger.warning(f"L3 LLM failed: {e}; using fallback")
            parsed = None
        
        if parsed is None:
            # 降级: 直接用 L2 前 K 个
            top_qids = [q for q, _ in candidates_with_score[:k]]
            return {
                "selected_qids": top_qids,
                "predicted_correct_rates": [0.5] * len(top_qids),
                "rationale": "[LLM 降级] 直接采用向量精排结果",
                "strategy_label": "fallback",
                "raw_llm_response": "",
            }
        
        # indices → qids (1-based → 0-based)
        selected_qids = []
        for idx in parsed["selected_indices"]:
            if 1 <= idx <= len(candidates_with_score):
                qid = candidates_with_score[idx - 1][0]
                selected_qids.append(qid)
        if not selected_qids:
            selected_qids = [q for q, _ in candidates_with_score[:k]]
        
        # 对齐 predicted_correct_rates 长度
        probs = parsed["predicted_correct_rates"]
        while len(probs) < len(selected_qids):
            probs.append(0.5)
        probs = probs[:len(selected_qids)]
        
        return {
            "selected_qids": selected_qids,
            "predicted_correct_rates": probs,
            "rationale": parsed["rationale"],
            "strategy_label": parsed["strategy_label"],
            "raw_llm_response": "",
        }
    
    @staticmethod
    def _parse_l3(content: str, n_cands: int) -> Optional[dict]:
        content = content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        
        try:
            indices = data.get("selected_indices", []) or []
            indices = [int(x) for x in indices if isinstance(x, (int, float, str))]
            indices = [i for i in indices if 1 <= i <= n_cands]
            
            probs = data.get("predicted_correct_rates", []) or []
            probs = [float(np.clip(float(x), 0.0, 1.0)) for x in probs]
            
            return {
                "selected_indices": indices,
                "predicted_correct_rates": probs,
                "rationale": str(data.get("rationale", "")).strip(),
                "strategy_label": str(data.get("strategy_label", "")).strip() or "unspecified",
            }
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def _strip_latex_brief(text: str) -> str:
        """简化 LaTeX 公式,只为 prompt 紧凑性"""
        if not text:
            return ""
        text = re.sub(r"\$\$[^$]*?\$\$", "[公式]", text)
        text = re.sub(r"\$[^$]+?\$", "[公式]", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
