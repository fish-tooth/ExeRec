"""
学生 Agent 主类
统一对外接口:
  - build_profile(student_id) → StudentProfile
  - simulate(student_id, questions) → dict
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from .student_profile import StudentProfile
from .dkt_wrapper import DKTSimulator
from .profile_generator import ProfileGenerator
from .group_prior import GroupPriorModule
from data_processing import QuestionBank, InteractionLoader, EmbeddingStore
from utils.logger import get_logger

logger = get_logger("student.agent")


class StudentAgent:
    """
    学生 Agent
    - 用 DKT 诊断 + 模拟
    - 用 GMM 群体先验做表征融合
    - 用 LLM 生成自然语言画像
    """
    
    def __init__(
        self,
        cfg: dict,
        question_bank: QuestionBank,
        interaction_train: InteractionLoader,
        interaction_test: InteractionLoader,
        embedding_store: EmbeddingStore,
        dkt: DKTSimulator,
        profile_generator: ProfileGenerator,
        group_prior: GroupPriorModule,
    ):
        self.cfg = cfg
        self.qb = question_bank
        self.itx_train = interaction_train
        self.itx_test = interaction_test
        self.es = embedding_store
        self.dkt = dkt
        self.profile_gen = profile_generator
        self.group_prior = group_prior
        
        sa_cfg = cfg.get("student_agent", {})
        dkt_cfg = cfg.get("dkt", {})
        self.weak_kc_threshold = dkt_cfg.get("weak_kc_threshold", 0.4)
        self.weak_kc_top_n = dkt_cfg.get("weak_kc_top_n", 5)
        self.use_llm_profile = sa_cfg.get("profile", {}).get("use_llm", True)
        self.use_group_prior = sa_cfg.get("group_prior", {}).get("enable", True)
        
        self._profile_cache: Dict[str, StudentProfile] = {}
    
    # ============ 表征构建 ============
    def _build_individual_feature(
        self,
        kc_mastery: Dict[str, float],
        correct_rate: float,
        interaction_count: int,
    ) -> np.ndarray:
        """
        从 DKT 诊断 + 统计量构建学生个体表征
        简化版: 用全KC掌握度向量 + 几个全局统计量 拼接
        
        TODO: 论文级实现替换为 Transformer 编码的输出
        """
        all_kcs = sorted(self.qb.all_kcs)
        mastery_vec = np.array(
            [kc_mastery.get(kc, 0.5) for kc in all_kcs], dtype=np.float32
        )
        global_stats = np.array([
            correct_rate,
            min(interaction_count / 200.0, 1.0),
            float(np.mean(list(kc_mastery.values()))) if kc_mastery else 0.5,
            float(np.std(list(kc_mastery.values()))) if kc_mastery else 0.0,
        ], dtype=np.float32)
        return np.concatenate([mastery_vec, global_stats])
    
    def _build_sig_embedding(self, profile: StudentProfile) -> np.ndarray:
        """
        构建用于反思检索的签名嵌入
        基于 weak_kcs 的均值嵌入 + 能力水平 one-hot + 偏好 one-hot
        """
        # 1) 薄弱 KC 嵌入均值
        weak_kc_emb = self.es.get_kcs_emb(profile.weak_kcs)
        if weak_kc_emb is None:
            weak_kc_emb = np.zeros(self.es.kc_dim or 1024, dtype=np.float32)
        
        # 2) 能力 one-hot (low/middle/high)
        ability_map = {"low": [1, 0, 0], "middle": [0, 1, 0], "high": [0, 0, 1]}
        ability_oh = np.array(ability_map.get(profile.ability_level, [0, 1, 0]), dtype=np.float32)
        
        # 3) 密度 one-hot
        density_map = {"sparse": [1, 0, 0], "medium": [0, 1, 0], "dense": [0, 0, 1]}
        density_oh = np.array(density_map.get(profile.interaction_density, [0, 1, 0]), dtype=np.float32)
        
        # 4) 偏好(简单字符串 hash 到 4 维)
        pref_map = {
            "step_by_step": [1, 0, 0, 0],
            "visual": [0, 1, 0, 0],
            "fast_drill": [0, 0, 1, 0],
            "deep_analysis": [0, 0, 0, 1],
            "balanced": [0.25, 0.25, 0.25, 0.25],
        }
        pref_oh = np.array(
            pref_map.get(profile.learning_preference, [0.25] * 4),
            dtype=np.float32
        )
        
        return np.concatenate([weak_kc_emb, ability_oh, density_oh, pref_oh])
    
    # ============ 对外接口 ============
    def build_profile(
        self,
        student_id: str,
        use_cache: bool = True,
    ) -> StudentProfile:
        """
        构建/检索学生画像
        """
        if use_cache and student_id in self._profile_cache:
            return self._profile_cache[student_id]
        
        # 从 test 集取交互(先验数据可从 train 拼接)
        seq = self.itx_test.get(student_id) or self.itx_train.get(student_id)
        if not seq:
            logger.warning(f"No interaction found for student {student_id}; using empty profile")
            return self._build_empty_profile(student_id)
        
        # 注意: test集做评估时,我们其实只用"前面一段"作为已知交互,后面用作 ground truth
        # 这里 build_profile 假设传入的 seq 已经是"截止当前时刻"的可用数据
        questions = seq["questions"]
        concepts = seq["concepts"]
        responses = seq["responses"]
        
        # 1) DKT 诊断
        mastery = self.dkt.diagnose(
            questions, concepts, responses,
            all_kcs=self.qb.all_kcs,
        )
        
        # 2) 薄弱 KC
        sorted_kcs = sorted(mastery.items(), key=lambda x: x[1])
        weak_kcs = [kc for kc, m in sorted_kcs[:self.weak_kc_top_n] if m < self.weak_kc_threshold]
        if not weak_kcs:
            # 没有掌握度 < 阈值的, 退而取最低的 top_n
            weak_kcs = [kc for kc, _ in sorted_kcs[:self.weak_kc_top_n]]
        
        # 3) 全局统计
        avg_mastery = float(np.mean(list(mastery.values()))) if mastery else 0.5
        correct_rate = (
            sum(r for r in responses if r != -1) / max(1, len([r for r in responses if r != -1]))
        )
        interaction_count = len([q for q in questions if q != -1])
        
        # 能力等级
        if avg_mastery < 0.4:
            ability_level = "low"
        elif avg_mastery < 0.7:
            ability_level = "middle"
        else:
            ability_level = "high"
        
        # 密度
        if interaction_count < 30:
            density = "sparse"
        elif interaction_count < 100:
            density = "medium"
        else:
            density = "dense"
        
        # 4) 已答题目集合(避免重复推荐)
        answered_qids = set()
        for q_int in questions:
            if q_int == -1:
                continue
            qid_str = self.qb.int_to_qid.get(q_int)
            if qid_str:
                answered_qids.add(qid_str)
        
        # 5) 自然语言画像
        if self.use_llm_profile:
            recent_briefs = self._build_recent_briefs(questions[-5:], responses[-5:])
            profile_text = self.profile_gen.generate(
                student_id=student_id,
                weak_kcs=weak_kcs,
                avg_mastery=avg_mastery,
                correct_rate=correct_rate,
                interaction_count=interaction_count,
                recent_questions_brief=recent_briefs,
            )
        else:
            profile_text = {
                "weak_kcs_summary": "",
                "ability_summary": "",
                "preference_summary": "",
                "learning_preference_tag": "balanced",
            }
        
        # 6) 个体表征
        individual_feat = self._build_individual_feature(
            mastery, correct_rate, interaction_count
        )
        
        # 7) 群体先验融合
        if self.use_group_prior and self.group_prior.fitted:
            fused, group_emb, alpha = self.group_prior.fuse(individual_feat, interaction_count)
        else:
            fused = individual_feat
            group_emb = None
            alpha = 1.0
        
        # 8) 构造 profile
        profile = StudentProfile(
            student_id=student_id,
            kc_mastery=mastery,
            weak_kcs=weak_kcs,
            ability_level=ability_level,
            avg_mastery=avg_mastery,
            interaction_count=interaction_count,
            correct_rate=correct_rate,
            interaction_density=density,
            answered_qids=list(answered_qids),
            weak_kcs_summary=profile_text["weak_kcs_summary"],
            ability_summary=profile_text["ability_summary"],
            preference_summary=profile_text["preference_summary"],
            learning_preference=profile_text["learning_preference_tag"],
            individual_embedding=individual_feat,
            group_embedding=group_emb,
            fused_embedding=fused,
        )
        # sig embedding 依赖前面的字段
        profile.sig_embedding = self._build_sig_embedding(profile)
        
        if use_cache:
            self._profile_cache[student_id] = profile
        return profile
    
    def _build_empty_profile(self, student_id: str) -> StudentProfile:
        """无交互数据时的空画像(冷启动)"""
        return StudentProfile(
            student_id=student_id,
            kc_mastery={},
            weak_kcs=[],
            ability_level="middle",
            avg_mastery=0.5,
            interaction_count=0,
            correct_rate=0.5,
            interaction_density="sparse",
        )
    
    def _build_recent_briefs(self, recent_q_ints, recent_responses) -> List[str]:
        briefs = []
        for q_int, r in zip(recent_q_ints, recent_responses):
            if q_int == -1:
                continue
            qid = self.qb.int_to_qid.get(q_int)
            if not qid:
                continue
            q = self.qb.get(qid)
            if not q:
                continue
            kcs = ", ".join(self.qb.get_kcs_of(qid)[:2])
            outcome = "答对" if r == 1 else "答错"
            briefs.append(f"- 难度[{q.get('difficulty','')}] KC[{kcs}] {outcome}")
        return briefs
    
    # ============ 模拟答题 ============
    def simulate(
        self,
        student_id: str,
        question_qids: List[str],
        seed: Optional[int] = None,
    ) -> Dict:
        """
        模拟学生在 question_qids 上的作答
        
        Returns:
            {
              "predicted_probs": [...],
              "simulated_correct": [0/1, ...],
              "kg_before": {...},
              "kg_after": {...},
            }
        """
        seq = self.itx_test.get(student_id) or self.itx_train.get(student_id) or {
            "questions": [], "concepts": [], "responses": []
        }
        history_q = seq["questions"]
        history_c = seq["concepts"]
        history_r = seq["responses"]
        
        # 转换 next 题的 int qid 和 KC
        next_q_int = []
        next_c = []
        for qid in question_qids:
            q_int = self.qb.qid_to_int.get(qid, -1)
            next_q_int.append(q_int)
            next_c.append(self.qb.get_kcs_of(qid))
        
        # 模拟前掌握度
        kg_before = self.dkt.diagnose(history_q, history_c, history_r, self.qb.all_kcs)
        
        # 预测每道题的正确概率
        probs = self.dkt.simulate(history_q, history_c, history_r, next_q_int, next_c)
        simulated = self.dkt.sample_outcomes(probs, seed=seed)
        
        # 模拟后掌握度: 把模拟结果追加到历史再诊断
        new_q = list(history_q) + next_q_int
        new_c = list(history_c) + next_c
        new_r = list(history_r) + simulated
        kg_after = self.dkt.diagnose(new_q, new_c, new_r, self.qb.all_kcs)
        
        return {
            "predicted_probs": probs,
            "simulated_correct": simulated,
            "kg_before": kg_before,
            "kg_after": kg_after,
        }
    
    # ============ 群体先验训练 ============
    def fit_group_prior(self, sample_size: Optional[int] = None):
        """
        用训练集学生拟合 GMM
        sample_size: 用多少学生(None = 全部)
        """
        sids = list(self.itx_train.student_ids)
        if sample_size and sample_size < len(sids):
            import random
            random.seed(42)
            sids = random.sample(sids, sample_size)
        
        feats = []
        for sid in sids:
            seq = self.itx_train.get(sid)
            if not seq or not seq["questions"]:
                continue
            mastery = self.dkt.diagnose(
                seq["questions"], seq["concepts"], seq["responses"],
                self.qb.all_kcs,
            )
            n = len([r for r in seq["responses"] if r != -1])
            cr = sum(r for r in seq["responses"] if r != -1) / max(1, n)
            feats.append(
                self._build_individual_feature(mastery, cr, n)
            )
        
        if not feats:
            logger.warning("No features for fit_group_prior")
            return
        X = np.stack(feats)
        self.group_prior.fit(X)
