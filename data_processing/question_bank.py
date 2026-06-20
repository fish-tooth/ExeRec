"""
题目库: 加载题目元数据,提供按KC、难度等多种检索方式
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
from utils.logger import get_logger

logger = get_logger("data.question_bank")


class QuestionBank:
    """
    题目库,加载 high_school_annotated_clean.json
    并构建 KC → 题目, 难度 → 题目 等倒排索引
    """
    
    DIFFICULTY_RANK = {"容易": 1, "中等": 2, "较难": 3, "困难": 4, "极难": 5}
    
    def __init__(self, question_file: str, qid_to_int_file: str):
        self.question_file = question_file
        self.qid_to_int_file = qid_to_int_file
        
        self.questions: Dict[str, dict] = {}    # ques_id → 题目字典
        self.qid_to_int: Dict[str, int] = {}    # ques_id → 整数id
        self.int_to_qid: Dict[int, str] = {}
        
        # 倒排索引
        self.kc_to_qids: Dict[str, List[str]] = defaultdict(list)
        self.difficulty_to_qids: Dict[str, List[str]] = defaultdict(list)
        self.subject_to_qids: Dict[str, List[str]] = defaultdict(list)
        
        self._load()
    
    def _load(self):
        """加载题目和ID映射"""
        with open(self.qid_to_int_file, "r", encoding="utf-8") as f:
            self.qid_to_int = json.load(f)
        # qid_to_int 的 value 可能是 int 也可能是 str
        self.qid_to_int = {k: int(v) for k, v in self.qid_to_int.items()}
        self.int_to_qid = {v: k for k, v in self.qid_to_int.items()}
        
        with open(self.question_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        # 文件是字典格式 {"0": {...}, "1": {...}}
        for _, q in raw.items():
            qid = q.get("ques_id")
            if not qid:
                continue
            self.questions[qid] = q
            
            # 倒排索引
            for kc in (q.get("knowledge_concepts_list") or []):
                self.kc_to_qids[kc].append(qid)
            # 同时把 kc_mapping_gpt4o 中的 list_KCs 也算上(更细的KC)
            kc_mapping = q.get("kc_mapping_gpt4o", {})
            if isinstance(kc_mapping, dict):
                for kc in (kc_mapping.get("list_KCs") or []):
                    # 去掉括号注释 "求回归直线方程（方程与不等式）" → "求回归直线方程"
                    kc_clean = kc.split("（")[0].strip()
                    if kc_clean and qid not in self.kc_to_qids[kc_clean]:
                        self.kc_to_qids[kc_clean].append(qid)
            
            diff = q.get("difficulty", "")
            if diff:
                self.difficulty_to_qids[diff].append(qid)
            
            subj = q.get("subject", "")
            if subj:
                self.subject_to_qids[subj].append(qid)
        
        logger.info(
            f"QuestionBank loaded: {len(self.questions)} questions, "
            f"{len(self.kc_to_qids)} KCs, qid_to_int={len(self.qid_to_int)}"
        )
    
    # ============ 检索接口 ============
    def get(self, qid: str) -> Optional[dict]:
        return self.questions.get(qid)
    
    def get_qids_by_kc(self, kc: str) -> List[str]:
        return list(self.kc_to_qids.get(kc, []))
    
    def get_qids_by_kcs(self, kcs: List[str], union: bool = True) -> List[str]:
        """
        union=True: 任一KC命中
        union=False: 所有KC都命中
        """
        sets = [set(self.kc_to_qids.get(kc, [])) for kc in kcs]
        if not sets:
            return []
        if union:
            result = set().union(*sets)
        else:
            result = sets[0].intersection(*sets[1:])
        return list(result)
    
    def get_qids_by_difficulty(self, levels: List[str]) -> List[str]:
        result = []
        for lv in levels:
            result.extend(self.difficulty_to_qids.get(lv, []))
        return result
    
    def filter(
        self,
        kcs: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        subject: Optional[str] = None,
        exclude_qids: Optional[Set[str]] = None,
    ) -> List[str]:
        """组合过滤"""
        if kcs:
            pool = set(self.get_qids_by_kcs(kcs, union=True))
        elif subject:
            pool = set(self.subject_to_qids.get(subject, []))
        else:
            pool = set(self.questions.keys())
        
        if difficulties:
            diff_set = set(self.get_qids_by_difficulty(difficulties))
            pool &= diff_set
        
        if exclude_qids:
            pool -= exclude_qids
        
        return list(pool)
    
    def get_kcs_of(self, qid: str) -> List[str]:
        q = self.get(qid)
        if not q:
            return []
        kcs = list(q.get("knowledge_concepts_list") or [])
        mapping = q.get("kc_mapping_gpt4o", {})
        if isinstance(mapping, dict):
            for kc in (mapping.get("list_KCs") or []):
                kc_clean = kc.split("（")[0].strip()
                if kc_clean and kc_clean not in kcs:
                    kcs.append(kc_clean)
        return kcs
    
    def difficulty_rank(self, qid: str) -> int:
        q = self.get(qid)
        if not q:
            return 0
        return self.DIFFICULTY_RANK.get(q.get("difficulty", ""), 2)
    
    @property
    def all_qids(self) -> List[str]:
        return list(self.questions.keys())
    
    @property
    def all_kcs(self) -> List[str]:
        return list(self.kc_to_qids.keys())
    
    def __len__(self):
        return len(self.questions)
