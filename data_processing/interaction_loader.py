"""
学生交互序列加载器
处理 pykt 风格的 train_valid_sequences.csv 和 test_sequences.csv
"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("data.interaction")


class InteractionLoader:
    """
    pykt 序列文件加载器
    
    pykt CSV 格式特点:
        student_id, questions, concepts, responses, seq_len, fold, ...
        每行是一个学生的部分序列(长序列可能切多行)
        questions/concepts/responses 是逗号分隔字符串
        -1 是填充符号
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.sequences: Dict[str, dict] = {}
        self._kc_remapped = False
        self._load()
    
    def _load(self):
        if not Path(self.csv_path).exists():
            logger.warning(f"Interaction file not found: {self.csv_path}")
            return
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "")
                if not sid:
                    continue
                
                questions = self._parse_int_list(row.get("questions", ""))
                concepts = self._parse_concepts(row.get("concepts", ""))
                responses = self._parse_int_list(row.get("responses", ""))
                seq_len = int(row.get("seq_len", 0) or 0)
                
                # 过滤填充
                clean_q, clean_c, clean_r = [], [], []
                for q, c, r in zip(questions, concepts, responses):
                    if q == -1:
                        continue
                    clean_q.append(q)
                    clean_c.append(c)
                    clean_r.append(r)
                
                if sid not in self.sequences:
                    self.sequences[sid] = {
                        "student_id": sid,
                        "questions": [],   # int qid 列表
                        "concepts": [],    # 每个 step 的 KC 列表
                        "responses": [],   # 0/1
                        "fold": row.get("fold", ""),
                    }
                # 同一学生有多行时拼接
                self.sequences[sid]["questions"].extend(clean_q)
                self.sequences[sid]["concepts"].extend(clean_c)
                self.sequences[sid]["responses"].extend(clean_r)
        
        logger.info(
            f"InteractionLoader loaded {len(self.sequences)} students from {self.csv_path}"
        )
    
    @staticmethod
    def _parse_int_list(s: str) -> List[int]:
        if not s:
            return []
        result = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                result.append(int(x))
            except ValueError:
                result.append(-1)
        return result
    
    @staticmethod
    def _parse_concepts(s: str) -> List[List[str]]:
        """concepts 字段用 _ 分隔多 KC, 逗号分隔多 step"""
        if not s:
            return []
        result = []
        for step in s.split(","):
            step = step.strip()
            if not step or step == "-1":
                result.append([])
                continue
            result.append([kc for kc in step.split("_") if kc])
        return result
    
    # ============ KC 命名空间对齐 ============
    def attach_kc_mapping(
        self,
        kc_to_int_file: str,
        question_bank=None,
    ):
        """
        ★ 关键修复: 把所有 step 的 KC 从 pykt 数字编号替换为题库的中文 KC
        
        CSV 中的 concepts 字段是 pykt 内部数字编号(如 "39", "34950_365"),
        与 QuestionBank.kc_to_qids 的中文 key 完全不匹配。
        
        本方法用 kc_to_int.json 做精确的"数字 → 中文" 映射:
            kc_to_int.json 格式: {"求回归直线方程": 0, "解释回归直线方程的意义": 1, ...}
            反转后: {0: "求回归直线方程", 1: "解释回归直线方程的意义", ...}
        
        对每个 step 的 KC 列表,把数字字符串映射回中文名。
        如果有 ID 在 kc_to_int.json 里找不到(可能是题库扩展但映射没更新),
        且提供了 question_bank,则用题库的题目 KC 反查作为兜底。
        """
        if self._kc_remapped:
            logger.info("KCs already remapped, skip")
            return
        
        # 1) 加载 kc_to_int.json,反转为 int → 中文 KC
        from pathlib import Path
        import json
        
        kc_path = Path(kc_to_int_file)
        if not kc_path.exists():
            logger.error(f"kc_to_int_file not found: {kc_to_int_file}")
            return
        
        with open(kc_path, "r", encoding="utf-8") as f:
            kc_to_int = json.load(f)
        # kc_to_int 的 value 可能是 int 也可能是 str
        int_to_kc: Dict[str, str] = {}
        for cn_kc, int_id in kc_to_int.items():
            int_to_kc[str(int_id)] = cn_kc
        
        logger.info(f"Loaded kc_to_int: {len(int_to_kc)} entries")
        
        # 2) 遍历所有 step,替换 KC
        total_steps = 0
        steps_with_any_cn = 0
        unmapped_kc_ids = set()
        used_bank_fallback = 0
        
        for sid, seq in self.sequences.items():
            new_concepts = []
            for q_int, old_kc_list in zip(seq["questions"], seq["concepts"]):
                total_steps += 1
                cn_kcs: List[str] = []
                
                # 优先用 kc_to_int 直接映射(精确)
                for kc_int_str in old_kc_list:
                    cn = int_to_kc.get(str(kc_int_str))
                    if cn:
                        cn_kcs.append(cn)
                    else:
                        unmapped_kc_ids.add(str(kc_int_str))
                
                # 兜底: 从题库反查该 q_int 对应题目的中文 KC
                if not cn_kcs and question_bank is not None:
                    qid_str = question_bank.int_to_qid.get(q_int)
                    if qid_str:
                        cn_kcs = question_bank.get_kcs_of(qid_str)
                        if cn_kcs:
                            used_bank_fallback += 1
                
                new_concepts.append(cn_kcs)
                if cn_kcs:
                    steps_with_any_cn += 1
            seq["concepts"] = new_concepts
        
        self._kc_remapped = True
        logger.info(
            f"KC remap complete: "
            f"{steps_with_any_cn}/{total_steps} steps got Chinese KCs, "
            f"used bank fallback for {used_bank_fallback} steps, "
            f"unmapped KC IDs: {len(unmapped_kc_ids)} "
            f"(first 10: {list(unmapped_kc_ids)[:10]})"
        )
    
    # 旧名字保留以保持向后兼容
    def attach_question_bank(self, question_bank):
        """已废弃: 请用 attach_kc_mapping(kc_to_int_file=..., question_bank=...) """
        logger.warning(
            "attach_question_bank() is legacy. Use attach_kc_mapping() with kc_to_int.json instead."
        )
        if self._kc_remapped:
            return
        # 旧实现:只用题库反查
        total_steps = 0
        remapped_steps = 0
        unresolved_qids = set()
        for sid, seq in self.sequences.items():
            new_concepts = []
            for q_int, _old_kcs in zip(seq["questions"], seq["concepts"]):
                total_steps += 1
                qid_str = question_bank.int_to_qid.get(q_int)
                if qid_str is None:
                    new_concepts.append([])
                    unresolved_qids.add(q_int)
                    continue
                cn_kcs = question_bank.get_kcs_of(qid_str)
                new_concepts.append(cn_kcs)
                if cn_kcs:
                    remapped_steps += 1
            seq["concepts"] = new_concepts
        self._kc_remapped = True
        logger.info(
            f"KC remap via bank fallback: {remapped_steps}/{total_steps} steps remapped, "
            f"unresolved qids: {len(unresolved_qids)}"
        )
    
    # ============ 接口 ============
    def get(self, student_id: str) -> Optional[dict]:
        return self.sequences.get(student_id)
    
    def get_truncated(self, student_id: str, max_len: int) -> Optional[dict]:
        """取最近 max_len 条交互"""
        seq = self.get(student_id)
        if not seq:
            return None
        return {
            "student_id": student_id,
            "questions": seq["questions"][-max_len:],
            "concepts": seq["concepts"][-max_len:],
            "responses": seq["responses"][-max_len:],
            "fold": seq.get("fold", ""),
        }
    
    @property
    def student_ids(self) -> List[str]:
        return list(self.sequences.keys())
    
    def __len__(self):
        return len(self.sequences)
