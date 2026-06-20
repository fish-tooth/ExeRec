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
