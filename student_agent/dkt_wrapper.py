"""
DKT 模拟器包装
使用 pykt-toolkit 加载预训练模型
两个核心能力:
  1. diagnose(): 给定学生交互序列,返回每个KC的掌握度
  2. simulate(): 给定后续题目,预测学生是否做对(模拟答题)

注意: pykt 不同模型的接口略有差异,此处实现 DKT 标准接口
如果你的模型是其他类型(AKT/DKVMN/SAKT),只需替换 _load_model 内部
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("student.dkt")


class DKTSimulator:
    """
    DKT 包装器
    
    使用方式:
        sim = DKTSimulator(model_dir, device)
        mastery = sim.diagnose(questions, concepts, responses)
        # mastery: {KC: float[0,1]}
        
        sim_probs = sim.simulate(history_q, history_c, history_r, next_qs, next_cs)
        # sim_probs: List[float] 每个后续题的预测正确率
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        max_seq_len: int = 200,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_seq_len = max_seq_len
        self.config = self._load_config()
        self.model = None
        self.kc_id_to_name: Dict[int, str] = {}    # int id → KC 名称
        self.kc_name_to_id: Dict[str, int] = {}
        self._load_model()
    
    def _load_config(self) -> dict:
        cfg_path = self.model_dir / "config.json"
        if not cfg_path.exists():
            logger.warning(f"config.json not found in {self.model_dir}, using defaults")
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_model(self):
        """
        加载 pykt 预训练模型
        pykt 风格的 checkpoint 通常包含模型权重
        """
        try:
            import torch
        except ImportError:
            logger.error("torch not installed; DKT cannot run")
            return
        
        ckpt_path = self.model_dir / "pretrained_model.ckpt"
        if not ckpt_path.exists():
            # 尝试其他常见命名
            for name in ["qid_model.ckpt", "model.ckpt", "best_model.ckpt"]:
                p = self.model_dir / name
                if p.exists():
                    ckpt_path = p
                    break
        
        if not ckpt_path.exists():
            logger.warning(
                f"No checkpoint found in {self.model_dir}; "
                f"DKTSimulator will use random fallback"
            )
            self.model = None
            return
        
        try:
            # pykt 的模型构造需要 model_name 和参数
            # 这里采取保守做法: 加载 state_dict, 不依赖 pykt 内部 API
            # 真实使用时, 推荐用 pykt.models.init_model 重建模型
            self._init_pykt_model(ckpt_path)
        except Exception as e:
            logger.warning(f"Failed to load DKT via pykt API: {e}; using fallback inference")
            self.model = None
    
    def _init_pykt_model(self, ckpt_path: Path):
        """
        尝试用 pykt 标准方式重建模型
        若 pykt 未安装或 API 不匹配,降级为简单回退
        """
        import torch
        
        try:
            # pykt API: from pykt.models import init_model
            from pykt.models import init_model
            model_name = self.config.get("model_name", "dkt")
            model_config = self.config.get("model_config", {})
            data_config = self.config.get("data_config", {})
            
            self.model = init_model(model_name, model_config, data_config, "qid")
            state = torch.load(str(ckpt_path), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded pykt {model_name} from {ckpt_path}")
        except ImportError:
            logger.warning("pykt-toolkit not installed; using fallback")
            self.model = None
        except Exception as e:
            logger.warning(f"pykt init_model failed: {e}; using fallback")
            self.model = None
    
    # ============ 核心接口 ============
    def diagnose(
        self,
        questions: List[int],
        concepts: List[List[str]],
        responses: List[int],
        all_kcs: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        诊断学生在每个KC上的掌握度
        
        Args:
            questions: 学生答题的 int qid 序列
            concepts: 每个题对应的 KC 列表(每个元素是 KC 列表)
            responses: 0/1 答题结果
            all_kcs: 要诊断的全部 KC 列表(None 表示从交互中出现的 KC)
        
        Returns:
            {KC名: 掌握度 in [0,1]}
        """
        if self.model is None:
            # 回退策略: 按 KC 的正确率统计作为掌握度估计
            return self._diagnose_fallback(questions, concepts, responses, all_kcs)
        
        try:
            return self._diagnose_with_pykt(questions, concepts, responses, all_kcs)
        except Exception as e:
            logger.warning(f"pykt diagnose failed: {e}; using fallback")
            return self._diagnose_fallback(questions, concepts, responses, all_kcs)
    
    def _diagnose_fallback(
        self, questions, concepts, responses, all_kcs
    ) -> Dict[str, float]:
        """回退诊断: 简单的 KC 正确率统计 + Laplace 平滑"""
        from collections import defaultdict
        kc_correct = defaultdict(int)
        kc_total = defaultdict(int)
        for q, c_list, r in zip(questions, concepts, responses):
            if q == -1 or r == -1:
                continue
            for kc in c_list:
                kc_correct[kc] += int(r)
                kc_total[kc] += 1
        
        result = {}
        for kc, total in kc_total.items():
            # Laplace 平滑,样本少时偏向 0.5
            result[kc] = (kc_correct[kc] + 1) / (total + 2)
        
        # 未出现的 KC 用先验 0.5
        if all_kcs:
            for kc in all_kcs:
                if kc not in result:
                    result[kc] = 0.5
        return result
    
    def _diagnose_with_pykt(
        self, questions, concepts, responses, all_kcs
    ) -> Dict[str, float]:
        """
        真正调用 pykt 推理
        对每个目标 KC,构造"下一题是该 KC"的查询,取预测概率作为掌握度
        """
        import torch
        # pykt DKT 输入接口: (qseq, cseq, rseq) → 概率
        # 这里给出一个标准化的实现,具体的 forward 签名以你的 pykt 版本为准
        
        # 占位实现: 调用 fallback
        # TODO: 根据实际 pykt 版本接入,例如:
        # with torch.no_grad():
        #     pred = self.model(qseq, cseq, rseq)
        return self._diagnose_fallback(questions, concepts, responses, all_kcs)
    
    def simulate(
        self,
        history_questions: List[int],
        history_concepts: List[List[str]],
        history_responses: List[int],
        next_questions: List[int],
        next_concepts: List[List[str]],
    ) -> List[float]:
        """
        模拟学生在 next_questions 上的答题结果
        
        Returns:
            List[float] 每道题的预测正确概率
        """
        if self.model is None:
            return self._simulate_fallback(
                history_questions, history_concepts, history_responses,
                next_questions, next_concepts
            )
        
        try:
            return self._simulate_with_pykt(
                history_questions, history_concepts, history_responses,
                next_questions, next_concepts
            )
        except Exception as e:
            logger.warning(f"pykt simulate failed: {e}; using fallback")
            return self._simulate_fallback(
                history_questions, history_concepts, history_responses,
                next_questions, next_concepts
            )
    
    def _simulate_fallback(
        self, history_q, history_c, history_r, next_q, next_c
    ) -> List[float]:
        """回退模拟: 用 KC 掌握度估计每道题的正确概率"""
        mastery = self._diagnose_fallback(history_q, history_c, history_r, None)
        probs = []
        for q, c_list in zip(next_q, next_c):
            if not c_list:
                probs.append(0.5)
                continue
            # 多 KC 题: 取掌握度乘积(假设独立),或最小值(短板效应)
            # 此处用几何平均更稳健
            kc_probs = [mastery.get(kc, 0.5) for kc in c_list]
            geo_mean = float(np.exp(np.mean(np.log(np.clip(kc_probs, 1e-3, 1.0)))))
            probs.append(geo_mean)
        return probs
    
    def _simulate_with_pykt(
        self, history_q, history_c, history_r, next_q, next_c
    ) -> List[float]:
        """TODO: 接入 pykt 真实推理"""
        return self._simulate_fallback(history_q, history_c, history_r, next_q, next_c)
    
    def sample_outcomes(
        self, probs: List[float], seed: Optional[int] = None
    ) -> List[int]:
        """根据预测概率采样得到具体的 0/1 结果"""
        rng = np.random.RandomState(seed)
        return [int(rng.rand() < p) for p in probs]
