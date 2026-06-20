"""
顺序运行所有消融实验
用法:
    python scripts/run_all_ablations.py

会依次执行:
  configs/default.yaml
  configs/ablation_no_reflection.yaml
  configs/ablation_flat_reflection.yaml
  configs/ablation_no_confidence.yaml
所有结果集中输出到 logs/eval_results/ 目录
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config, setup_logging, get_logger
from scripts.run_pipeline import run_pipeline


CONFIGS = [
    "configs/default.yaml",
    "configs/ablation_no_reflection.yaml",
    "configs/ablation_flat_reflection.yaml",
    "configs/ablation_no_confidence.yaml",
]


def main():
    logger = get_logger("scripts.ablations")
    all_results = {}
    
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        name = cfg["experiment"]["name"]
        setup_logging(cfg["experiment"]["log_dir"], run_name=name)
        logger.info(f"\n{'='*70}\nRunning: {name}\n{'='*70}")
        
        # 清除经验库,确保各实验互相独立
        bank_path = Path(cfg["reflection"]["persist"]["path"])
        if bank_path.exists():
            bank_path.unlink()
        
        summary = run_pipeline(cfg)
        all_results[name] = {
            k: v for k, v in summary.items()
            if isinstance(v, (int, float)) or k in ("global_agent", "llm_stats")
        }
    
    # 汇总输出
    out_path = Path(CONFIGS[0]).parent.parent / "logs" / "eval_results" / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"All ablations done. Summary saved to {out_path}")


if __name__ == "__main__":
    main()
