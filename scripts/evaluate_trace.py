#!/usr/bin/env python3
"""Evaluate a JSONL trace file produced by the pipeline and produce evaluation summary.

Usage:
  python scripts/evaluate_trace.py --trace path/to/trace.jsonl [--config configs/default.yaml] [--name output_name]

This creates files under `cfg.evaluation.output_dir` (default ./logs/eval_results).
"""
import argparse
import json
from pathlib import Path
import sys

# ensure repo root is on sys.path so top-level packages like `utils` import correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_config, get_logger
from data_processing import QuestionBank
from evaluation import Evaluator

logger = get_logger("scripts.evaluate_trace")


class SimpleProfile:
    def __init__(self, weak_kcs, kc_mastery=None, ability_level="middle"):
        self.weak_kcs = weak_kcs or []
        if kc_mastery is None:
            self.kc_mastery = {kc: 0.5 for kc in self.weak_kcs}
        else:
            self.kc_mastery = kc_mastery
        self.ability_level = ability_level


def evaluate_trace(trace_path: str, cfg: dict, name: str):
    qb = QuestionBank(
        question_file=cfg["data"]["question_file"],
        qid_to_int_file=cfg["data"]["qid_to_int_file"],
    )
    evaluator = Evaluator(cfg=cfg, question_bank=qb)

    trace_p = Path(trace_path)
    if not trace_p.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    with open(trace_p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("student_id") or f"sid_{i}"
            weak_kcs = rec.get("weak_kcs", [])
            ability = rec.get("ability", "middle")
            profile = SimpleProfile(weak_kcs=weak_kcs, ability_level=ability)

            recommendation = {
                "questions": rec.get("recommended_qids", rec.get("recommended_qids", [])),
                "predicted_correct_rates": rec.get("predicted_correct_rates", []),
                "applied_experience_ids": rec.get("applied_exp_ids", []) or rec.get("applied_experience_ids", []),
            }

            # ground_truth not available in trace → pass None
            evaluator.add_round(sid, profile, recommendation, ground_truth=None)

    summary = evaluator.summarize()
    evaluator.save(summary, name=name)
    logger.info(f"Evaluation complete. Summary saved under {evaluator.output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--name", default=None, help="output name prefix")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_name = args.name or Path(args.trace).stem + "_eval"
    summary = evaluate_trace(args.trace, cfg, out_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
