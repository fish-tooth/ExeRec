"""
主推荐流水线
完整闭环:
  build_profile → retrieve_experiences → recommend → simulate → process_round
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

# 把项目根目录加入 sys.path,允许 `python scripts/run_pipeline.py` 直接跑
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config, get_logger, setup_logging, set_seed
from utils.logger import JsonlLogger
from llm_adapter import LLMFactory
from data_processing import QuestionBank, InteractionLoader, EmbeddingStore
from student_agent import (
    DKTSimulator, ProfileGenerator, GroupPriorModule, StudentAgent
)
from teacher_agent import TeacherAgent
from global_agent import GlobalAgent
from evaluation import Evaluator

logger = get_logger("scripts.pipeline")


def build_all(cfg: dict):
    """构建所有组件并返回"""
    set_seed(cfg.get("experiment", {}).get("seed", 42))
    
    # 数据
    logger.info("Loading question bank...")
    qb = QuestionBank(
        question_file=cfg["data"]["question_file"],
        qid_to_int_file=cfg["data"]["qid_to_int_file"],
    )
    
    logger.info("Loading interactions...")
    itx_train = InteractionLoader(cfg["data"]["train_valid_seq"])
    itx_test = InteractionLoader(cfg["data"]["test_seq"])
    
    # ★ 关键: 把 CSV 中的数字 KC 替换为题库的中文 KC
    # 否则 weak_kcs 与 QuestionBank.kc_to_qids 命名空间不对齐,
    # 检索/反思/评估全失效
    kc_to_int_file = cfg["data"].get("kc_to_int_file")
    if kc_to_int_file:
        logger.info(f"Remapping interaction KCs via {kc_to_int_file}...")
        itx_train.attach_kc_mapping(kc_to_int_file, question_bank=qb)
        itx_test.attach_kc_mapping(kc_to_int_file, question_bank=qb)
    else:
        logger.warning(
            "cfg.data.kc_to_int_file not configured; falling back to bank-only remap. "
            "Recommendation/evaluation may be degraded due to KC namespace mismatch."
        )
        itx_train.attach_question_bank(qb)
        itx_test.attach_question_bank(qb)
    
    logger.info("Loading embeddings...")
    es = EmbeddingStore(
        question_content_emb_path=cfg["data"]["question_content_emb"],
        question_int_emb_path=cfg["data"]["question_int_emb"],
        kc_emb_path=cfg["data"]["kc_emb"],
        qid_to_int=qb.qid_to_int,
    )
    
    # LLM
    logger.info(f"Initializing LLM backend: {cfg['llm']['default_backend']}")
    llm = LLMFactory.from_config(cfg)
    
    # DKT
    logger.info("Initializing DKT simulator...")
    dkt = DKTSimulator(
        model_dir=cfg["dkt"]["model_dir"],
        device=cfg["dkt"].get("device", "cuda:0"),
        max_seq_len=cfg["dkt"].get("max_seq_len", 200),
    )
    
    # Profile / Group prior
    profile_gen = ProfileGenerator(llm)
    group_prior = GroupPriorModule(
        n_clusters=cfg["student_agent"]["group_prior"].get("n_clusters", 8),
        fusion_alpha_mode=cfg["student_agent"]["group_prior"].get("fusion_alpha_mode", "adaptive"),
        fusion_alpha_fixed=cfg["student_agent"]["group_prior"].get("fusion_alpha_fixed", 0.7),
    )
    # 加载已保存的群体先验
    gp_path = Path(cfg.get("experiment", {}).get("log_dir", "./logs")) / "group_prior.pkl"
    if gp_path.exists():
        try:
            group_prior = GroupPriorModule.load(str(gp_path))
            logger.info(f"Loaded group prior from {gp_path}")
        except Exception as e:
            logger.warning(f"Failed to load group prior: {e}")
    
    # Agents
    student_agent = StudentAgent(
        cfg=cfg,
        question_bank=qb,
        interaction_train=itx_train,
        interaction_test=itx_test,
        embedding_store=es,
        dkt=dkt,
        profile_generator=profile_gen,
        group_prior=group_prior,
    )
    
    # 如果群体先验未拟合,即时拟合(用一小部分训练学生)
    if cfg["student_agent"]["group_prior"].get("enable", True) and not group_prior.fitted:
        logger.info("Group prior not fitted; fitting now with 500 training students...")
        student_agent.fit_group_prior(sample_size=500)
        group_prior.save(str(gp_path))
    
    teacher_agent = TeacherAgent(
        cfg=cfg,
        question_bank=qb,
        embedding_store=es,
        llm=llm,
    )
    
    global_agent = GlobalAgent(cfg=cfg, llm=llm)
    
    return {
        "cfg": cfg,
        "llm": llm,
        "qb": qb,
        "itx_train": itx_train,
        "itx_test": itx_test,
        "es": es,
        "student_agent": student_agent,
        "teacher_agent": teacher_agent,
        "global_agent": global_agent,
    }


def split_seq_for_eval(seq: dict, split_ratio: float = 0.7):
    """
    将测试集学生的交互序列按时序切分:
      前 split_ratio 比例作为"已知交互"(供 build_profile 用)
      后 1-split_ratio 部分作为 ground truth
    
    返回 (known_seq, future_qids, future_kcs)
    """
    qs = seq["questions"]
    cs = seq["concepts"]
    rs = seq["responses"]
    n = len(qs)
    if n < 5:
        return None, [], []
    split = max(2, int(n * split_ratio))
    
    known = {
        "student_id": seq["student_id"],
        "questions": qs[:split],
        "concepts": cs[:split],
        "responses": rs[:split],
    }
    
    # 未来"答对"的题作为正样本(也可以选所有未来题,这里取答对的)
    future_qids_int = [qs[i] for i in range(split, n) if rs[i] == 1 and qs[i] != -1]
    future_kcs = []
    for i in range(split, n):
        if rs[i] == 1:
            future_kcs.extend(cs[i])
    
    return known, future_qids_int, future_kcs


def run_pipeline(cfg: dict):
    """主流水线"""
    components = build_all(cfg)
    qb = components["qb"]
    itx_test = components["itx_test"]
    student_agent = components["student_agent"]
    teacher_agent = components["teacher_agent"]
    global_agent = components["global_agent"]
    
    # 评估器
    evaluator = Evaluator(cfg=cfg, question_bank=qb)
    
    # 详细 trace 日志
    log_dir = Path(cfg["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trace_logger = JsonlLogger(
        str(log_dir / f"trace_{cfg['experiment']['name']}_{timestamp}.jsonl")
    )
    
    # 测试学生
    test_sids = itx_test.student_ids
    max_students = cfg["evaluation"].get("max_students", 200)
    if max_students and len(test_sids) > max_students:
        test_sids = test_sids[:max_students]
    
    K_for_recommend = max(cfg["evaluation"].get("K_list", [5]))
    
    logger.info(f"Starting pipeline: {len(test_sids)} students, K={K_for_recommend}")
    t0 = time.time()
    
    for i, sid in enumerate(test_sids):
        try:
            seq = itx_test.get(sid)
            if not seq:
                continue
            
            # 时序切分
            known, future_qids_int, future_kcs = split_seq_for_eval(seq, split_ratio=0.7)
            if known is None:
                continue
            
            # 把已知部分塞回 itx_test 的临时副本(简化: 直接修改 sequences 字典,跑完再恢复)
            original = itx_test.sequences[sid]
            itx_test.sequences[sid] = known
            
            try:
                # 1) 构建画像
                profile = student_agent.build_profile(sid, use_cache=False)
                
                # 2) 检索相关经验(由全局Agent提供)
                relevant_exps = global_agent.retrieve_experiences(profile)
                
                # 3) 教师Agent推荐
                recommendation = teacher_agent.recommend(
                    student_profile=profile,
                    relevant_experiences=relevant_exps,
                    K=K_for_recommend,
                )
                
                # 4) 学生Agent模拟答题
                sim_result = student_agent.simulate(
                    sid, recommendation["questions"], seed=i,
                )
                
                # 计算实际 KG (DKT 模拟前后掌握度的平均提升)
                kg_before = sim_result.get("kg_before", {})
                kg_after = sim_result.get("kg_after", {})
                weak = set(profile.weak_kcs)
                if weak:
                    deltas = [
                        max(0, kg_after.get(kc, 0.5) - kg_before.get(kc, 0.5))
                        for kc in weak
                    ]
                    actual_kg = float(sum(deltas) / len(deltas))
                else:
                    actual_kg = 0.0
                sim_result["actual_kg"] = actual_kg
                
                # 5) 全局Agent: 反思生成 + 经验维护
                new_exp, severity, judge_debug = global_agent.process_round(
                    student_profile=profile,
                    recommendation=recommendation,
                    simulated_results=sim_result,
                    applied_experiences=relevant_exps,
                    round_id=i,
                )
                
                # 6) 评估器记账
                gt_qids_str = [qb.int_to_qid.get(qi) for qi in future_qids_int if qb.int_to_qid.get(qi)]
                ground_truth = {
                    "qids": gt_qids_str,
                    "kcs": list(set(future_kcs)),
                }
                evaluator.add_round(sid, profile, recommendation, ground_truth)
                
                # 7) trace
                trace_logger.log({
                    "round": i,
                    "student_id": sid,
                    "weak_kcs": profile.weak_kcs[:5],
                    "ability": profile.ability_level,
                    "n_relevant_exps": len(relevant_exps),
                    "applied_exp_ids": recommendation["applied_experience_ids"],
                    "n_recommended": len(recommendation["questions"]),  # ★ 诊断
                    "strategy_label": recommendation.get("strategy_label", ""),  # ★ 诊断
                    "recommended_qids": recommendation["questions"],
                    "predicted_correct_rates": recommendation["predicted_correct_rates"],
                    "simulated_correct": sim_result["simulated_correct"],
                    "expected_kg": recommendation["expected_kg"],
                    "actual_kg": actual_kg,
                    "severity": severity,
                    "new_exp_id": new_exp.exp_id if new_exp else None,
                    "judge_debug": judge_debug,
                })
            finally:
                # 恢复原始序列
                itx_test.sequences[sid] = original
            
            if (i + 1) % 20 == 0:
                logger.info(
                    f"[{i+1}/{len(test_sids)}] elapsed {time.time()-t0:.0f}s, "
                    f"exp_bank={global_agent.exp_bank.stats()}"
                )
        
        except Exception as e:
            logger.warning(f"Student {sid} failed: {e}", exc_info=True)
            continue
    
    # 持久化经验库
    global_agent.save()
    
    # 输出报告
    summary = evaluator.summarize(global_agent_stats=global_agent.stats())
    summary["llm_stats"] = components["llm"].stats()
    summary["elapsed_seconds"] = time.time() - t0
    
    evaluator.save(summary, name=cfg["experiment"]["name"])
    
    trace_logger.close()
    
    logger.info(f"Pipeline finished in {time.time()-t0:.0f}s")
    logger.info(f"Summary: {json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float))}, indent=2, ensure_ascii=False)}")
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-students", type=int, default=None,
                        help="覆盖配置中的 max_students(便于快速调试)")
    parser.add_argument("--backend", type=str, default=None,
                        help="覆盖 LLM 后端(deepseek/openai_compat/modelscope/mock)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if args.max_students is not None:
        cfg.setdefault("evaluation", {})["max_students"] = args.max_students
    if args.backend:
        cfg.setdefault("llm", {})["default_backend"] = args.backend
    
    log_file = setup_logging(
        log_dir=cfg["experiment"]["log_dir"],
        run_name=cfg["experiment"]["name"],
        level="INFO",
    )
    logger.info(f"Log file: {log_file}")
    logger.info(f"Config: {args.config}")
    
    summary = run_pipeline(cfg)
    return summary


if __name__ == "__main__":
    main()
