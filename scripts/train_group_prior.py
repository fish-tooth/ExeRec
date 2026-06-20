"""
独立训练群体先验(GMM)
用法:
    python scripts/train_group_prior.py --config configs/default.yaml --sample 1000

之后 run_pipeline.py 会自动加载训练好的 group_prior.pkl
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config, get_logger, setup_logging, set_seed
from llm_adapter import LLMFactory
from data_processing import QuestionBank, InteractionLoader, EmbeddingStore
from student_agent import (
    DKTSimulator, ProfileGenerator, GroupPriorModule, StudentAgent
)

logger = get_logger("scripts.train_gp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    setup_logging(cfg["experiment"]["log_dir"], run_name="train_group_prior")
    set_seed(cfg["experiment"].get("seed", 42))
    
    qb = QuestionBank(
        question_file=cfg["data"]["question_file"],
        qid_to_int_file=cfg["data"]["qid_to_int_file"],
    )
    itx_train = InteractionLoader(cfg["data"]["train_valid_seq"])
    itx_test = InteractionLoader(cfg["data"]["test_seq"])
    es = EmbeddingStore(
        question_content_emb_path=cfg["data"]["question_content_emb"],
        question_int_emb_path=cfg["data"]["question_int_emb"],
        kc_emb_path=cfg["data"]["kc_emb"],
        qid_to_int=qb.qid_to_int,
    )
    llm = LLMFactory.from_config(cfg)
    dkt = DKTSimulator(
        model_dir=cfg["dkt"]["model_dir"],
        device=cfg["dkt"].get("device", "cuda:0"),
    )
    profile_gen = ProfileGenerator(llm)
    gp = GroupPriorModule(
        n_clusters=cfg["student_agent"]["group_prior"].get("n_clusters", 8),
    )
    
    student_agent = StudentAgent(
        cfg=cfg, question_bank=qb,
        interaction_train=itx_train, interaction_test=itx_test,
        embedding_store=es, dkt=dkt, profile_generator=profile_gen,
        group_prior=gp,
    )
    
    logger.info(f"Fitting group prior with {args.sample} students...")
    student_agent.fit_group_prior(sample_size=args.sample)
    
    save_path = Path(cfg["experiment"]["log_dir"]) / "group_prior.pkl"
    gp.save(str(save_path))
    logger.info(f"Saved group prior to {save_path}")


if __name__ == "__main__":
    main()
