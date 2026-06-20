"""
日志工具
- 控制台 + 文件双输出
- 结构化日志(JSONL)用于推荐过程追踪
"""
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any


_LOGGER_CACHE = {}


def setup_logging(log_dir: str, run_name: str = "default", level: str = "INFO"):
    """
    全局日志初始化,只需在程序入口调用一次
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{run_name}_{timestamp}.log"
    
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt))
    
    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(getattr(logging, level.upper()))
    
    return log_file


def get_logger(name: str) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    return logger


class JsonlLogger:
    """
    结构化日志(JSONL格式),用于记录每次推荐的详细数据
    便于后期 pandas 读取分析
    """
    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.path = log_path
        self.fh = open(log_path, "a", encoding="utf-8")
    
    def log(self, record: dict):
        record["_ts"] = datetime.now().isoformat()
        self.fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()
    
    def close(self):
        self.fh.close()
    
    def __del__(self):
        try:
            self.fh.close()
        except Exception:
            pass
