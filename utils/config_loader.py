"""
配置加载器
- 支持 _base_ 字段做继承(子配置覆盖父配置)
- 支持环境变量插值(${ENV:VAR_NAME})
"""
import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base, override 优先"""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_env_vars(obj: Any) -> Any:
    """支持 ${ENV:VAR_NAME} 语法从环境变量读取"""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${ENV:") and obj.endswith("}"):
        var_name = obj[6:-1]
        return os.environ.get(var_name, "")
    return obj


def load_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件,支持 _base_ 继承
    
    Args:
        config_path: 配置文件路径(可绝对可相对)
    
    Returns:
        合并后的配置字典
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        cfg = {}
    
    # 处理 _base_ 继承
    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = config_path.parent / base_name
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, cfg)
    
    cfg = _resolve_env_vars(cfg)
    return cfg


def save_config(cfg: dict, path: str):
    """保存合并后的最终配置(便于实验复现)"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
