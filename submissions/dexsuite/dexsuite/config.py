"""Tiny config loader: reads config.yaml (if present) into typed parameter
objects. Falls back to built-in defaults so the package runs with zero config.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from .env import EnvConfig
from .skills import SkillParams

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")


def load(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def env_config(cfg: Dict[str, Any] | None = None) -> EnvConfig:
    cfg = cfg if cfg is not None else load()
    e = cfg.get("env", {})
    return EnvConfig(seed=e.get("seed", 0),
                     render_w=e.get("render_w", 1280),
                     render_h=e.get("render_h", 720))


def skill_params(cfg: Dict[str, Any] | None = None) -> SkillParams:
    cfg = cfg if cfg is not None else load()
    s = cfg.get("skills", {})
    return SkillParams(**{k: s[k] for k in s if k in SkillParams.__dataclass_fields__})
