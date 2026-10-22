"""Load the YAML pipeline configuration (configs/default.yaml)."""

from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base, override):
    out = deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None):
    """Default config, optionally overridden by a second YAML file."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()))
    return cfg
