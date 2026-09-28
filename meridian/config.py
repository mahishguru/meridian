"""Configuration loader.

Loads a YAML file into a dotted-attribute namespace so callers can write
`cfg.optimizer.dante.c0` instead of `cfg["optimizer"]["dante"]["c0"]`.

Supports `--override key.subkey=value` style flat overrides.

String values undergo `${VAR}` environment-variable expansion at load
time. Two roots are auto-set if not already exported:

* ``MERIDIAN_ROOT``: this repository (configs, seed caches, DAMASK inputs);
* ``MSED_ROOT``: the encoder-decoder weights and texture priors
  (default ``$MERIDIAN_ROOT/weights``, filled by ``scripts/download_weights.sh``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# <MERIDIAN_ROOT>/meridian/config.py
os.environ.setdefault("MERIDIAN_ROOT", str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MSED_ROOT", str(Path(os.environ["MERIDIAN_ROOT"]) / "weights"))

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(s: str) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        return os.environ.get(name, m.group(0))
    return _VAR_RE.sub(sub, s)


class ConfigNode(dict):
    """Dict that also exposes its keys as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_node(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigNode({k: _to_node(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_node(x) for x in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def load_config(path: str | Path) -> ConfigNode:
    """Load a YAML config file into a ConfigNode."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_node(raw)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def apply_overrides(cfg: ConfigNode, overrides: list[str]) -> ConfigNode:
    """Apply `key.subkey=value` overrides. Values are parsed as YAML scalars."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key=value', got {ov!r}")
        key, val = ov.split("=", 1)
        parsed = yaml.safe_load(val)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = ConfigNode()
            node = node[p]
        node[parts[-1]] = parsed
    return cfg
