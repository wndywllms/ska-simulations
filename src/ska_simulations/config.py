"""
Configuration handling for ska-simulations.

A run is described by a YAML file that is deep-merged on top of the packaged
``default_config.yaml``. Unknown keys are rejected so typos are caught early.
"""
from __future__ import annotations

import copy
import os
import re
from importlib import resources
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG_RESOURCE = "default_config.yaml"

# Pipeline steps, in execution order (see ska_simulations.pipeline).
STEPS = ("sky", "sensitivity", "oskar", "dp3", "msoverview", "image_dirty", "mask",
         "image_clean", "find_sources")

# Sub-trees whose keys are free-form (user may add new entries).
_FREE_FORM = {
    ("telescope_models",),
    ("sensitivity", "subarray"),
    ("simulator", "extra_settings"),
}


# PyYAML follows YAML 1.1, which reads 1e-3 or 120.0e6 (no dot / no exponent sign)
# as strings. Values that look like that are converted to floats.
_SCI_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _coerce_numbers(node):
    if isinstance(node, dict):
        return {k: _coerce_numbers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_numbers(v) for v in node]
    if isinstance(node, str) and _SCI_FLOAT.match(node.strip()):
        return float(node)
    return node


class ConfigError(ValueError):
    """Raised for invalid configuration files or values."""


def default_config_text() -> str:
    """Return the packaged default configuration as text (with comments)."""
    return resources.files("ska_simulations").joinpath("data", DEFAULT_CONFIG_RESOURCE).read_text()


def default_config() -> dict:
    return yaml.safe_load(default_config_text())


def _merge(base: dict, override: dict, path: tuple = ()) -> dict:
    out = copy.deepcopy(base)
    free = any(path[: len(p)] == p for p in _FREE_FORM)
    for key, val in (override or {}).items():
        here = path + (key,)
        if key not in out and not free:
            raise ConfigError(f"Unknown configuration key: {'.'.join(map(str, here))}")
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val, here)
        else:
            out[key] = copy.deepcopy(val)
    return out


def deep_update(base: dict, *others: dict) -> dict:
    """Plain recursive dict merge (no key checking); returns a new dict."""
    out = copy.deepcopy(base)
    for other in others:
        for key, val in (other or {}).items():
            if isinstance(val, dict) and isinstance(out.get(key), dict):
                out[key] = deep_update(out[key], val)
            else:
                out[key] = copy.deepcopy(val)
    return out


def parse_overrides(items: Iterable[str]) -> dict:
    """Turn ``["a.b=1", "c=foo"]`` into a nested dict (values parsed as YAML)."""
    result: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Load a run configuration.

    Parameters
    ----------
    path      : YAML file with the run settings (None -> defaults only)
    overrides : nested dict applied last (e.g. from ``--set`` on the CLI)
    """
    cfg = default_config()
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise ConfigError(f"{path} does not contain a YAML mapping")
        cfg = _merge(cfg, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg = _coerce_numbers(cfg)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    obs = cfg["observation"]
    if obs["array"] not in cfg["telescope_models"]:
        raise ConfigError(
            f"observation.array={obs['array']!r} has no entry in telescope_models "
            f"(known: {', '.join(cfg['telescope_models'])})"
        )
    for key in ("nhours", "time_step_s", "freq_step_khz", "start_freq_mhz"):
        if not obs[key] or obs[key] <= 0:
            raise ConfigError(f"observation.{key} must be > 0")
    if int(obs["nchan"]) < 1 or int(obs["nsubbands"]) < 1:
        raise ConfigError("observation.nchan and observation.nsubbands must be >= 1")
    if obs["nsubbands"] > obs["nchan"]:
        raise ConfigError("observation.nsubbands cannot exceed observation.nchan")
    if cfg["sky"]["model"] not in ("LoTSSDEEP", "COSMOS", "testsky", "cal", "none"):
        raise ConfigError(f"Unknown sky.model {cfg['sky']['model']!r}")
    if cfg["run"]["overwrite"] not in ("ask", "yes", "no"):
        raise ConfigError("run.overwrite must be one of ask | yes | no")
    bad = [s for s in cfg["run"]["steps"] if s not in STEPS]
    if bad:
        raise ConfigError(f"Unknown run.steps {bad}; valid steps are {list(STEPS)}")


def resolve_path(path: str | None, base: str | None) -> str | None:
    """Expand ~ and $VARS, and make relative paths relative to ``base``."""
    if path is None or str(path) in ("", "None", "none"):
        return None
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if base and not os.path.isabs(path):
        path = os.path.join(os.path.expanduser(os.path.expandvars(str(base))), path)
    return os.path.normpath(path)


def data_path(cfg: dict, name: str | None) -> str | None:
    return resolve_path(name, cfg["paths"]["data_dir"])


def telescope_model_path(cfg: dict) -> str:
    array = cfg["observation"]["array"]
    models = cfg["telescope_models"][array]
    key = "cable_errors" if cfg["simulator"]["cable_errors"] else "default"
    name = models.get(key)
    if not name:
        raise ConfigError(f"No '{key}' telescope model configured for array {array}")
    return resolve_path(name, cfg["paths"]["telescope_dir"])


def noise_file_paths(cfg: dict) -> tuple[str, str]:
    obs = cfg["observation"]
    fields = {k: v for k, v in obs.items() if isinstance(v, (int, float, str))}
    noise = cfg["simulator"]["noise"]
    return (
        data_path(cfg, noise["freq_file"].format(**fields)),
        data_path(cfg, noise["rms_file"].format(**fields)),
    )


def dump_config(cfg: dict, path: str, header: str | None = None) -> None:
    with open(path, "w") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
