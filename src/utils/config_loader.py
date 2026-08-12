"""
Loads config/config.yaml and resolves data file paths relative to the
project root, regardless of whether the caller is a notebook in /notebooks
or a script in /src.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def raw_path(key: str, cfg: dict | None = None) -> Path:
    """Return the full path to a raw dataset by its config key, e.g. raw_path('scada')."""
    cfg = cfg or load_config()
    filename = cfg["raw_files"][key]
    return PROJECT_ROOT / cfg["paths"]["raw_dir"] / filename


def processed_path(filename: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["processed_dir"] / filename


def interim_path(filename: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["interim_dir"] / filename
