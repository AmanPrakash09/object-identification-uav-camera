"""Resolve CARLA's Python API (.egg) and prepend it to ``sys.path`` for all project scripts."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _config_candidates() -> list[Path]:
    root = _repo_root()
    return [
        root / "config" / "carla_api.local.json",
        root / "config" / "carla_api.json",
        root / "carla_paths.json",
    ]


def _load_json_config() -> dict:
    for path in _config_candidates():
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    return {}


def _egg_glob_pattern(dist_dir: Path) -> str:
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    plat = "win-amd64" if os.name == "nt" else "linux-x86_64"
    return str(dist_dir / f"carla-*{py}-{plat}.egg")


def _append_egg(path: Path) -> bool:
    if path.is_file() and path.suffix.lower() == ".egg":
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.append(resolved)
        return True
    return False


def ensure_carla_on_path() -> None:
    """Ensure ``import carla`` can resolve the official pre-built API egg.

    Resolution order:

    1. ``CARLA_PYTHON_EGG`` — full path to the ``.egg`` file.
    2. ``CARLA_DIST_DIR`` — directory containing the matching ``carla-*.egg``.
    3. JSON config: ``config/carla_api.local.json``, then ``config/carla_api.json``,
       then legacy ``carla_paths.json`` in the repo root — each may set ``egg`` or ``dist_dir``.
    4. Default install layout under the user profile ``Downloads/CARLA_0.9.10/.../dist``.
    """
    try:
        import carla  # noqa: F401
    except ImportError:
        pass
    else:
        return

    egg = os.environ.get("CARLA_PYTHON_EGG", "").strip()
    if egg and _append_egg(Path(egg)):
        return

    dist_env = os.environ.get("CARLA_DIST_DIR", "").strip()
    if dist_env:
        dist = Path(dist_env).expanduser().resolve()
        for hit in glob.glob(_egg_glob_pattern(dist)):
            if _append_egg(Path(hit)):
                return

    cfg = _load_json_config()

    egg_cfg = (cfg.get("egg") or "").strip()
    if egg_cfg and _append_egg(Path(egg_cfg).expanduser()):
        return

    dist_cfg = (cfg.get("dist_dir") or "").strip()
    if dist_cfg:
        dist = Path(dist_cfg).expanduser().resolve()
        for hit in glob.glob(_egg_glob_pattern(dist)):
            if _append_egg(Path(hit)):
                return

    default_dist = (
        _repo_root().parents[3]
        / "Downloads"
        / "CARLA_0.9.10"
        / "WindowsNoEditor"
        / "PythonAPI"
        / "carla"
        / "dist"
    )
    for hit in glob.glob(_egg_glob_pattern(default_dist)):
        if _append_egg(Path(hit)):
            return
