#!/usr/bin/env python
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
METHODS = os.path.join(CONFIG_DIR, "methods.yaml")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_card(path: str | None = None) -> dict:
    """The data card: $WALKTHROUGH_CARD unless a path is given."""
    path = path or os.environ.get("WALKTHROUGH_CARD")
    if not path:
        raise SystemExit("no data card: pass --card or set WALKTHROUGH_CARD")
    return load_yaml(path)


def load_methods(path: str | None = None) -> dict:
    return load_yaml(path or os.environ.get("WALKTHROUGH_METHODS") or METHODS)


def run_root() -> str:
    """This run's directory. Absolute."""
    env = os.environ.get("WALKTHROUGH_RUN")
    return os.path.abspath(env) if env else os.path.join(REPO_ROOT, "runs", "default")


RUN_ROOT = run_root()


def rpath(*parts) -> str:
    """A path inside this run: rpath('storm_out', tag)."""
    return os.path.join(RUN_ROOT, *parts)


def cpath(*parts) -> str:
    """A path inside the repository: shared references, caches, code."""
    return os.path.join(REPO_ROOT, *parts)


if __name__ == "__main__":
    print(f"REPO_ROOT {REPO_ROOT}")
    print(f"RUN_ROOT  {RUN_ROOT}")
    for n in ("cellbin", "prepped", "storm_out", "celltypes_qc", "composition",
              "program_activity", "deg_out"):
        print(f"  {'ok ' if os.path.isdir(rpath(n)) else 'MISSING'} {rpath(n)}")
