"""Run configuration, seeding, and the run manifest.

The lab handouts require every quantitative claim to carry its conditions:
which sequence, how many trials, what seed, what hardware. Rather than asking
you to remember that, `RunManifest` captures it automatically and writes it
next to your results. Cite the manifest in your report and the conditions
requirement is satisfied.
"""
from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch, including CUDA.

    Note this makes runs reproducible, NOT deterministic: several CUDA kernels
    are nondeterministic regardless of seed. If you need bitwise determinism,
    also set torch.use_deterministic_algorithms(True) and the CUBLAS workspace
    env var -- and expect a slowdown. Say which you used in your report.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


@dataclass
class RunManifest:
    """Everything a grader needs to know to interpret one number.

    Usage:
        m = RunManifest(run_name="icp_basin", seed=0, notes="20 inits x 6 levels")
        ... do work ...
        m.add_result("median_ate_m", 0.031)
        m.save(Path("results/icp_basin"))
    """

    run_name: str
    seed: int = 0
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    # Captured automatically.
    hostname: str = field(default_factory=socket.gethostname)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform_: str = field(default_factory=platform.platform)
    numpy: str = field(default_factory=lambda: np.__version__)
    gpu: str | None = field(default_factory=_gpu_name)
    git_commit: str | None = field(default_factory=_git_commit)

    def __post_init__(self) -> None:
        set_global_seed(self.seed)

    def add_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def add_result(self, key: str, value: Any) -> None:
        self.results[key] = value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    def conditions_line(self) -> str:
        """A one-line conditions string suitable for a table caption.

        The handout contrasts "ICP: 0.03" (worth nothing) with a row carrying
        seed, trial count, and hardware (worth full marks). This builds the
        second kind.
        """
        bits = [f"seed={self.seed}"]
        bits += [f"{k}={v}" for k, v in self.params.items()]
        if self.gpu:
            bits.append(self.gpu)
        else:
            bits.append(platform.processor() or "CPU")
        return ", ".join(bits)
