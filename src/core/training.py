"""Shared training utilities (device, seed, checkpoint helpers)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make Python, NumPy and PyTorch runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only host
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return output


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model_state, "config": config}
    if extra:
        payload["extra"] = extra
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
