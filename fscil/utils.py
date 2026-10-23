"""Small deterministic utilities shared by the audited experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_revision(root: Path) -> Dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(to_builtin(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor.float(), dim=-1)


def normalized_mean(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor.float().mean(0), dim=0)


def resolve_dataset_items(values: Any, code_root: Path) -> np.ndarray:
    """Make repository-relative image paths safe for DataLoader workers.

    Array-backed datasets (for example CIFAR) are returned unchanged.  Path-
    backed datasets are resolved once in the parent process, so their behavior
    cannot depend on the experiment launch directory.
    """

    array = np.asarray(values)
    if array.ndim == 1 and array.size:
        first = array[0]
        if isinstance(first, (str, np.str_, Path)):
            root = code_root.resolve()
            resolved = []
            for value in array.tolist():
                item = Path(str(value))
                resolved.append(str(item if item.is_absolute() else root / item))
            return np.asarray(resolved)
    return array


def backbone_display(backbone: Dict[str, Any]) -> str:
    return "{} {} visual encoder only".format(
        backbone.get("family", "frozen backbone"),
        backbone.get("architecture", "unknown"),
    )
