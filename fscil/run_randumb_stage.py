#!/usr/bin/env python3
"""Run the official-formula RanDumb-RBF cosine-NCM comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from .constants import BACKBONES, DATASETS
from .utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="required only for the original ViT-B/16 cache directory",
    )
    parser.add_argument(
        "--backbone-key",
        default="clip_vitb16",
        choices=("clip_vitb16", "clip_vitb32", "clip_rn50"),
    )
    parser.add_argument("--projection-dimension", type=int, default=10000)
    parser.add_argument("--audit-episodes", type=int, default=20)
    parser.add_argument("--target-seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_logged(command: List[str], log: Path) -> Dict[str, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "runtime_sec": time.time() - started,
        "log": str(log),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    token = {
        "clip_vitb16": "vitb16",
        "clip_vitb32": "vitb32",
        "clip_rn50": "rn50",
    }[args.backbone_key]
    if args.backbone_key == "clip_vitb16" and args.feature_root is None:
        raise ValueError("--feature-root is required for clip_vitb16")
    result_root = project / "results" / ("randumb_" + token)
    log_root = project / "logs" / ("randumb_" + token)
    protocol_root = project / "artifacts" / "protocol"
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / "STATUS.json"
    status: Dict[str, object] = {
        "stage": "randumb_rbf_ncm",
        "state": "running",
        "backbone": BACKBONES[args.backbone_key],
        "started_unix": time.time(),
        "projection_dimension": args.projection_dimension,
        "audit_episodes": args.audit_episodes,
        "target_seeds": args.target_seeds,
        "jobs": [],
        "datasets": {},
    }
    atomic_write_json(status_path, status)

    def execute(dataset: str, name: str, command: List[str]) -> None:
        job = run_logged(command, log_root / (dataset + "_" + name + ".log"))
        status["jobs"].append(job)
        atomic_write_json(status_path, status)
        if job["returncode"] != 0:
            status["state"] = "failed_" + name
            status["failed_dataset"] = dataset
            atomic_write_json(status_path, status)
            raise SystemExit(int(job["returncode"]))

    for dataset in ("cub200", "cifar100", "miniimagenet"):
        selection_bundle = protocol_root / (
            dataset + "_{}_protocol_base_selection.pt".format(args.backbone_key)
        )
        protocol = protocol_root / (
            dataset + "_{}_protocol.pt".format(args.backbone_key)
        )
        if args.backbone_key == "clip_vitb16":
            feature = args.feature_root.resolve() / DATASETS[dataset]["feature_cache"]
        else:
            feature = project / "artifacts" / "features" / (
                "clip_feats_cls_{}_{}.pt".format(dataset, args.backbone_key)
            )
        parameters = result_root / (dataset + "_parameters.json")
        audit = result_root / (dataset + "_audit.json")
        target = result_root / (dataset + "_target.json")
        execute(
            dataset,
            "lock",
            [
                sys.executable,
                "-m",
                "fscil.randumb_ncm",
                "lock",
                "--bundle",
                str(selection_bundle),
                "--output",
                str(parameters),
                "--projection-dimension",
                str(args.projection_dimension),
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "audit",
            [
                sys.executable,
                "-m",
                "fscil.randumb_ncm",
                "audit",
                "--bundle",
                str(selection_bundle),
                "--parameters",
                str(parameters),
                "--output",
                str(audit),
                "--episodes",
                str(args.audit_episodes),
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "target",
            [
                sys.executable,
                "-m",
                "fscil.randumb_ncm",
                "target",
                "--feature-cache",
                str(feature),
                "--protocol",
                str(protocol),
                "--parameters",
                str(parameters),
                "--audit",
                str(audit),
                "--output",
                str(target),
                "--seed-limit",
                str(args.target_seeds),
                "--device",
                args.device,
            ],
        )
        audit_data = load_json(audit)
        target_data = load_json(target)
        status["datasets"][dataset] = {
            "audit_minus_cosine": audit_data["selected_minus_cosine"],
            "audit_balanced_non_regression": audit_data[
                "passes_balanced_non_regression_gate"
            ],
            "session_average_mean": target_data["summary"][
                "session_average_mean"
            ],
            "final_accuracy_mean": target_data["summary"]["final_accuracy_mean"],
        }
        atomic_write_json(status_path, status)
    status["state"] = "complete"
    status["finished_unix"] = time.time()
    status["runtime_sec"] = status["finished_unix"] - status["started_unix"]
    atomic_write_json(status_path, status)


if __name__ == "__main__":
    main()
