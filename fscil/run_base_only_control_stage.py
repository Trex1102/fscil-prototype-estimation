#!/usr/bin/env python3
"""Run the class-disjoint, base-session-only representation control.

The expensive strict-inductive table is only run if the diagnostic survives on
at least two datasets and the full-data prototype reference is non-trivial.  A
failed gate is a recorded scientific outcome, not a crashed job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from .constants import BACKBONES
from .utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--selection-episodes", type=int, default=200)
    parser.add_argument("--target-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override every dataset's documented default; zero keeps defaults.",
    )
    return parser.parse_args()


def run_logged(command: List[str], log_path: Path) -> Dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "runtime_sec": time.time() - started,
        "log": str(log_path),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    code_root = args.code_root.resolve()
    feature_root = project / "artifacts" / "features"
    checkpoint_root = project / "artifacts" / "base_only_checkpoints"
    protocol_root = project / "artifacts" / "protocol"
    result_root = project / "results" / "control_base_only_resnet18"
    log_root = project / "logs" / "control_base_only_resnet18"
    for path in (
        feature_root,
        checkpoint_root,
        protocol_root,
        result_root,
        log_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    status_path = result_root / "STATUS.json"
    status: Dict[str, object] = {
        "stage": "class_disjoint_base_only_control",
        "state": "running_diagnostics",
        "backbone": BACKBONES["base_only_resnet18"],
        "started_unix": time.time(),
        "selection_episodes": args.selection_episodes,
        "target_seeds": args.target_seeds,
        "jobs": [],
        "datasets": {},
        "decision_gate": {
            "rule": (
                "continue only if estimation-majority holds on >=2/3 datasets "
                "and A_n_full >=10% on those passing datasets"
            )
        },
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

    paths: Dict[str, Dict[str, Path]] = {}
    for dataset in ("cub200", "cifar100", "miniimagenet"):
        feature = feature_root / (
            "clip_feats_cls_{}_base_only_resnet18.pt".format(dataset)
        )
        checkpoint = checkpoint_root / (dataset + "_resnet18.pt")
        protocol = protocol_root / (
            "{}_base_only_resnet18_protocol.pt".format(dataset)
        )
        selection_bundle = protocol_root / (
            "{}_base_only_resnet18_protocol_base_selection.pt".format(dataset)
        )
        shrink_selection = result_root / (dataset + "_shrink_selection.json")
        shrink_audit = result_root / (dataset + "_shrink_audit.json")
        baseline_selection = result_root / (dataset + "_baseline_selection.json")
        baseline_audit = result_root / (dataset + "_baseline_audit.json")
        diagnostic = result_root / (dataset + "_diagnostic.json")
        inductive = result_root / (dataset + "_inductive.json")
        paths[dataset] = {
            "feature": feature,
            "protocol": protocol,
            "selection_bundle": selection_bundle,
            "shrink_selection": shrink_selection,
            "shrink_audit": shrink_audit,
            "baseline_selection": baseline_selection,
            "baseline_audit": baseline_audit,
            "diagnostic": diagnostic,
            "inductive": inductive,
        }

        if not feature.exists():
            command = [
                sys.executable,
                "-m",
                "fscil.train_base_only_backbone",
                "--dataset",
                dataset,
                "--code-root",
                str(code_root),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(feature),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ]
            if args.epochs > 0:
                command.extend(["--epochs", str(args.epochs)])
            execute(dataset, "train_and_extract", command)
        execute(
            dataset,
            "protocol",
            [
                sys.executable,
                "-m",
                "fscil.build_protocol_bundle",
                "--dataset",
                dataset,
                "--backbone-key",
                "base_only_resnet18",
                "--code-root",
                str(code_root),
                "--feature-cache",
                str(feature),
                "--output",
                str(protocol),
                "--selection-output",
                str(selection_bundle),
            ],
        )
        execute(
            dataset,
            "shrink_selection",
            [
                sys.executable,
                "-m",
                "fscil.selection",
                "select",
                "--bundle",
                str(selection_bundle),
                "--output",
                str(shrink_selection),
                "--episodes",
                str(args.selection_episodes),
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "shrink_audit",
            [
                sys.executable,
                "-m",
                "fscil.selection",
                "audit",
                "--bundle",
                str(selection_bundle),
                "--parameters",
                str(shrink_selection),
                "--output",
                str(shrink_audit),
                "--episodes",
                str(args.selection_episodes),
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "baseline_selection",
            [
                sys.executable,
                "-m",
                "fscil.baseline_selection",
                "select",
                "--bundle",
                str(selection_bundle),
                "--output",
                str(baseline_selection),
                "--episodes",
                str(args.selection_episodes),
                "--ranpac-episodes",
                "20",
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "baseline_audit",
            [
                sys.executable,
                "-m",
                "fscil.baseline_selection",
                "audit",
                "--bundle",
                str(selection_bundle),
                "--parameters",
                str(baseline_selection),
                "--output",
                str(baseline_audit),
                "--episodes",
                str(args.selection_episodes),
                "--ranpac-episodes",
                "20",
                "--device",
                args.device,
            ],
        )
        execute(
            dataset,
            "diagnostic",
            [
                sys.executable,
                "-m",
                "fscil.diagnostics",
                "--feature-cache",
                str(feature),
                "--protocol",
                str(protocol),
                "--output",
                str(diagnostic),
                "--seed-limit",
                str(args.target_seeds),
                "--bootstrap",
                str(args.bootstrap),
                "--device",
                args.device,
            ],
        )
        diagnosis = load_json(diagnostic)
        accounting = diagnosis["accounting_decomposition_pct"]
        utility = float(accounting["A_n_full"]) >= 10.0
        status["datasets"][dataset] = {
            "diagnosis_gate": bool(
                diagnosis["diagnosis_majority_gate_for_dataset"]
            ),
            "full_prototype_utility_gate": utility,
            "accounting_decomposition_pct": accounting,
            "inductive_state": "held_until_global_gate",
        }
        atomic_write_json(status_path, status)

    passing = [
        dataset
        for dataset, row in status["datasets"].items()
        if row["diagnosis_gate"] and row["full_prototype_utility_gate"]
    ]
    status["decision_gate"]["passing_datasets"] = passing
    status["decision_gate"]["passed"] = len(passing) >= 2
    if len(passing) < 2:
        status["state"] = "stopped_at_diagnosis_gate"
        status["finished_unix"] = time.time()
        status["runtime_sec"] = status["finished_unix"] - status["started_unix"]
        atomic_write_json(status_path, status)
        return

    status["state"] = "running_inductive_after_gate"
    atomic_write_json(status_path, status)
    methods = (
        "cosine,euclidean,covariance_lambda0,shared_lda,diagonal_lda,"
        "shrinksel,teen_official,ula_official"
    )
    for dataset in ("cub200", "cifar100", "miniimagenet"):
        item = paths[dataset]
        execute(
            dataset,
            "inductive",
            [
                sys.executable,
                "-m",
                "fscil.run_inductive",
                "--feature-cache",
                str(item["feature"]),
                "--protocol",
                str(item["protocol"]),
                "--shrink-parameters",
                str(item["shrink_selection"]),
                "--shrink-audit",
                str(item["shrink_audit"]),
                "--baseline-parameters",
                str(item["baseline_selection"]),
                "--baseline-audit",
                str(item["baseline_audit"]),
                "--output",
                str(item["inductive"]),
                "--seed-limit",
                str(args.target_seeds),
                "--methods",
                methods,
                "--device",
                args.device,
            ],
        )
        result = load_json(item["inductive"])
        status["datasets"][dataset]["inductive_state"] = "complete"
        status["datasets"][dataset]["summary"] = {
            method: float(values["session_average_mean"])
            for method, values in result["summary"].items()
        }
        atomic_write_json(status_path, status)

    status["state"] = "complete"
    status["finished_unix"] = time.time()
    status["runtime_sec"] = status["finished_unix"] - status["started_unix"]
    atomic_write_json(status_path, status)


if __name__ == "__main__":
    main()
