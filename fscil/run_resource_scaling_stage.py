#!/usr/bin/env python3
"""Run each cohort mode separately so latency and memory remain mode-resolved."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .run_qpr_benchmark_queue import LOCKED_FINAL_SMOKE
from .utils import atomic_write_json, stable_json_sha256



DATASETS = ("cub200", "cifar100", "miniimagenet")

COMPONENT_METHODS = ("shrinksel", "qpr_selected")

ROBUSTNESS_MODES = (
    "full",
    "balanced1",
    "balanced2",
    "balanced5",
    "balanced10",
    "balanced20",
    "random64",
    "random128",
    "random512",
    "random2048",
    "causal128x32",
    "causal512x128",
    "causal2048x256",
    "imbalanced10",
    "imbalanced100",
)

COMPARATOR_METHODS = (
    "shrinksel",
    "qpr",
    "tim_gd",
    "laplacian_shot",
    "laplacian_shot_core",
    "protolp_paper",
    "ptmap_uniform",
)

COMPARATOR_MODES = ("full", "random512", "causal512x128", "imbalanced10")

def _parse_datasets(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("--datasets must be a non-empty unique list")
    unknown = sorted(set(values).difference(DATASETS))
    if unknown:
        raise ValueError("unknown datasets {}".format(unknown))
    return values


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _valid_embedded_hash(payload: Mapping[str, object]) -> bool:
    expected = str(payload["result_sha256"])
    copy_payload = copy.deepcopy(dict(payload))
    copy_payload.pop("result_sha256")
    return stable_json_sha256(copy_payload) == expected


def _run(command: Sequence[str], log_path: Path) -> Dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    return {
        "command": list(command),
        "log": str(log_path),
        "returncode": int(completed.returncode),
        "runtime_sec": time.time() - started,
    }


def _resource_snapshot(payload: Mapping[str, object]) -> Dict[str, object]:
    resources = payload["resources"]
    return {
        method: {
            "runtime_sec": float(values["runtime_sec"]),
            "peak_allocated_bytes": int(values["peak_allocated_bytes"]),
            "calls": int(values["calls"]),
            "largest_cohort": (
                int(values["largest_cohort"])
                if "largest_cohort" in values
                else None
            ),
        }
        for method, values in resources.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument(
        "--feature-template", default="clip_feats_cls_{dataset}_vitb16.pt"
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--shrink-root", required=True, type=Path)
    parser.add_argument("--qpr-root", required=True, type=Path)
    parser.add_argument("--comparator-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qpr-seeds", type=int, default=5)
    parser.add_argument("--comparator-seeds", type=int, default=5)
    parser.add_argument("--anchor-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    datasets = _parse_datasets(args.datasets)
    project = args.project_root.resolve()
    feature_root = args.feature_root.resolve()
    protocol_root = (args.protocol_root or project / "artifacts" / "protocol").resolve()
    shrink_root = args.shrink_root.resolve()
    qpr_root = args.qpr_root.resolve()
    comparator_root = args.comparator_root.resolve()
    output_root = args.output_root.resolve()
    log_root = project / "logs" / output_root.name
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "STATUS.json"
    status: Dict[str, object] = {
        "schema_version": 1,
        "stage": "mode_resolved_transductive_resource_scaling",
        "state": "running",
        "started_unix": time.time(),
        "datasets": list(datasets),
        "qpr_modes": list(ROBUSTNESS_MODES),
        "comparator_modes": list(COMPARATOR_MODES),
        "qpr_support_draws": int(args.qpr_seeds),
        "comparator_support_draws": int(args.comparator_seeds),
        "jobs": [],
        "results": {},
    }
    atomic_write_json(status_path, status)

    def paths(dataset: str) -> Dict[str, Path]:
        return {
            "feature": feature_root / args.feature_template.format(dataset=dataset),
            "protocol": protocol_root / (dataset + "_clip_vitb16_protocol.pt"),
            "shrink": shrink_root / (dataset + "_selection.json"),
            "shrink_audit": shrink_root / (dataset + "_audit.json"),
            "qpr": qpr_root / (dataset + "_selection.json"),
            "qpr_audit": qpr_root / (dataset + "_audit.json"),
            "transductive": comparator_root / "base_selection" / (dataset + "_selection.json"),
            "transductive_audit": comparator_root / "base_selection" / (dataset + "_audit.json"),
        }

    for dataset in datasets:
        missing = [str(path) for path in paths(dataset).values() if not path.is_file()]
        if missing:
            status["state"] = "failed_missing_inputs"
            status["missing_inputs"] = missing
            atomic_write_json(status_path, status)
            raise FileNotFoundError(", ".join(missing))

    def execute(dataset: str, name: str, command: Sequence[str]) -> Mapping[str, object]:
        job = _run(command, log_root / (dataset + "_" + name + ".log"))
        job["dataset"] = dataset
        job["name"] = name
        status["jobs"].append(job)
        atomic_write_json(status_path, status)
        if int(job["returncode"]) != 0:
            status["state"] = "failed_" + name
            status["failed_dataset"] = dataset
            atomic_write_json(status_path, status)
            raise SystemExit(int(job["returncode"]))
        return job

    for dataset in datasets:
        item = paths(dataset)
        dataset_root = output_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_status: Dict[str, object] = {"qpr": {}, "comparators": {}}
        status["results"][dataset] = dataset_status
        qpr_common = [
            "--feature-cache", str(item["feature"]),
            "--protocol", str(item["protocol"]),
            "--shrink-parameters", str(item["shrink"]),
            "--qpr-parameters", str(item["qpr"]),
            "--qpr-audit", str(item["qpr_audit"]),
            "--device", args.device,
            "--sessions", "final",
            "--methods", ",".join(COMPONENT_METHODS),
            "--seed-limit", str(args.qpr_seeds),
        ]
        for mode in ROBUSTNESS_MODES:
            output = dataset_root / ("qpr_" + mode + ".json")
            execute(
                dataset,
                "qpr_" + mode,
                [
                    sys.executable,
                    "-m",
                    "fscil.run_qpr",
                    *qpr_common,
                    "--modes",
                    mode,
                    "--output",
                    str(output),
                ],
            )
            payload = _load(output)
            if not _valid_embedded_hash(payload):
                raise RuntimeError("resource output hash mismatch: " + str(output))
            if mode == "full":
                row = payload["per_seed"][0]["sessions"][0]["modes"][mode]["methods"]
                observed = {
                    "shrinksel": float(row["shrinksel"]["overall"]),
                    "qpr": float(row["qpr_selected"]["overall"]),
                }
                expected = LOCKED_FINAL_SMOKE[dataset]
                if any(
                    not math.isfinite(observed[key])
                    or abs(observed[key] - float(expected[key]))
                    > float(args.anchor_tolerance)
                    for key in observed
                ):
                    status["state"] = "stopped_at_qpr_anchor"
                    status["failed_dataset"] = dataset
                    status["anchor"] = {"observed": observed, "expected": expected}
                    atomic_write_json(status_path, status)
                    raise SystemExit(4)
                dataset_status["anchor"] = {"observed": observed, "expected": expected}
            dataset_status["qpr"][mode] = {
                "output": str(output),
                "result_sha256": payload["result_sha256"],
                "resources": _resource_snapshot(payload),
            }
            atomic_write_json(status_path, status)

        comparator_common = [
            "--feature-cache", str(item["feature"]),
            "--protocol", str(item["protocol"]),
            "--shrink-parameters", str(item["shrink"]),
            "--radial-audit", str(item["shrink_audit"]),
            "--transductive-parameters", str(item["transductive"]),
            "--transductive-audit", str(item["transductive_audit"]),
            "--qpr-parameters", str(item["qpr"]),
            "--qpr-audit", str(item["qpr_audit"]),
            "--device", args.device,
            "--sessions", "final",
            "--methods", ",".join(COMPARATOR_METHODS),
            "--seed-limit", str(args.comparator_seeds),
        ]
        for mode in COMPARATOR_MODES:
            output = dataset_root / ("comparators_" + mode + ".json")
            execute(
                dataset,
                "comparators_" + mode,
                [
                    sys.executable,
                    "-m",
                    "fscil.run_transductive",
                    *comparator_common,
                    "--modes",
                    mode,
                    "--output",
                    str(output),
                ],
            )
            payload = _load(output)
            if not _valid_embedded_hash(payload):
                raise RuntimeError("resource output hash mismatch: " + str(output))
            dataset_status["comparators"][mode] = {
                "output": str(output),
                "result_sha256": payload["result_sha256"],
                "resources": _resource_snapshot(payload),
            }
            atomic_write_json(status_path, status)

    status["state"] = "complete"
    status["finished_unix"] = time.time()
    status["runtime_sec"] = status["finished_unix"] - status["started_unix"]
    status["result_sha256"] = stable_json_sha256(status)
    atomic_write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
