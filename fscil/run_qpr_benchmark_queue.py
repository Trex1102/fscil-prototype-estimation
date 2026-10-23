#!/usr/bin/env python3
"""Regenerate comparator selection, smoke-test, then run the QPR benchmark.

This runner is intentionally fail-closed: the twenty-draw run never starts unless
the class-disjoint base audit passes and the one-draw target smoke reproduces the
locked ShrinkSel/QPR anchors while producing a finite, non-degenerate ProtoLP row.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .constants import DATASETS
from .utils import atomic_write_json


METHODS = (
    "shrinksel",
    "qpr",
    "tim_gd",
    "laplacian_shot",
    "laplacian_shot_core",
    "protolp_paper",
    "ptmap_uniform",
)

LOCKED_FINAL_SMOKE = {
    "cub200": {"shrinksel": 73.351741, "qpr": 75.543666},
    "cifar100": {"shrinksel": 67.799997, "qpr": 71.819997},
    "miniimagenet": {"shrinksel": 88.269997, "qpr": 90.629995},
}


def run(command: Sequence[str], log_path: Path) -> Dict[str, object]:
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


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_datasets(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    allowed = tuple(LOCKED_FINAL_SMOKE)
    if not values or len(set(values)) != len(values):
        raise ValueError("--datasets must be a non-empty unique list")
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError("unknown benchmark datasets {}".format(unknown))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--datasets", default="cub200,cifar100,miniimagenet")
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--shrink-root", type=Path)
    parser.add_argument(
        "--shrink-name-template", default="{dataset}_shrink_selection.json"
    )
    parser.add_argument(
        "--shrink-audit-name-template", default="{dataset}_shrink_audit.json"
    )
    parser.add_argument("--qpr-root", type=Path)
    parser.add_argument(
        "--qpr-name-template",
        default="{dataset}_{kind}.json",
        help=(
            "Filename template under --qpr-root; fields are dataset and kind "
            "(selection/audit)."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-episodes", type=int, default=20)
    parser.add_argument("--audit-episodes", type=int, default=20)
    parser.add_argument("--native-episodes", type=int, default=1000)
    parser.add_argument("--anchor-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--minimum-protolp-accuracy",
        type=float,
        default=5.0,
        help="Numerical-degeneracy guard only, not a target performance gate.",
    )
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    project = args.project_root.resolve()
    feature_root = args.feature_root.resolve()
    protocol_root = (args.protocol_root or project / "artifacts" / "protocol").resolve()
    shrink_root = (args.shrink_root or project / "results" / "primary_vitb16").resolve()
    qpr_root = (args.qpr_root or project / "results" / "qpr_vitb16").resolve()
    try:
        args.qpr_name_template.format(dataset="cub200", kind="selection")
    except (KeyError, ValueError) as error:
        raise ValueError("invalid --qpr-name-template") from error
    output_root = (
        args.output_root or project / "results" / "qpr_transductive_benchmark_v7"
    ).resolve()
    selection_root = output_root / "base_selection"
    smoke_root = output_root / "smoke"
    benchmark_root = output_root / "benchmark"
    native_root = output_root / "native_sanity"
    log_root = project / "logs" / output_root.name
    status_path = output_root / "STATUS.json"
    output_root.mkdir(parents=True, exist_ok=True)
    status: Dict[str, object] = {
        "schema_version": 1,
        "stage": "qpr_matched_transductive_queue",
        "state": "running_base_selection",
        "started_unix": time.time(),
        "datasets": list(datasets),
        "methods": list(METHODS),
        "support_draws": 20,
        "sessions": "all",
        "modes": ["full"],
        "jobs": [],
        "smoke": {},
        "native_sanity": {},
        "roots": {
            "project": str(project),
            "features": str(feature_root),
            "protocol": str(protocol_root),
            "shrink": str(shrink_root),
            "qpr": str(qpr_root),
            "output": str(output_root),
        },
    }
    atomic_write_json(status_path, status)

    def qpr_path(dataset: str, kind: str) -> Path:
        return qpr_root / args.qpr_name_template.format(
            dataset=dataset, kind=kind
        )

    def shrink_path(dataset: str) -> Path:
        return shrink_root / args.shrink_name_template.format(dataset=dataset)

    def shrink_audit_path(dataset: str) -> Path:
        return shrink_root / args.shrink_audit_name_template.format(
            dataset=dataset
        )

    required_inputs = []
    for dataset in datasets:
        required_inputs.extend(
            [
                feature_root / DATASETS[dataset]["feature_cache"],
                protocol_root / (dataset + "_clip_vitb16_protocol.pt"),
                protocol_root
                / (dataset + "_clip_vitb16_protocol_base_selection.pt"),
                shrink_path(dataset),
                shrink_audit_path(dataset),
                qpr_path(dataset, "selection"),
                qpr_path(dataset, "audit"),
            ]
        )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        status["state"] = "failed_missing_inputs"
        status["missing_inputs"] = missing
        atomic_write_json(status_path, status)
        raise FileNotFoundError("missing QPR benchmark inputs: " + ", ".join(missing))

    def execute(dataset: str, name: str, command: List[str]) -> None:
        job = run(command, log_root / (dataset + "_" + name + ".log"))
        status["jobs"].append(job)
        atomic_write_json(status_path, status)
        if job["returncode"] != 0:
            status["state"] = "failed_" + name
            status["failed_dataset"] = dataset
            atomic_write_json(status_path, status)
            raise SystemExit(int(job["returncode"]))

    for dataset in datasets:
        bundle = protocol_root / (
            dataset + "_clip_vitb16_protocol_base_selection.pt"
        )
        selection = selection_root / (dataset + "_selection.json")
        audit = selection_root / (dataset + "_audit.json")
        execute(
            dataset,
            "selection",
            [
                sys.executable,
                "-m",
                "fscil.transductive_selection",
                "select",
                "--bundle",
                str(bundle),
                "--output",
                str(selection),
                "--episodes",
                str(args.selection_episodes),
                "--device",
                args.device,
                "--families",
                "laplacian,laplacian_core,ptmap,protolp_paper",
            ],
        )
        execute(
            dataset,
            "audit",
            [
                sys.executable,
                "-m",
                "fscil.transductive_selection",
                "audit",
                "--bundle",
                str(bundle),
                "--parameters",
                str(selection),
                "--output",
                str(audit),
                "--episodes",
                str(args.audit_episodes),
                "--device",
                args.device,
            ],
        )
        selected = load_json(selection)
        audited = load_json(audit)
        if int(selected.get("schema_version", 0)) != 7:
            raise RuntimeError(
                "comparator selection is not two-variant Laplacian schema v7"
            )
        proto_values = selected["selected"]["protolp_paper"]["values"]
        if len(proto_values) != 2 or not all(
            math.isfinite(float(value)) for value in proto_values
        ):
            raise RuntimeError("invalid ProtoLP transferred parameters")
        if not bool(audited["passes_numerical_sanity_gate"]):
            raise RuntimeError("base-only transductive audit failed")

    status["state"] = "running_smoke"
    atomic_write_json(status_path, status)
    for dataset in datasets:
        feature = feature_root / DATASETS[dataset]["feature_cache"]
        protocol = protocol_root / (dataset + "_clip_vitb16_protocol.pt")
        selection = selection_root / (dataset + "_selection.json")
        audit = selection_root / (dataset + "_audit.json")
        smoke = smoke_root / (dataset + "_final1_full.json")
        execute(
            dataset,
            "smoke",
            [
                sys.executable,
                "-m",
                "fscil.run_transductive",
                "--feature-cache",
                str(feature),
                "--protocol",
                str(protocol),
                "--shrink-parameters",
                str(shrink_path(dataset)),
                "--radial-audit",
                str(shrink_audit_path(dataset)),
                "--transductive-parameters",
                str(selection),
                "--transductive-audit",
                str(audit),
                "--qpr-parameters",
                str(qpr_path(dataset, "selection")),
                "--qpr-audit",
                str(qpr_path(dataset, "audit")),
                "--output",
                str(smoke),
                "--device",
                args.device,
                "--seed-limit",
                "1",
                "--sessions",
                "final",
                "--modes",
                "full",
                "--methods",
                ",".join(METHODS),
            ],
        )
        payload = load_json(smoke)
        summary = payload["summary"]["full"]
        observed = {
            method: float(summary[method]["overall_mean"])
            for method in METHODS
        }
        anchors = LOCKED_FINAL_SMOKE[dataset]
        anchor_errors = {
            method: observed[method] - expected
            for method, expected in anchors.items()
        }
        proto_metadata = payload["resources"]["protolp_paper"]["last_metadata"]
        laplacian_metadata = payload["resources"]["laplacian_shot"][
            "last_metadata"
        ]
        core_metadata = payload["resources"]["laplacian_shot_core"][
            "last_metadata"
        ]
        map_metadata = payload["resources"]["ptmap_uniform"]["last_metadata"]
        checks = {
            "locked_anchor_reproduced": all(
                abs(error) <= float(args.anchor_tolerance)
                for error in anchor_errors.values()
            ),
            "protolp_finite": math.isfinite(observed["protolp_paper"]),
            "protolp_non_degenerate": (
                observed["protolp_paper"] >= float(args.minimum_protolp_accuracy)
            ),
            "protolp_support_clamped": (
                proto_metadata.get("support_assignment")
                == "one_hot_labels_per_Eq_4"
            ),
            "protolp_marginal_free": (
                proto_metadata.get("uses_query_marginal_constraint") is False
            ),
            "laplacian_full_components": (
                laplacian_metadata.get("includes_cl2n") is True
                and laplacian_metadata.get("binary_directed_euclidean_knn") is True
            ),
            "laplacian_rectification_matches_selector": (
                laplacian_metadata.get("includes_prototype_rectification")
                is bool(
                    int(
                        load_json(selection)["selected"]["laplacian"]["values"][2]
                    )
                )
            ),
            "laplacian_core_is_explicit": (
                core_metadata.get("includes_cl2n") is False
                and core_metadata.get("includes_prototype_rectification") is False
            ),
            "map_ot_core_is_honestly_labeled": (
                map_metadata.get("power_transform_applied") is False
                and map_metadata.get("power_transform_reason")
                == "signed_CLIP_embeddings"
            ),
        }
        status["smoke"][dataset] = {
            "accuracy": observed,
            "anchor_error": anchor_errors,
            "checks": checks,
            "selected_protolp_values": load_json(selection)["selected"][
                "protolp_paper"
            ]["values"],
        }
        atomic_write_json(status_path, status)
        if not all(checks.values()):
            status["state"] = "stopped_at_smoke_gate"
            status["failed_dataset"] = dataset
            atomic_write_json(status_path, status)
            raise SystemExit(4)

    status["state"] = "running_native_sanity"
    atomic_write_json(status_path, status)
    for dataset in datasets:
        feature = feature_root / DATASETS[dataset]["feature_cache"]
        protocol = protocol_root / (dataset + "_clip_vitb16_protocol.pt")
        selection = selection_root / (dataset + "_selection.json")
        output = native_root / (dataset + "_5way5shot15query.json")
        execute(
            dataset,
            "native_sanity",
            [
                sys.executable,
                "-m",
                "fscil.run_native_transductive_sanity",
                "--feature-cache",
                str(feature),
                "--protocol",
                str(protocol),
                "--transductive-parameters",
                str(selection),
                "--qpr-selection",
                str(qpr_path(dataset, "selection")),
                "--output",
                str(output),
                "--device",
                args.device,
                "--episodes",
                str(args.native_episodes),
            ],
        )
        native = load_json(output)
        native_checks = {
            "above_random": bool(native["passes_above_random_sanity_gate"]),
            "full_laplacian_exercised": (
                native["last_metadata"]["laplacian_shot"].get("includes_cl2n")
                is True
                and native["last_metadata"]["laplacian_shot"].get(
                    "includes_prototype_rectification"
                )
                is True
            ),
            "balanced_5way_5shot_15query": (
                native["episode_protocol"]["ways"] == 5
                and native["episode_protocol"]["shots"] == 5
                and native["episode_protocol"]["queries_per_class"] == 15
            ),
        }
        status["native_sanity"][dataset] = {
            "accuracy": native["accuracy"],
            "checks": native_checks,
            "result_sha256": native["result_sha256"],
        }
        atomic_write_json(status_path, status)
        if not all(native_checks.values()):
            status["state"] = "stopped_at_native_sanity_gate"
            status["failed_dataset"] = dataset
            atomic_write_json(status_path, status)
            raise SystemExit(5)

    status["state"] = "running_twenty_draw_benchmark"
    atomic_write_json(status_path, status)
    for dataset in datasets:
        feature = feature_root / DATASETS[dataset]["feature_cache"]
        protocol = protocol_root / (dataset + "_clip_vitb16_protocol.pt")
        selection = selection_root / (dataset + "_selection.json")
        audit = selection_root / (dataset + "_audit.json")
        output = benchmark_root / (dataset + "_all20_full.json")
        execute(
            dataset,
            "benchmark",
            [
                sys.executable,
                "-m",
                "fscil.run_transductive",
                "--feature-cache",
                str(feature),
                "--protocol",
                str(protocol),
                "--shrink-parameters",
                str(shrink_path(dataset)),
                "--radial-audit",
                str(shrink_audit_path(dataset)),
                "--transductive-parameters",
                str(selection),
                "--transductive-audit",
                str(audit),
                "--qpr-parameters",
                str(qpr_path(dataset, "selection")),
                "--qpr-audit",
                str(qpr_path(dataset, "audit")),
                "--output",
                str(output),
                "--device",
                args.device,
                "--seed-limit",
                "20",
                "--sessions",
                "all",
                "--modes",
                "full",
                "--methods",
                ",".join(METHODS),
            ],
        )

    status["state"] = "complete"
    status["finished_unix"] = time.time()
    status["runtime_sec"] = status["finished_unix"] - status["started_unix"]
    atomic_write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
