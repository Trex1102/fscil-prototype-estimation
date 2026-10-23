#!/usr/bin/env python3
"""Lock one radial-ShrinkSel configuration across datasets before target use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Mapping, Sequence, Tuple

from .constants import LAMBDA_GRID
from .radial_selection import PROTOTYPE_MODES
from .utils import atomic_write_json, stable_json_sha256


CORE_DATASETS = ("cub200", "cifar100", "miniimagenet")


def _load_unique(paths: Sequence[Path], expected_stage: str) -> Dict[str, Mapping[str, object]]:
    result = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["stage"] != expected_stage:
            raise RuntimeError("unexpected stage in {}".format(path))
        dataset = str(payload["dataset"])
        if dataset in result:
            raise RuntimeError("duplicate dataset: {}".format(dataset))
        result[dataset] = payload
    if set(result) != set(CORE_DATASETS):
        raise RuntimeError("global selection requires exactly {}".format(CORE_DATASETS))
    return result


def _gain(payload: Mapping[str, object], mode: str, shrinkage: float) -> float:
    curve = payload["evaluation"]["curve"]
    objective = str(payload["objective"])
    score = float(curve[mode][str(float(shrinkage))][objective]["mean"])
    reference = float(curve["raw"]["0.0"][objective]["mean"])
    return score - reference


def lock_global(
    selections: Mapping[str, Mapping[str, object]],
    audits: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    for dataset in CORE_DATASETS:
        if (
            audits[dataset]["selection_result_sha256"]
            != selections[dataset]["selection_result_sha256"]
        ):
            raise RuntimeError("selection/audit provenance mismatch for {}".format(dataset))
    candidates = []
    for mode in PROTOTYPE_MODES:
        for shrinkage in LAMBDA_GRID:
            gains = {
                dataset: _gain(selections[dataset], mode, float(shrinkage))
                for dataset in CORE_DATASETS
            }
            candidates.append(
                {
                    "prototype_mode": mode,
                    "lambda": float(shrinkage),
                    "selection_gain_vs_shared_lda": gains,
                    "selection_average_gain": float(mean(gains.values())),
                    "selection_worst_gain": float(min(gains.values())),
                }
            )
    feasible = [
        row for row in candidates if float(row["selection_worst_gain"]) >= 0.0
    ]
    if not feasible:
        raise RuntimeError("no candidate is non-regressive on every selection dataset")
    mode_rank = {"raw": 2, "resultant_debiased": 1, "normalized": 0}
    selected = max(
        feasible,
        key=lambda row: (
            float(row["selection_average_gain"]),
            float(row["selection_worst_gain"]),
            mode_rank[str(row["prototype_mode"])],
            -float(row["lambda"]),
        ),
    )
    audit_gains = {
        dataset: _gain(
            audits[dataset],
            str(selected["prototype_mode"]),
            float(selected["lambda"]),
        )
        for dataset in CORE_DATASETS
    }
    passes_audit = all(value >= -0.10 for value in audit_gains.values())
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "global_radial_configuration_lock_before_target_evaluation",
        "datasets": list(CORE_DATASETS),
        "selected_prototype_mode": selected["prototype_mode"],
        "selected_lambda": selected["lambda"],
        "ridge_gamma": float(selections[CORE_DATASETS[0]]["ridge_gamma"]),
        "selection_rule": (
            "maximize mean gain over shared LDA subject to nonnegative gain on "
            "every dataset's class-disjoint cross-validation folds"
        ),
        "selected_selection_gain_vs_shared_lda": selected[
            "selection_gain_vs_shared_lda"
        ],
        "selected_selection_average_gain": selected["selection_average_gain"],
        "audit_gain_vs_shared_lda": audit_gains,
        "passes_untouched_base_audit": passes_audit,
        "deployed_prototype_mode": (
            selected["prototype_mode"] if passes_audit else "raw"
        ),
        "deployed_lambda": selected["lambda"] if passes_audit else 0.0,
        "audit_gate": "no dataset may trail shared LDA by more than 0.10 point",
        "source_feature_file_sha256": {
            dataset: selections[dataset]["source_feature_file_sha256"]
            for dataset in CORE_DATASETS
        },
        "selection_result_sha256": {
            dataset: selections[dataset]["selection_result_sha256"]
            for dataset in CORE_DATASETS
        },
        "audit_result_sha256": {
            dataset: audits[dataset]["audit_result_sha256"]
            for dataset in CORE_DATASETS
        },
        "candidate_summary": candidates,
    }
    result["global_selection_sha256"] = stable_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", nargs="+", required=True, type=Path)
    parser.add_argument("--audit", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selections = _load_unique(args.selection, "base_selection_radial_shrinksel")
    audits = _load_unique(args.audit, "untouched_base_audit_radial_shrinksel")
    result = lock_global(selections, audits)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
