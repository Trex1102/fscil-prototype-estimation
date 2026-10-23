#!/usr/bin/env python3
"""Repeat corrected-QPR selection across deterministic base-class partitions.

This is a target-free stability diagnostic.  It changes only which base
identities play construction, selection, and audit roles; the parent ShrinkSel
choice, feature cache, QPR grid, objectives, and episode generators stay fixed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .qpr_calibration import audit_clean, evaluate, select_clean
from .qpr_selection import QPR_GRID, load_shrink_parameters, parameter_key
from .selection import AuditData, ConstructionSelectionData
from .transductive_selection import _from_audit, _from_selection
from .utils import atomic_write_json, stable_json_sha256


PARTITION_SEEDS = (314159, 271828, 161803, 141421, 173205)


def _features_by_class(
    features: torch.Tensor, labels: torch.Tensor
) -> Dict[int, torch.Tensor]:
    return {
        int(class_id): features[labels == int(class_id)]
        for class_id in sorted(int(value) for value in labels.unique().tolist())
    }


def _partition(
    class_ids: Sequence[int], counts: Tuple[int, int, int], seed: int
) -> Mapping[str, List[int]]:
    order = [
        int(value)
        for value in np.random.RandomState(int(seed)).permutation(class_ids).tolist()
    ]
    construction_count, selection_count, audit_count = counts
    if construction_count + selection_count + audit_count != len(order):
        raise RuntimeError("partition counts do not cover all base identities")
    first = construction_count
    second = first + selection_count
    return {
        "construction": sorted(order[:first]),
        "selection": sorted(order[first:second]),
        "audit": sorted(order[second:]),
    }


def _audit_regret(
    evaluation: Mapping[str, object], selected_key: str
) -> Mapping[str, object]:
    curve = evaluation["combined"]
    baseline = curve["shrinksel"]
    feasible = []
    for values in QPR_GRID:
        key = parameter_key(values)
        row = curve[key]
        if float(row["anchor"]["mean"]) >= float(baseline["anchor"]["mean"]) - 0.5:
            feasible.append((values, key, row))
    if not feasible:
        raise RuntimeError("audit grid contains no anchor-safe QPR candidate")
    oracle_values, oracle_key, oracle_row = max(
        feasible,
        key=lambda item: (
            float(item[2]["class_balanced"]["mean"]),
            float(item[2]["pseudo_novel"]["mean"]),
            float(item[2]["gate_assignment_purity"]["mean"]),
            float(item[0][1]),
            float(item[0][2]),
            -float(item[0][0]),
            -float(item[0][3]),
        ),
    )
    selected_score = float(curve[selected_key]["class_balanced"]["mean"])
    oracle_score = float(oracle_row["class_balanced"]["mean"])
    return {
        "selected_key": selected_key,
        "selected_class_balanced": selected_score,
        "audit_oracle_key": oracle_key,
        "audit_oracle_values": [
            float(oracle_values[0]),
            float(oracle_values[1]),
            int(oracle_values[2]),
            float(oracle_values[3]),
        ],
        "audit_oracle_class_balanced": oracle_score,
        "regret": max(0.0, oracle_score - selected_score),
        "anchor_safety_constraint_pp": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regime", default="one_increment", choices=("one_increment",))
    args = parser.parse_args()

    if int(args.episodes) != 20:
        raise ValueError("the locked stability diagnostic requires 20 episodes")
    payload = torch.load(str(args.bundle), map_location="cpu")
    forbidden = {key for key in payload if "test" in key.lower() or "novel" in key.lower()}
    if forbidden:
        raise RuntimeError(
            "selector-stability bundle exposes forbidden fields: {}".format(
                sorted(forbidden)
            )
        )
    features = payload["base_features"].float().to(args.device)
    labels = payload["base_labels"].long().to(args.device)
    by_class = _features_by_class(features, labels)
    class_ids = sorted(by_class)
    original = payload["base_partition"]
    counts = (
        len(original["construction"]),
        len(original["selection"]),
        len(original["audit"]),
    )
    source_sha256 = str(payload["source_feature_file_sha256"])
    dataset = str(payload["dataset"])
    repeats: List[Dict[str, object]] = []

    for repeat_index, seed in enumerate(PARTITION_SEEDS):
        partition = _partition(class_ids, counts, seed)
        bundle_sha256 = stable_json_sha256(
            {
                "stage": "qpr_selector_stability_partition",
                "dataset": dataset,
                "source_feature_file_sha256": source_sha256,
                "seed": int(seed),
                "partition": partition,
            }
        )
        selection_data = ConstructionSelectionData(
            dataset=dataset,
            construction={int(key): by_class[int(key)] for key in partition["construction"]},
            selection={int(key): by_class[int(key)] for key in partition["selection"]},
            bundle_sha256=bundle_sha256,
            source_feature_file_sha256=source_sha256,
        )
        audit_data = AuditData(
            dataset=dataset,
            anchors={
                int(key): by_class[int(key)]
                for key in partition["construction"] + partition["selection"]
            },
            audit={int(key): by_class[int(key)] for key in partition["audit"]},
            bundle_sha256=bundle_sha256,
            source_feature_file_sha256=source_sha256,
        )
        selection_pools = _from_selection(selection_data)
        audit_pools = _from_audit(audit_data)
        shrink = load_shrink_parameters(args.shrink_parameters, selection_pools)
        selected = select_clean(
            selection_pools, shrink, int(args.episodes), str(args.regime)
        )
        audit = audit_clean(audit_pools, shrink, selected, int(args.episodes))
        audit_grid = evaluate(
            audit_pools,
            float(shrink["selected_lambda"]),
            str(shrink.get("selected_prototype_mode", "normalized")),
            int(args.episodes),
            namespace_seed=141421,
            candidates=QPR_GRID,
            regime=str(args.regime),
        )
        regret = _audit_regret(audit_grid, str(selected["selected"]["key"]))
        repeats.append(
            {
                "repeat_index": int(repeat_index),
                "partition_seed": int(seed),
                "partition": partition,
                "partition_sha256": bundle_sha256,
                "selected": selected["selected"],
                "selection_result_sha256": selected["selection_result_sha256"],
                "audit": {
                    "passes_transfer_safety_gate": bool(
                        audit["passes_transfer_safety_gate"]
                    ),
                    "selected_class_balanced_gain": float(
                        audit["selected_class_balanced_gain"]
                    ),
                    "selected_anchor_gain": float(audit["selected_anchor_gain"]),
                    "deployed": audit["deployed"],
                    "audit_result_sha256": audit["audit_result_sha256"],
                },
                "target_free_audit_regret": regret,
                "selection_evaluation": selected["evaluation"],
                "audit_grid_evaluation": audit_grid,
            }
        )

    selected_keys = [str(row["selected"]["key"]) for row in repeats]
    regrets = [float(row["target_free_audit_regret"]["regret"]) for row in repeats]
    pass_count = sum(
        bool(row["audit"]["passes_transfer_safety_gate"]) for row in repeats
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "corrected_qpr_target_free_selector_stability",
        "dataset": dataset,
        "source_feature_file_sha256": source_sha256,
        "source_bundle_sha256": str(payload["selection_bundle_sha256"]),
        "episodes_per_condition": int(args.episodes),
        "calibration_regime": str(args.regime),
        "partition_seeds": [int(value) for value in PARTITION_SEEDS],
        "partition_counts": {
            "construction": int(counts[0]),
            "selection": int(counts[1]),
            "audit": int(counts[2]),
        },
        "selection_frequency": dict(sorted(Counter(selected_keys).items())),
        "unique_selected_tuple_count": len(set(selected_keys)),
        "audit_pass_count": int(pass_count),
        "audit_pass_rate": float(pass_count / len(repeats)),
        "target_free_audit_regret": {
            "mean": float(mean(regrets)),
            "std": float(pstdev(regrets)),
            "minimum": float(min(regrets)),
            "maximum": float(max(regrets)),
        },
        "repeats": repeats,
        "target_data_used": False,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
