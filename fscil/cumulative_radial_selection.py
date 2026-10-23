#!/usr/bin/env python3
"""Cumulative-competition stress test for the radial ShrinkSel selector.

This module preserves the locked radial selector except for the number of
pseudo-novel classes in an episode.  Counts cycle through the public cumulative
FSCIL schedule and are clipped only when the class-disjoint base fold is too
small to realize a requested target-session count.
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

from .constants import DATASETS, LAMBDA_GRID, PSEUDO_EPISODES, RIDGE_GAMMA
from .geometry import (
    covariance_eigensystem,
    metric_scores_rotated,
    rotate_to_covariance_basis,
)
from .radial_selection import (
    PROTOTYPE_MODES,
    _best_candidate,
    _best_crossvalidated_candidate,
    _candidate_score,
    _episode,
    _group_metrics,
    _pooled_summary,
    session_average_group_weights,
)
from .selection import (
    AuditData,
    ConstructionSelectionData,
    _anchor_split,
    load_audit,
    load_construction_selection,
)
from .utils import atomic_write_json, stable_json_sha256


CALIBRATION_REGIME = "cumulative_competition_clipped_to_disjoint_base_pool"


def cumulative_requested_counts(dataset: str) -> Tuple[int, ...]:
    """Return the public cumulative novel-class counts for sessions 1..T."""
    if dataset not in DATASETS:
        raise ValueError("unknown dataset: {}".format(dataset))
    spec = DATASETS[dataset]
    num_base = int(spec["num_base"])
    num_total = int(spec["num_total"])
    increment = int(spec["increment"])
    sessions = (num_total - num_base) // increment
    if sessions < 1 or num_base + sessions * increment != num_total:
        raise ValueError("invalid public FSCIL schedule for {}".format(dataset))
    return tuple(increment * session for session in range(1, sessions + 1))


def _count_summary(requested: Sequence[int], realized: Sequence[int]) -> Dict[str, object]:
    if len(requested) != len(realized) or not requested:
        raise ValueError("requested/realized count records must be non-empty and paired")
    requested_histogram = Counter(int(value) for value in requested)
    realized_histogram = Counter(int(value) for value in realized)
    saturated = sum(int(left > right) for left, right in zip(requested, realized))
    return {
        "requested_histogram": {
            str(key): int(value) for key, value in sorted(requested_histogram.items())
        },
        "realized_histogram": {
            str(key): int(value) for key, value in sorted(realized_histogram.items())
        },
        "requested_min": int(min(requested)),
        "requested_max": int(max(requested)),
        "realized_min": int(min(realized)),
        "realized_max": int(max(realized)),
        "saturated_episodes": int(saturated),
        "saturation_fraction": float(saturated / len(requested)),
        "episodes": int(len(requested)),
    }


def evaluate_curve(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episodes: int,
    namespace_seed: int,
    schedule_offset: int = 0,
) -> Dict[str, object]:
    if not pseudo_novel_pool:
        raise ValueError("cumulative calibration requires pseudo-novel identities")
    anchor_train, anchor_queries = _anchor_split(anchor_pool, namespace_seed)
    system = covariance_eigensystem(dict(anchor_train), sorted(anchor_train))
    base_weight, novel_weight = session_average_group_weights(dataset)
    curves = {
        mode: {
            str(float(value)): {
                metric: []
                for metric in ("anchor", "pseudo_novel", "balanced", "session_weighted")
            }
            for value in LAMBDA_GRID
        }
        for mode in PROTOTYPE_MODES
    }
    schedule = cumulative_requested_counts(dataset)
    requested_counts: List[int] = []
    realized_counts: List[int] = []
    for episode_index in range(int(episodes)):
        requested = int(schedule[(int(schedule_offset) + episode_index) % len(schedule)])
        realized = min(requested, len(pseudo_novel_pool))
        requested_counts.append(requested)
        realized_counts.append(realized)
        queries, labels, num_anchor, prototypes = _episode(
            anchor_train,
            anchor_queries,
            pseudo_novel_pool,
            episode_index,
            realized,
        )
        rotated_queries = rotate_to_covariance_basis(queries, system)
        rotated = {
            mode: rotate_to_covariance_basis(values, system)
            for mode, values in prototypes.items()
        }
        for mode in PROTOTYPE_MODES:
            for shrinkage in LAMBDA_GRID:
                prediction = metric_scores_rotated(
                    rotated_queries,
                    rotated[mode],
                    system,
                    float(shrinkage),
                    RIDGE_GAMMA,
                ).argmax(1)
                metrics = _group_metrics(
                    prediction, labels, num_anchor, base_weight, novel_weight
                )
                for metric, value in metrics.items():
                    curves[mode][str(float(shrinkage))][metric].append(value)
    summary = {
        mode: {
            shrinkage: {
                metric: {
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)),
                    "episodes": len(values),
                }
                for metric, values in metrics.items()
            }
            for shrinkage, metrics in mode_curve.items()
        }
        for mode, mode_curve in curves.items()
    }
    return {
        "curve": summary,
        "covariance_dof": int(system.dof),
        "delta": float(system.delta.item()),
        "episodes": int(episodes),
        "calibration_regime": CALIBRATION_REGIME,
        "public_cumulative_requested_counts": list(schedule),
        "available_pseudo_identities": int(len(pseudo_novel_pool)),
        "count_coverage": _count_summary(requested_counts, realized_counts),
        "session_average_group_weights": {
            "base": base_weight,
            "novel": novel_weight,
        },
    }


def evaluate_crossvalidated_curve(
    dataset: str,
    pools: Mapping[int, torch.Tensor],
    episodes: int,
    folds: int = 4,
) -> Dict[str, object]:
    class_ids = np.asarray(sorted(pools), dtype=np.int64)
    if int(folds) < 2 or len(class_ids) < int(folds) * 2:
        raise ValueError("insufficient classes for cross-validated selection")
    rng = np.random.RandomState(8675309)
    shuffled = class_ids[rng.permutation(len(class_ids))]
    pseudo_folds = [values.tolist() for values in np.array_split(shuffled, int(folds))]
    quotient, remainder = divmod(int(episodes), int(folds))
    episode_counts = [
        quotient + (1 if fold_index < remainder else 0)
        for fold_index in range(int(folds))
    ]
    if min(episode_counts) < 1:
        raise ValueError("episodes must be at least the number of folds")
    evaluations = []
    fold_manifest = []
    schedule_offset = 0
    all_requested: List[int] = []
    all_realized: List[int] = []
    for fold_index, (pseudo_ids, fold_episodes) in enumerate(
        zip(pseudo_folds, episode_counts)
    ):
        pseudo_set = {int(value) for value in pseudo_ids}
        anchors = {
            class_id: values
            for class_id, values in pools.items()
            if int(class_id) not in pseudo_set
        }
        pseudo = {
            class_id: values
            for class_id, values in pools.items()
            if int(class_id) in pseudo_set
        }
        evaluated = evaluate_curve(
            dataset,
            anchors,
            pseudo,
            fold_episodes,
            namespace_seed=271828 + fold_index * 100003,
            schedule_offset=schedule_offset,
        )
        evaluations.append(evaluated)
        coverage = evaluated["count_coverage"]
        schedule = cumulative_requested_counts(dataset)
        for local_episode in range(int(fold_episodes)):
            requested = int(
                schedule[(schedule_offset + local_episode) % len(schedule)]
            )
            all_requested.append(requested)
            all_realized.append(min(requested, len(pseudo)))
        fold_manifest.append(
            {
                "fold_index": int(fold_index),
                "anchor_classes": sorted(int(value) for value in anchors),
                "pseudo_novel_classes": sorted(int(value) for value in pseudo),
                "episodes": int(fold_episodes),
                "schedule_offset": int(schedule_offset),
                "count_coverage": coverage,
            }
        )
        schedule_offset += int(fold_episodes)
    curve = {
        mode: {
            str(float(shrinkage)): {
                metric: _pooled_summary(
                    [
                        evaluation["curve"][mode][str(float(shrinkage))][metric]
                        for evaluation in evaluations
                    ]
                )
                for metric in (
                    "anchor",
                    "pseudo_novel",
                    "balanced",
                    "session_weighted",
                )
            }
            for shrinkage in LAMBDA_GRID
        }
        for mode in PROTOTYPE_MODES
    }
    base_weight, novel_weight = session_average_group_weights(dataset)
    return {
        "curve": curve,
        "selection_scheme": "four_fold_class_disjoint_cross_validation",
        "calibration_regime": CALIBRATION_REGIME,
        "public_cumulative_requested_counts": list(cumulative_requested_counts(dataset)),
        "count_coverage": _count_summary(all_requested, all_realized),
        "fold_manifest": fold_manifest,
        "fold_evaluations": evaluations,
        "episodes": int(episodes),
        "session_average_group_weights": {
            "base": base_weight,
            "novel": novel_weight,
        },
    }


def select_cumulative(data: ConstructionSelectionData, episodes: int) -> Dict[str, object]:
    selection_pool = dict(data.construction)
    selection_pool.update(data.selection)
    if len(selection_pool) != len(data.construction) + len(data.selection):
        raise RuntimeError("construction and selection classes overlap")
    evaluated = evaluate_crossvalidated_curve(data.dataset, selection_pool, episodes)
    objective = "session_weighted"
    mode, shrinkage, robust_selection = _best_crossvalidated_candidate(
        evaluated, objective
    )
    result: Dict[str, object] = {
        "schema_version": 3,
        "stage": "base_selection_radial_shrinksel",
        "calibration_regime": CALIBRATION_REGIME,
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_prototype_mode": mode,
        "selected_lambda": shrinkage,
        "ridge_gamma": RIDGE_GAMMA,
        "objective": objective,
        "objective_rationale": (
            "group weights and cumulative requested counts derive only from the "
            "public FSCIL class schedule"
        ),
        "selection_scheme": evaluated["selection_scheme"],
        "robust_selection_rule": (
            "maximize mean gain over shared LDA subject to nonnegative gain on "
            "every class-disjoint selection fold"
        ),
        "robust_selection": robust_selection,
        "prototype_modes": list(PROTOTYPE_MODES),
        "candidate_lambda_grid": list(LAMBDA_GRID),
        "evaluation": evaluated,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_cumulative(
    data: AuditData, parameters: Mapping[str, object], episodes: int
) -> Dict[str, object]:
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("dataset mismatch between parameters and audit bundle")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("selection/audit bundle provenance mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("selection/audit feature provenance mismatch")
    if parameters.get("calibration_regime") != CALIBRATION_REGIME:
        raise RuntimeError("parameters are not from cumulative calibration")
    evaluated = evaluate_curve(
        data.dataset,
        data.anchors,
        data.audit,
        episodes,
        namespace_seed=314159,
    )
    objective = str(parameters["objective"])
    curve = evaluated["curve"]
    mode = str(parameters["selected_prototype_mode"])
    shrinkage = float(parameters["selected_lambda"])
    selected = _candidate_score(curve, mode, shrinkage, objective)
    shared_lda = _candidate_score(curve, "raw", 0.0, objective)
    old_family_best = max(
        _candidate_score(curve, "normalized", value, objective)
        for value in LAMBDA_GRID
    )
    oracle_mode, oracle_lambda = _best_candidate(curve, objective)
    oracle = _candidate_score(curve, oracle_mode, oracle_lambda, objective)
    passes = bool(np.isfinite(selected) and selected >= shared_lda - 0.10)
    result: Dict[str, object] = {
        "schema_version": 2,
        "stage": "untouched_base_audit_radial_shrinksel",
        "calibration_regime": CALIBRATION_REGIME,
        "dataset": data.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_prototype_mode": mode,
        "selected_lambda": shrinkage,
        "objective": objective,
        "selected_score": selected,
        "shared_lda_score": shared_lda,
        "selected_minus_shared_lda": selected - shared_lda,
        "audit_best_normalized_score": old_family_best,
        "selected_minus_audit_best_normalized": selected - old_family_best,
        "audit_oracle_prototype_mode": oracle_mode,
        "audit_oracle_lambda": oracle_lambda,
        "audit_oracle_score": oracle,
        "selection_regret_to_audit_oracle": oracle - selected,
        "passes_non_regression_gate": passes,
        "deployed_prototype_mode": mode if passes else "raw",
        "deployed_lambda": shrinkage if passes else 0.0,
        "gate_reference": "raw means with lambda=0 (shared-covariance LDA)",
        "evaluation": evaluated,
    }
    result["audit_result_sha256"] = stable_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--bundle", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--episodes", type=int, default=PSEUDO_EPISODES)
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        result = select_cumulative(
            load_construction_selection(args.bundle, args.device), args.episodes
        )
    else:
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit_cumulative(
            load_audit(args.bundle, args.device), parameters, args.episodes
        )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
