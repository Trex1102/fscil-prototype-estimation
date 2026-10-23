#!/usr/bin/env python3
"""Base-only selection of covariance shrinkage and prototype radial treatment.

The submitted ShrinkSel selector always normalizes class means.  That excludes the
raw-mean shared-LDA baseline from its hypothesis class and its balanced selection
objective overweights pseudo-novel classes relative to the FSCIL session-average
metric.  This development module tests a pre-target correction of both mismatches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    DATASETS,
    LAMBDA_GRID,
    PSEUDO_EPISODES,
    PSEUDO_NOVEL_PER_EPISODE,
    PSEUDO_QUERY_PER_CLASS,
    RIDGE_GAMMA,
)
from .geometry import (
    covariance_eigensystem,
    metric_scores_rotated,
    rotate_to_covariance_basis,
)
from .selection import (
    AuditData,
    ConstructionSelectionData,
    _anchor_split,
    load_audit,
    load_construction_selection,
)
from .utils import atomic_write_json, stable_json_sha256


PROTOTYPE_MODES = ("normalized", "raw", "resultant_debiased")


def deployed_radial_choice(
    parameters: Mapping[str, object], audit: Mapping[str, object]
) -> Tuple[str, float]:
    """Resolve the audited target-facing head, including the exact LDA fallback."""
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("radial selection/audit provenance mismatch")
    passed = bool(audit["passes_non_regression_gate"])
    expected_mode = str(parameters["selected_prototype_mode"]) if passed else "raw"
    expected_lambda = float(parameters["selected_lambda"]) if passed else 0.0
    deployed_mode = str(audit.get("deployed_prototype_mode", expected_mode))
    deployed_lambda = float(audit.get("deployed_lambda", expected_lambda))
    if deployed_mode != expected_mode or deployed_lambda != expected_lambda:
        raise RuntimeError("radial audit deployment record is inconsistent with its gate")
    return deployed_mode, deployed_lambda


def session_average_group_weights(dataset: str) -> Tuple[float, float]:
    """Return base/novel weights implied by the public FSCIL class schedule."""
    if dataset not in DATASETS:
        raise ValueError("unknown dataset: {}".format(dataset))
    spec = DATASETS[dataset]
    num_base = int(spec["num_base"])
    num_total = int(spec["num_total"])
    increment = int(spec["increment"])
    sessions = (num_total - num_base) // increment
    novel_weights = [
        float(session * increment) / float(num_base + session * increment)
        for session in range(sessions + 1)
    ]
    novel = float(mean(novel_weights))
    return 1.0 - novel, novel


def prototype_from_samples(values: torch.Tensor, mode: str) -> torch.Tensor:
    """Estimate a class center while optionally retaining radial information.

    ``resultant_debiased`` removes the finite-sample inflation in the squared
    resultant length, then combines that length with the sample-mean direction.
    The formula supports features that are only approximately unit normalized.
    """
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("prototype samples must be a non-empty matrix")
    if mode not in PROTOTYPE_MODES:
        raise ValueError("unknown prototype mode: {}".format(mode))
    samples = values.float()
    raw = samples.mean(0)
    if mode == "raw":
        return raw
    direction = F.normalize(raw[None, :], dim=1).squeeze(0)
    if mode == "normalized":
        return direction
    count = int(samples.shape[0])
    if count == 1:
        return direction * raw.norm()
    sample_second_moment = samples.square().sum(1).mean()
    population_resultant_square = (
        float(count) * raw.square().sum() - sample_second_moment
    ) / float(count - 1)
    radius = population_resultant_square.clamp_min(0.0).sqrt()
    return direction * radius


def _episode(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_queries: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episode_index: int,
    pseudo_novel_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, torch.Tensor]]:
    rng = np.random.RandomState(9137 + episode_index * 7919)
    anchor_ids = sorted(anchor_train)
    candidate_ids = sorted(pseudo_novel_pool)
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidate_ids),
            size=min(pseudo_novel_count, len(candidate_ids)),
            replace=False,
        ).tolist()
    )
    per_mode: Dict[str, List[torch.Tensor]] = {
        mode: [prototype_from_samples(anchor_train[class_id], mode) for class_id in anchor_ids]
        for mode in PROTOTYPE_MODES
    }
    queries: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    device = next(iter(anchor_train.values())).device
    for new_label, class_id in enumerate(anchor_ids):
        values = anchor_queries[class_id]
        offset = (episode_index * PSEUDO_QUERY_PER_CLASS) % values.shape[0]
        indices = (
            torch.arange(offset, offset + PSEUDO_QUERY_PER_CLASS, device=device)
            % values.shape[0]
        )
        queries.append(values[indices])
        labels.append(
            torch.full(
                (PSEUDO_QUERY_PER_CLASS,), new_label, dtype=torch.long, device=device
            )
        )
    num_anchor = len(anchor_ids)
    for position, class_id in enumerate(chosen):
        values = pseudo_novel_pool[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15485863 + episode_index * 32452843 + class_id * 49999)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        support = values[order[:5]]
        query = values[order[5 : 5 + PSEUDO_QUERY_PER_CLASS]]
        if query.shape[0] < PSEUDO_QUERY_PER_CLASS:
            raise RuntimeError(
                "insufficient pseudo-query samples for class {}".format(class_id)
            )
        for mode in PROTOTYPE_MODES:
            per_mode[mode].append(prototype_from_samples(support, mode))
        queries.append(query)
        labels.append(
            torch.full(
                (query.shape[0],),
                num_anchor + position,
                dtype=torch.long,
                device=device,
            )
        )
    return (
        torch.cat(queries, 0),
        torch.cat(labels, 0),
        num_anchor,
        {mode: torch.stack(values, 0) for mode, values in per_mode.items()},
    )


def _group_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    num_anchor: int,
    base_weight: float,
    novel_weight: float,
) -> Dict[str, float]:
    anchor_mask = labels < int(num_anchor)
    novel_mask = ~anchor_mask
    anchor = 100.0 * float((prediction[anchor_mask] == labels[anchor_mask]).float().mean())
    novel = 100.0 * float((prediction[novel_mask] == labels[novel_mask]).float().mean())
    return {
        "anchor": anchor,
        "pseudo_novel": novel,
        "balanced": 0.5 * (anchor + novel),
        "session_weighted": base_weight * anchor + novel_weight * novel,
    }


def evaluate_curve(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episodes: int,
    namespace_seed: int,
) -> Dict[str, object]:
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
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[dataset]
    for episode_index in range(episodes):
        queries, labels, num_anchor, prototypes = _episode(
            anchor_train,
            anchor_queries,
            pseudo_novel_pool,
            episode_index,
            pseudo_count,
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
        "covariance_dof": system.dof,
        "delta": float(system.delta.item()),
        "episodes": int(episodes),
        "pseudo_novel_per_episode": int(pseudo_count),
        "session_average_group_weights": {
            "base": base_weight,
            "novel": novel_weight,
        },
    }


def _pooled_summary(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    count = sum(int(row["episodes"]) for row in rows)
    if count < 1:
        raise ValueError("cannot pool an empty evaluation")
    pooled_mean = sum(
        int(row["episodes"]) * float(row["mean"]) for row in rows
    ) / float(count)
    second_moment = sum(
        int(row["episodes"])
        * (float(row["std"]) ** 2 + float(row["mean"]) ** 2)
        for row in rows
    ) / float(count)
    return {
        "mean": float(pooled_mean),
        "std": float(max(0.0, second_moment - pooled_mean**2) ** 0.5),
        "episodes": int(count),
    }


def evaluate_crossvalidated_curve(
    dataset: str,
    pools: Mapping[int, torch.Tensor],
    episodes: int,
    folds: int = 4,
) -> Dict[str, object]:
    """Select on repeated class-disjoint folds while reserving audit classes."""
    class_ids = np.asarray(sorted(pools), dtype=np.int64)
    if int(folds) < 2 or len(class_ids) < int(folds) * 2:
        raise ValueError("insufficient classes for cross-validated selection")
    rng = np.random.RandomState(8675309)
    shuffled = class_ids[rng.permutation(len(class_ids))]
    pseudo_folds = [values.tolist() for values in np.array_split(shuffled, int(folds))]
    quotient, remainder = divmod(int(episodes), int(folds))
    episode_counts = [quotient + (1 if index < remainder else 0) for index in range(folds)]
    if min(episode_counts) < 1:
        raise ValueError("episodes must be at least the number of folds")
    evaluations = []
    fold_manifest = []
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
        )
        evaluations.append(evaluated)
        fold_manifest.append(
            {
                "fold_index": fold_index,
                "anchor_classes": sorted(int(value) for value in anchors),
                "pseudo_novel_classes": sorted(int(value) for value in pseudo),
                "episodes": fold_episodes,
            }
        )
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
        "fold_manifest": fold_manifest,
        "fold_evaluations": evaluations,
        "episodes": int(episodes),
        "session_average_group_weights": {
            "base": base_weight,
            "novel": novel_weight,
        },
    }


def _candidate_score(
    curve: Mapping[str, object], mode: str, shrinkage: float, objective: str
) -> float:
    return float(curve[mode][str(float(shrinkage))][objective]["mean"])


def _best_candidate(
    curve: Mapping[str, object], objective: str
) -> Tuple[str, float]:
    # Prefer the lower-complexity classical raw endpoint on an exact score tie.
    tie_rank = {"raw": 2, "resultant_debiased": 1, "normalized": 0}
    return max(
        ((mode, float(value)) for mode in PROTOTYPE_MODES for value in LAMBDA_GRID),
        key=lambda candidate: (
            _candidate_score(curve, candidate[0], candidate[1], objective),
            tie_rank[candidate[0]],
            -candidate[1],
        ),
    )


def _best_crossvalidated_candidate(
    evaluated: Mapping[str, object], objective: str
) -> Tuple[str, float, Dict[str, object]]:
    folds = evaluated.get("fold_evaluations")
    if not isinstance(folds, list) or not folds:
        raise RuntimeError("robust selection requires class-disjoint fold evaluations")
    rows = []
    for mode in PROTOTYPE_MODES:
        for shrinkage in LAMBDA_GRID:
            gains = []
            for fold in folds:
                curve = fold["curve"]
                score = _candidate_score(curve, mode, float(shrinkage), objective)
                reference = _candidate_score(curve, "raw", 0.0, objective)
                gains.append(score - reference)
            rows.append(
                {
                    "prototype_mode": mode,
                    "lambda": float(shrinkage),
                    "fold_gains_vs_shared_lda": gains,
                    "mean_gain_vs_shared_lda": float(mean(gains)),
                    "worst_fold_gain_vs_shared_lda": float(min(gains)),
                }
            )
    # Raw means at lambda=0 have exactly zero gain and make this set non-empty.
    feasible = [row for row in rows if row["worst_fold_gain_vs_shared_lda"] >= 0.0]
    if not feasible:
        raise RuntimeError("shared-LDA endpoint is unexpectedly absent")
    tie_rank = {"raw": 2, "resultant_debiased": 1, "normalized": 0}
    selected = max(
        feasible,
        key=lambda row: (
            float(row["mean_gain_vs_shared_lda"]),
            float(row["worst_fold_gain_vs_shared_lda"]),
            tie_rank[str(row["prototype_mode"])],
            -float(row["lambda"]),
        ),
    )
    return (
        str(selected["prototype_mode"]),
        float(selected["lambda"]),
        {"selected": selected, "candidates": rows},
    )


def select_radial(data: ConstructionSelectionData, episodes: int) -> Dict[str, object]:
    selection_pool = dict(data.construction)
    selection_pool.update(data.selection)
    if len(selection_pool) != len(data.construction) + len(data.selection):
        raise RuntimeError("construction and selection classes overlap")
    evaluated = evaluate_crossvalidated_curve(
        data.dataset,
        selection_pool,
        episodes,
    )
    objective = "session_weighted"
    mode, shrinkage, robust_selection = _best_crossvalidated_candidate(
        evaluated, objective
    )
    result: Dict[str, object] = {
        "schema_version": 2,
        "stage": "base_selection_radial_shrinksel",
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_prototype_mode": mode,
        "selected_lambda": shrinkage,
        "ridge_gamma": RIDGE_GAMMA,
        "objective": objective,
        "objective_rationale": (
            "base/novel weights are derived only from the public FSCIL class schedule"
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


def audit_radial(
    data: AuditData, parameters: Mapping[str, object], episodes: int
) -> Dict[str, object]:
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("dataset mismatch between parameters and audit bundle")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("selection/audit bundle provenance mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("selection/audit feature provenance mismatch")
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
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "untouched_base_audit_radial_shrinksel",
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
        "passes_non_regression_gate": selected >= shared_lda - 0.10,
        "deployed_prototype_mode": (
            mode if selected >= shared_lda - 0.10 else "raw"
        ),
        "deployed_lambda": (
            shrinkage if selected >= shared_lda - 0.10 else 0.0
        ),
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
        result = select_radial(
            load_construction_selection(args.bundle, args.device), args.episodes
        )
    else:
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit_radial(load_audit(args.bundle, args.device), parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
