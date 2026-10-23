#!/usr/bin/env python3
"""Select and audit this work's QPR on class-disjoint base episodes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import PROTOTYPE_MODES, prototype_from_samples
from .selection import _anchor_split, load_audit, load_construction_selection
from .transductive_methods import (
    query_prototype_rectification,
)
from .transductive_selection import (
    BaseTransductivePools,
    CalibrationEpisode,
    _episode,
    _from_audit,
    _from_selection,
    _metrics,
)
from .constants import RIDGE_GAMMA
from .utils import atomic_write_json, stable_json_sha256


# The authors' pre-commit internal defaults are retained as a fixed comparator.
# They are not defaults from published prior work. The selected branch uses a
# deliberately small, committed grid and includes
# alpha=0 so base episodes can reject rectification rather than force a gain.
ALPHA_GRID = (0.0, 0.5, 0.75, 1.0)
MASS_THRESHOLD_GRID = (0.3, 0.6, 0.9)
MIN_COUNT_GRID = (5, 10)
LOGIT_SCALE_GRID = (0.03, 0.1, 0.3, 1.0)
QPR_GRID = tuple(
    (alpha, threshold, min_count, scale)
    for alpha in ALPHA_GRID
    for threshold in MASS_THRESHOLD_GRID
    for min_count in MIN_COUNT_GRID
    for scale in LOGIT_SCALE_GRID
)
FIXED_QPR = (0.75, 0.3, 5, 1.0)


def deployed_qpr_choice(
    parameters: Mapping[str, object], audit_result: Mapping[str, object]
) -> Dict[str, object]:
    """Resolve the audited tuple, deriving alpha=0 for legacy failed audits."""
    if audit_result["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("QPR selection/audit provenance mismatch")
    selected = parameters["selected"]
    if not isinstance(selected, Mapping):
        raise RuntimeError("invalid QPR selection record")
    expected = {
        "alpha": (
            float(selected["alpha"])
            if bool(audit_result["passes_transfer_safety_gate"])
            else 0.0
        ),
        "mass_threshold": float(selected["mass_threshold"]),
        "min_count": int(selected["min_count"]),
        "logit_scale": float(selected["logit_scale"]),
    }
    recorded = audit_result.get("deployed", expected)
    if not isinstance(recorded, Mapping):
        raise RuntimeError("invalid QPR deployment record")
    normalized = {
        "alpha": float(recorded["alpha"]),
        "mass_threshold": float(recorded["mass_threshold"]),
        "min_count": int(recorded["min_count"]),
        "logit_scale": float(recorded["logit_scale"]),
    }
    if normalized != expected:
        raise RuntimeError("QPR audit deployment record is inconsistent with its gate")
    return normalized


def estimation_aware_prototypes(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    class_count: int,
    prototype_mode: str,
) -> torch.Tensor:
    if prototype_mode not in PROTOTYPE_MODES:
        raise ValueError("unknown prototype mode: {}".format(prototype_mode))
    return torch.stack(
        [
            prototype_from_samples(
                support[support_labels == class_id], prototype_mode
            )
            for class_id in range(int(class_count))
        ],
        0,
    )


def parse_values(key: str) -> Tuple[float, float, int, float]:
    values = key.split(":", 1)[1].split(",")
    return float(values[0]), float(values[1]), int(float(values[2])), float(values[3])


def parameter_key(values: Sequence[float]) -> str:
    return "qpr:" + ",".join(str(float(value)) for value in values)


def gate_diagnostics(
    baseline_scores: torch.Tensor,
    episode: CalibrationEpisode,
    mass_threshold: float,
    logit_scale: float,
) -> Dict[str, float]:
    probability = torch.softmax(
        baseline_scores.float() * float(logit_scale), dim=1
    )
    predicted = probability.argmax(1)
    novel_probability = probability[:, episode.num_anchor :]
    novel_mass = novel_probability.sum(1)
    assigned = novel_probability.argmax(1) + episode.num_anchor
    gate = (predicted >= episode.num_anchor) & (
        novel_mass >= float(mass_threshold)
    )
    count = int(gate.sum().item())
    if count:
        actual_novel = episode.cohort_labels[gate] >= episode.num_anchor
        group_purity = 100.0 * float(actual_novel.float().mean().item())
        assignment_purity = 100.0 * float(
            (assigned[gate] == episode.cohort_labels[gate]).float().mean().item()
        )
    else:
        group_purity = 100.0
        assignment_purity = 100.0
    return {
        "gated_query_count": float(count),
        "gate_group_purity": group_purity,
        "gate_assignment_purity": assignment_purity,
    }


def summarize(rows: Mapping[str, List[Dict[str, float]]]) -> Dict[str, object]:
    fields = (
        "overall",
        "class_balanced",
        "anchor",
        "pseudo_novel",
        "gated_query_count",
        "gate_group_purity",
        "gate_assignment_purity",
        "eligible_class_count",
        "updated_class_count",
    )
    return {
        key: {
            field: {
                "mean": float(mean(row[field] for row in values)),
                "std": float(pstdev(row[field] for row in values)),
                "episodes": len(values),
            }
            for field in fields
        }
        for key, values in rows.items()
    }


def evaluate(
    pools: BaseTransductivePools,
    selected_lambda: float,
    prototype_mode: str,
    episodes: int,
    namespace_seed: int,
    candidates: Sequence[Tuple[float, float, int, float]],
) -> Dict[str, object]:
    construction, construction_cohort = _anchor_split(
        pools.construction, namespace_seed
    )
    heldout: Dict[int, torch.Tensor] = dict(construction_cohort)
    heldout.update({int(key): value for key, value in pools.candidate.items()})
    covariance = covariance_eigensystem(dict(construction), sorted(construction))
    keys = [parameter_key(values) for values in candidates]
    rows: Dict[str, List[Dict[str, float]]] = {"shrinksel": []}
    rows.update({key: [] for key in keys})
    condition_rows = {
        condition: {key: [] for key in rows}
        for condition in ("balanced", "imbalanced")
    }

    for episode_index in range(int(episodes)):
        for condition in ("balanced", "imbalanced"):
            episode = _episode(
                pools,
                construction,
                heldout,
                episode_index,
                imbalanced=condition == "imbalanced",
            )
            class_count = int(episode.support_labels.max().item()) + 1
            prototypes = estimation_aware_prototypes(
                episode.support,
                episode.support_labels,
                class_count,
                prototype_mode,
            )
            baseline_scores = metric_scores(
                episode.cohort,
                prototypes,
                covariance,
                float(selected_lambda),
                RIDGE_GAMMA,
            )
            baseline = _metrics(baseline_scores.argmax(1), episode)
            baseline.update(
                {
                    "gated_query_count": 0.0,
                    "gate_group_purity": 100.0,
                    "gate_assignment_purity": 100.0,
                    "eligible_class_count": 0.0,
                    "updated_class_count": 0.0,
                }
            )
            rows["shrinksel"].append(baseline)
            condition_rows[condition]["shrinksel"].append(baseline)

            for values, key in zip(candidates, keys):
                alpha, threshold, min_count, scale = values
                output = query_prototype_rectification(
                    episode.cohort,
                    prototypes,
                    baseline_scores,
                    episode.num_anchor,
                    alpha=float(alpha),
                    mass_threshold=float(threshold),
                    min_count=int(min_count),
                    logit_scale=float(scale),
                )
                prediction = metric_scores(
                    episode.cohort,
                    output.prototypes,
                    covariance,
                    float(selected_lambda),
                    RIDGE_GAMMA,
                ).argmax(1)
                result = _metrics(prediction, episode)
                result.update(
                    gate_diagnostics(
                        baseline_scores,
                        episode,
                        mass_threshold=float(threshold),
                        logit_scale=float(scale),
                    )
                )
                result["eligible_class_count"] = float(
                    output.metadata["eligible_class_count"]
                )
                result["updated_class_count"] = float(
                    output.metadata["updated_class_count"]
                )
                rows[key].append(result)
                condition_rows[condition][key].append(result)

    return {
        "combined": summarize(rows),
        "conditions": {
            condition: summarize(values)
            for condition, values in condition_rows.items()
        },
        "episodes_per_condition": int(episodes),
        "calibration_conditions": ["balanced", "deterministic_imbalanced"],
        "covariance_dof": int(covariance.dof),
        "selected_lambda": float(selected_lambda),
        "selected_prototype_mode": prototype_mode,
    }


def load_shrink_parameters(
    path: Path, pools: BaseTransductivePools
) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["dataset"] != pools.dataset:
        raise RuntimeError("shrinkage/QPR dataset mismatch")
    if payload["source_feature_file_sha256"] != pools.source_feature_file_sha256:
        raise RuntimeError("shrinkage/QPR feature mismatch")
    return payload


def select(
    pools: BaseTransductivePools,
    shrink: Mapping[str, object],
    episodes: int,
) -> Dict[str, object]:
    selected_lambda = float(shrink["selected_lambda"])
    prototype_mode = str(shrink.get("selected_prototype_mode", "normalized"))
    evaluation = evaluate(
        pools,
        selected_lambda,
        prototype_mode,
        episodes,
        namespace_seed=161803,
        candidates=QPR_GRID,
    )
    curve = evaluation["combined"]
    baseline = curve["shrinksel"]
    candidates = []
    for values in QPR_GRID:
        key = parameter_key(values)
        row = curve[key]
        anchor_safe = float(row["anchor"]["mean"]) >= (
            float(baseline["anchor"]["mean"]) - 0.5
        )
        if anchor_safe:
            candidates.append((values, key, row))
    if not candidates:
        raise RuntimeError("QPR grid contains no anchor-safe base candidate")
    selected_values, selected_key, selected_row = max(
        candidates,
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
    result: Dict[str, object] = {
        "schema_version": 2,
        "stage": "qpr_base_only_selection",
        "dataset": pools.dataset,
        "selection_bundle_sha256": pools.bundle_sha256,
        "source_feature_file_sha256": pools.source_feature_file_sha256,
        "shrink_selection_result_sha256": shrink["selection_result_sha256"],
        "selected_lambda": selected_lambda,
        "selected_prototype_mode": prototype_mode,
        "objective": (
            "class-balanced accuracy over balanced and deterministic-imbalanced "
            "base pseudo-novel episodes with anchor loss <=0.5 point"
        ),
        "grid": {
            "alpha": list(ALPHA_GRID),
            "mass_threshold": list(MASS_THRESHOLD_GRID),
            "min_count": list(MIN_COUNT_GRID),
            "logit_scale": list(LOGIT_SCALE_GRID),
        },
        "fixed_internal_qpr_defaults": {
            "alpha": FIXED_QPR[0],
            "mass_threshold": FIXED_QPR[1],
            "min_count": FIXED_QPR[2],
            "logit_scale": FIXED_QPR[3],
        },
        "selected": {
            "key": selected_key,
            "alpha": float(selected_values[0]),
            "mass_threshold": float(selected_values[1]),
            "min_count": int(selected_values[2]),
            "logit_scale": float(selected_values[3]),
            "metrics": selected_row,
            "minus_shrinksel": float(
                selected_row["class_balanced"]["mean"]
            )
            - float(baseline["class_balanced"]["mean"]),
        },
        "evaluation": evaluation,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit(
    pools: BaseTransductivePools,
    shrink: Mapping[str, object],
    parameters: Mapping[str, object],
    episodes: int,
) -> Dict[str, object]:
    if parameters["dataset"] != pools.dataset:
        raise RuntimeError("QPR selection/audit dataset mismatch")
    if parameters["selection_bundle_sha256"] != pools.bundle_sha256:
        raise RuntimeError("QPR selection/audit bundle mismatch")
    if parameters["source_feature_file_sha256"] != pools.source_feature_file_sha256:
        raise RuntimeError("QPR selection/audit feature mismatch")
    if parameters["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("QPR selection/audit shrinkage mismatch")
    prototype_mode = str(shrink.get("selected_prototype_mode", "normalized"))
    if parameters.get("selected_prototype_mode", "normalized") != prototype_mode:
        raise RuntimeError("QPR selection/audit prototype mode mismatch")
    selected_values = (
        float(parameters["selected"]["alpha"]),
        float(parameters["selected"]["mass_threshold"]),
        int(parameters["selected"]["min_count"]),
        float(parameters["selected"]["logit_scale"]),
    )
    candidates = tuple(dict.fromkeys((selected_values, FIXED_QPR)))
    evaluation = evaluate(
        pools,
        float(shrink["selected_lambda"]),
        prototype_mode,
        episodes,
        namespace_seed=141421,
        candidates=candidates,
    )
    baseline = evaluation["combined"]["shrinksel"]
    selected = evaluation["combined"][parameter_key(selected_values)]
    balanced_gain = float(selected["class_balanced"]["mean"]) - float(
        baseline["class_balanced"]["mean"]
    )
    anchor_gain = float(selected["anchor"]["mean"]) - float(
        baseline["anchor"]["mean"]
    )
    passes = bool(
        np.isfinite(balanced_gain)
        and np.isfinite(anchor_gain)
        and balanced_gain >= -0.10
        and anchor_gain >= -0.50
    )
    deployed = selected_values if passes else (
        0.0,
        selected_values[1],
        selected_values[2],
        selected_values[3],
    )
    result: Dict[str, object] = {
        "schema_version": 2,
        "stage": "qpr_untouched_base_audit",
        "dataset": pools.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": pools.source_feature_file_sha256,
        "selected": dict(parameters["selected"]),
        "deployed": {
            "alpha": float(deployed[0]),
            "mass_threshold": float(deployed[1]),
            "min_count": int(deployed[2]),
            "logit_scale": float(deployed[3]),
        },
        "selected_class_balanced_gain": balanced_gain,
        "selected_anchor_gain": anchor_gain,
        "passes_transfer_safety_gate": passes,
        "evaluation": evaluation,
    }
    result["audit_result_sha256"] = stable_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--bundle", required=True, type=Path)
        sub.add_argument("--shrink-parameters", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--episodes", type=int, default=20)
        sub.add_argument("--device", default="cuda")
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        pools = _from_selection(
            load_construction_selection(args.bundle, args.device)
        )
        shrink = load_shrink_parameters(args.shrink_parameters, pools)
        result = select(pools, shrink, args.episodes)
    else:
        pools = _from_audit(load_audit(args.bundle, args.device))
        shrink = load_shrink_parameters(args.shrink_parameters, pools)
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit(pools, shrink, parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
