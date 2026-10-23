#!/usr/bin/env python3
"""Corrected class-disjoint selection and audit for QPR.

The legacy QPR calibration episode placed held-out anchor images and candidate
identities in the same sampling mapping.  It could therefore assign one
semantic identity both an anchor label and a pseudo-novel label.  It also
forced anchor centers to unit norm even when the parent ShrinkSel head selected
another radial treatment.  This module fixes both mismatches while preserving
the committed QPR grid, objectives, constraints, and deterministic seeds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .constants import (
    PSEUDO_NOVEL_PER_EPISODE,
    PSEUDO_QUERY_PER_CLASS,
    RIDGE_GAMMA,
)
from .cumulative_radial_selection import _count_summary, cumulative_requested_counts
from .geometry import covariance_eigensystem, metric_scores
from .qpr_selection import (
    FIXED_QPR,
    QPR_GRID,
    load_shrink_parameters,
    parameter_key,
)
from .radial_selection import prototype_from_samples
from .selection import _anchor_split, load_audit, load_construction_selection
from .transductive_methods import query_prototype_rectification
from .transductive_selection import BaseTransductivePools, _from_audit, _from_selection
from .utils import atomic_write_json, stable_json_sha256


REGIMES = ("one_increment", "cumulative")
CORRECTION = (
    "strict anchor/pseudo identity disjointness and parent-ShrinkSel prototype "
    "treatment on both anchor and pseudo-novel classes"
)


@dataclass(frozen=True)
class CleanCalibrationEpisode:
    prototypes: torch.Tensor
    cohort: torch.Tensor
    cohort_labels: torch.Tensor
    num_anchor: int
    anchor_identity_ids: Tuple[int, ...]
    pseudo_identity_ids: Tuple[int, ...]
    requested_pseudo_count: int


def requested_count_schedule(dataset: str, regime: str) -> Tuple[int, ...]:
    if regime not in REGIMES:
        raise ValueError("unknown QPR calibration regime: {}".format(regime))
    if regime == "one_increment":
        return (int(PSEUDO_NOVEL_PER_EPISODE[dataset]),)
    return cumulative_requested_counts(dataset)


def _clean_episode(
    pools: BaseTransductivePools,
    construction: Mapping[int, torch.Tensor],
    anchor_cohort: Mapping[int, torch.Tensor],
    candidate_pool: Mapping[int, torch.Tensor],
    episode_index: int,
    imbalanced: bool,
    requested_pseudo_count: int,
    prototype_mode: str,
) -> CleanCalibrationEpisode:
    anchor_ids = tuple(sorted(int(value) for value in construction))
    candidate_ids = tuple(sorted(int(value) for value in candidate_pool))
    overlap = set(anchor_ids).intersection(candidate_ids)
    if overlap:
        raise RuntimeError(
            "anchor and pseudo-novel identities overlap: {}".format(sorted(overlap))
        )
    if set(anchor_cohort) != set(construction):
        raise RuntimeError("anchor construction/cohort identity sets disagree")
    count = min(int(requested_pseudo_count), len(candidate_ids))
    if count < 1:
        raise RuntimeError("a calibration episode requires a pseudo-novel identity")
    rng = np.random.RandomState(7717 + int(episode_index) * 104729)
    chosen = tuple(
        sorted(
            int(value)
            for value in rng.choice(
                np.asarray(candidate_ids), size=count, replace=False
            ).tolist()
        )
    )
    if set(anchor_ids).intersection(chosen):
        raise AssertionError("clean sampler produced an identity collision")

    prototypes: List[torch.Tensor] = []
    cohort_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    device = next(iter(construction.values())).device
    ordered = anchor_ids + chosen
    for new_label, class_id in enumerate(ordered):
        if new_label < len(anchor_ids):
            prototypes.append(
                prototype_from_samples(construction[class_id], prototype_mode)
            )
            values = anchor_cohort[class_id]
        else:
            values = candidate_pool[class_id]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                32452843 + int(episode_index) * 49999 + class_id * 15485863
            )
            order = torch.randperm(values.shape[0], generator=generator).to(values.device)
            support = values[order[:5]]
            values = values[order[5 : 5 + PSEUDO_QUERY_PER_CLASS]]
            prototypes.append(prototype_from_samples(support, prototype_mode))
        available = min(PSEUDO_QUERY_PER_CLASS, int(values.shape[0]))
        if available < 2:
            raise RuntimeError(
                "insufficient calibration cohort samples for identity {}".format(class_id)
            )
        if imbalanced:
            take = 2 + (
                int(episode_index) * 17 + class_id * 13 + new_label * 7
            ) % (available - 1)
        else:
            take = available
        offset = (int(episode_index) * 3 + class_id) % available
        index = torch.arange(offset, offset + take, device=device) % available
        cohort_parts.append(values[index])
        label_parts.append(
            torch.full((take,), new_label, device=device, dtype=torch.long)
        )
    return CleanCalibrationEpisode(
        prototypes=torch.stack(prototypes, 0),
        cohort=torch.cat(cohort_parts, 0),
        cohort_labels=torch.cat(label_parts, 0),
        num_anchor=len(anchor_ids),
        anchor_identity_ids=anchor_ids,
        pseudo_identity_ids=chosen,
        requested_pseudo_count=int(requested_pseudo_count),
    )


def _metrics(
    prediction: torch.Tensor, episode: CleanCalibrationEpisode
) -> Dict[str, float]:
    correct = prediction == episode.cohort_labels
    anchor = episode.cohort_labels < int(episode.num_anchor)
    novel = ~anchor
    per_class = []
    for class_id in range(int(episode.prototypes.shape[0])):
        mask = episode.cohort_labels == class_id
        if not bool(mask.any()):
            raise RuntimeError("calibration class has no query examples")
        per_class.append(float(correct[mask].float().mean().item()))
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "class_balanced": 100.0 * float(mean(per_class)),
        "anchor": 100.0 * float(correct[anchor].float().mean().item()),
        "pseudo_novel": 100.0 * float(correct[novel].float().mean().item()),
    }


def _gate_diagnostics(
    baseline_scores: torch.Tensor,
    episode: CleanCalibrationEpisode,
    mass_threshold: float,
    logit_scale: float,
) -> Dict[str, float]:
    probability = torch.softmax(baseline_scores.float() * float(logit_scale), dim=1)
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


def _summarize(rows: Mapping[str, List[Dict[str, float]]]) -> Dict[str, object]:
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
    regime: str,
) -> Dict[str, object]:
    construction, anchor_cohort = _anchor_split(pools.construction, namespace_seed)
    candidate_pool = {int(key): value for key, value in pools.candidate.items()}
    overlap = set(construction).intersection(candidate_pool)
    if overlap:
        raise RuntimeError(
            "base bundle violates anchor/pseudo identity disjointness: {}".format(
                sorted(overlap)
            )
        )
    covariance = covariance_eigensystem(dict(construction), sorted(construction))
    keys = [parameter_key(values) for values in candidates]
    rows: Dict[str, List[Dict[str, float]]] = {"shrinksel": []}
    rows.update({key: [] for key in keys})
    condition_rows = {
        condition: {key: [] for key in rows}
        for condition in ("balanced", "imbalanced")
    }
    schedule = requested_count_schedule(pools.dataset, regime)
    requested_counts: List[int] = []
    realized_counts: List[int] = []

    for episode_index in range(int(episodes)):
        requested = int(schedule[episode_index % len(schedule)])
        requested_counts.append(requested)
        realized_counts.append(min(requested, len(candidate_pool)))
        for condition in ("balanced", "imbalanced"):
            episode = _clean_episode(
                pools,
                construction,
                anchor_cohort,
                candidate_pool,
                episode_index,
                imbalanced=condition == "imbalanced",
                requested_pseudo_count=requested,
                prototype_mode=prototype_mode,
            )
            if set(episode.anchor_identity_ids).intersection(episode.pseudo_identity_ids):
                raise AssertionError("identity-disjointness invariant failed")
            baseline_scores = metric_scores(
                episode.cohort,
                episode.prototypes,
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
                    episode.prototypes,
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
                    _gate_diagnostics(
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
        "combined": _summarize(rows),
        "conditions": {
            condition: _summarize(values)
            for condition, values in condition_rows.items()
        },
        "episodes_per_condition": int(episodes),
        "calibration_conditions": ["balanced", "deterministic_imbalanced"],
        "balanced_query_policy": "15 available queries per identity",
        "imbalanced_query_policy": (
            "fixed prediction-independent 2--15 queries per identity; exact "
            "indices are determined by the committed episode and identity seeds"
        ),
        "calibration_regime": regime,
        "requested_count_schedule": list(schedule),
        "count_coverage_per_condition": _count_summary(
            requested_counts, realized_counts
        ),
        "identity_policy": "anchor and pseudo-novel semantic identities are disjoint",
        "identity_overlap_violations": 0,
        "anchor_identity_count": int(len(construction)),
        "candidate_identity_count": int(len(candidate_pool)),
        "prototype_policy": (
            "apply the parent ShrinkSel prototype treatment to anchor construction "
            "samples and each five-shot pseudo-novel support"
        ),
        "covariance_dof": int(covariance.dof),
        "selected_lambda": float(selected_lambda),
        "selected_prototype_mode": prototype_mode,
    }


def select_clean(
    pools: BaseTransductivePools,
    shrink: Mapping[str, object],
    episodes: int,
    regime: str,
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
        regime=regime,
    )
    curve = evaluation["combined"]
    baseline = curve["shrinksel"]
    feasible = []
    for values in QPR_GRID:
        key = parameter_key(values)
        row = curve[key]
        anchor_safe = float(row["anchor"]["mean"]) >= (
            float(baseline["anchor"]["mean"]) - 0.5
        )
        if anchor_safe:
            feasible.append((values, key, row))
    if not feasible:
        raise RuntimeError("QPR grid contains no anchor-safe base candidate")
    selected_values, selected_key, selected_row = max(
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
    result: Dict[str, object] = {
        "schema_version": 3,
        "stage": "qpr_base_only_selection",
        "selector_correction": CORRECTION,
        "calibration_regime": regime,
        "dataset": pools.dataset,
        "selection_bundle_sha256": pools.bundle_sha256,
        "source_feature_file_sha256": pools.source_feature_file_sha256,
        "shrink_selection_result_sha256": shrink["selection_result_sha256"],
        "selected_lambda": selected_lambda,
        "selected_prototype_mode": prototype_mode,
        "objective": (
            "class-balanced accuracy over balanced and deterministic-imbalanced "
            "identity-disjoint base pseudo-novel episodes with anchor loss <=0.5 point"
        ),
        "grid": {
            "alpha": sorted({float(value[0]) for value in QPR_GRID}),
            "mass_threshold": sorted({float(value[1]) for value in QPR_GRID}),
            "min_count": sorted({int(value[2]) for value in QPR_GRID}),
            "logit_scale": sorted({float(value[3]) for value in QPR_GRID}),
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
            "minus_shrinksel": float(selected_row["class_balanced"]["mean"])
            - float(baseline["class_balanced"]["mean"]),
        },
        "evaluation": evaluation,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_clean(
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
    if parameters.get("selector_correction") != CORRECTION:
        raise RuntimeError("QPR parameters do not contain the clean-selector correction")
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
    regime = str(parameters["calibration_regime"])
    evaluation = evaluate(
        pools,
        float(shrink["selected_lambda"]),
        prototype_mode,
        episodes,
        namespace_seed=141421,
        candidates=candidates,
        regime=regime,
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
    deployed = selected_values if passes else (0.0, selected_values[1], selected_values[2], selected_values[3])
    result: Dict[str, object] = {
        "schema_version": 3,
        "stage": "qpr_untouched_base_audit",
        "selector_correction": CORRECTION,
        "calibration_regime": regime,
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
        sub.add_argument("--regime", choices=REGIMES, default="one_increment")
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        pools = _from_selection(load_construction_selection(args.bundle, args.device))
        shrink = load_shrink_parameters(args.shrink_parameters, pools)
        result = select_clean(pools, shrink, args.episodes, args.regime)
    else:
        pools = _from_audit(load_audit(args.bundle, args.device))
        shrink = load_shrink_parameters(args.shrink_parameters, pools)
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        if str(parameters["calibration_regime"]) != args.regime:
            raise RuntimeError("CLI regime disagrees with selected parameters")
        result = audit_clean(pools, shrink, parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
