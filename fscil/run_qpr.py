#!/usr/bin/env python3
"""Evaluate this work's QPR as an I2 add-on to the locked ShrinkSel head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from .constants import PROTOCOL_SEED, RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .qpr_base_guard import (
    REHEARSAL_PER_BASE_CLASS,
    base_guarded_prototypes,
    boundary_rehearsal_coreset,
)
from .qpr_selection import FIXED_QPR, deployed_qpr_choice
from .run_transductive import (
    _accuracy,
    _fixed_balanced_subset,
    _fixed_imbalanced_subset,
    _measure,
    _mode_batches,
    _support_summary,
)
from .target_data import features_by_class, load_target_evaluation_data
from .transductive_methods import query_prototype_rectification
from .utils import atomic_write_json, backbone_display, stable_json_sha256


METHODS = (
    "shrinksel",
    "qpr_selected",
    "qpr_fixed_transfer",
    "qpr_margin",
    "qpr_guard",
)
DEFAULT_METHODS = ("shrinksel", "qpr_selected", "qpr_fixed_transfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--qpr-parameters", required=True, type=Path)
    parser.add_argument("--qpr-audit", required=True, type=Path)
    parser.add_argument("--qpr-margin-parameters", type=Path)
    parser.add_argument("--qpr-margin-audit", type=Path)
    parser.add_argument("--qpr-guard-parameters", type=Path)
    parser.add_argument("--qpr-guard-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=5)
    parser.add_argument("--sessions", choices=("final", "all"), default="final")
    parser.add_argument("--modes", default="full")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--shot-count", type=int, default=5)
    parser.add_argument("--cohort-seed-offset", type=int, default=0)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parameters_from_mapping(
    values: Mapping[str, object]
) -> Tuple[float, float, int, float]:
    return (
        float(values["alpha"]),
        float(values["mass_threshold"]),
        int(values["min_count"]),
        float(values["logit_scale"]),
    )


def gate_diagnostics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    num_base: int,
    parameters: Sequence[float],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    _, threshold, _, scale = parameters[:4]
    margin_threshold = float(parameters[4]) if len(parameters) > 4 else 0.0
    probability = torch.softmax(scores.float() * float(scale), dim=1)
    predicted = probability.argmax(1)
    novel_probability = probability[:, int(num_base) :]
    novel_mass = novel_probability.sum(1)
    conditional = novel_probability / novel_mass.unsqueeze(1).clamp_min(1e-12)
    assigned = conditional.argmax(1) + int(num_base)
    if conditional.shape[1] > 1:
        top_two = conditional.topk(2, dim=1).values
        novel_margin = top_two[:, 0] - top_two[:, 1]
    else:
        novel_margin = torch.ones_like(novel_mass)
    gate = (predicted >= int(num_base)) & (
        novel_mass >= float(threshold)
    ) & (novel_margin >= margin_threshold)
    gated = int(gate.sum().item())
    gated_base = int((gate & (labels < int(num_base))).sum().item())
    gated_novel = gated - gated_base
    correctly_assigned = int((gate & (assigned == labels)).sum().item())
    return {
        "gated_query_count": gated,
        "gated_base_count": gated_base,
        "gated_novel_count": gated_novel,
        "gate_group_purity": (
            100.0 * gated_novel / gated if gated else 100.0
        ),
        "gate_assignment_purity": (
            100.0 * correctly_assigned / gated if gated else 100.0
        ),
        "gated_novel_margin_mean": (
            float(novel_margin[gate].mean().item()) if gated else 1.0
        ),
        "eligible_class_count": int(metadata["eligible_class_count"]),
        "updated_class_count": int(metadata["updated_class_count"]),
        "diagnostic_labels_used_after_prediction_only": True,
    }


def combine_gate_calls(calls: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    gated = sum(int(row["gated_query_count"]) for row in calls)
    gated_base = sum(int(row["gated_base_count"]) for row in calls)
    gated_novel = sum(int(row["gated_novel_count"]) for row in calls)
    assignment_correct = sum(
        int(round(float(row["gate_assignment_purity"]) * int(row["gated_query_count"]) / 100.0))
        for row in calls
    )
    margin_sum = sum(
        float(row["gated_novel_margin_mean"]) * int(row["gated_query_count"])
        for row in calls
    )
    return {
        "calls": len(calls),
        "gated_query_count": gated,
        "gated_base_count": gated_base,
        "gated_novel_count": gated_novel,
        "gate_group_purity": 100.0 * gated_novel / gated if gated else 100.0,
        "gate_assignment_purity": (
            100.0 * assignment_correct / gated if gated else 100.0
        ),
        "gated_novel_margin_mean": margin_sum / gated if gated else 1.0,
        "eligible_class_count_mean": float(
            mean(float(row["eligible_class_count"]) for row in calls)
        ),
        "updated_class_count_mean": float(
            mean(float(row["updated_class_count"]) for row in calls)
        ),
    }


def base_only_noop_gate_diagnostics() -> Dict[str, object]:
    """Record the base session as an explicit QPR no-op.

    QPR is defined only after at least one novel class arrives.  Keeping session zero
    in an all-session average is nevertheless important, so every QPR variant copies
    the locked ShrinkSel prediction and records a zero-update call there.
    """
    return {
        "gated_query_count": 0,
        "gated_base_count": 0,
        "gated_novel_count": 0,
        "gate_group_purity": 100.0,
        "gate_assignment_purity": 100.0,
        "gated_novel_margin_mean": 1.0,
        "eligible_class_count": 0,
        "updated_class_count": 0,
        "diagnostic_labels_used_after_prediction_only": True,
        "base_only_session_noop": True,
    }


def summarize(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for mode, methods in rows.items():
        result[mode] = {}
        for method, values in methods.items():
            result[mode][method] = {}
            for metric in ("overall", "class_balanced", "base", "novel"):
                observed = [
                    float(row[metric])
                    for row in values
                    if row[metric] is not None
                ]
                result[mode][method][metric + "_mean"] = (
                    float(mean(observed)) if observed else None
                )
                result[mode][method][metric + "_std"] = (
                    float(pstdev(observed)) if observed else None
                )
            result[mode][method]["evaluations"] = len(values)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    requested = tuple(
        value.strip() for value in args.methods.split(",") if value.strip()
    )
    unknown = sorted(set(requested).difference(METHODS))
    if unknown:
        raise ValueError("unknown QPR methods {}".format(unknown))
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid seed limit")
    if int(args.shot_count) not in (1, 2, 5, 10, 20):
        raise ValueError("--shot-count must be one of 1,2,5,10,20")
    shrink = load_json(args.shrink_parameters)
    qpr = load_json(args.qpr_parameters)
    audit = load_json(args.qpr_audit)
    for payload, name in ((shrink, "shrink"), (qpr, "QPR"), (audit, "QPR audit")):
        if payload["dataset"] != data.dataset:
            raise RuntimeError(name + " dataset mismatch")
    if shrink["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("shrinkage feature mismatch")
    if qpr["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("QPR feature mismatch")
    if audit["selection_result_sha256"] != qpr["selection_result_sha256"]:
        raise RuntimeError("QPR selection/audit provenance mismatch")
    prototype_mode = str(shrink.get("selected_prototype_mode", "normalized"))
    if qpr.get("selected_prototype_mode", "normalized") != prototype_mode:
        raise RuntimeError("QPR prototype mode disagrees with ShrinkSel")
    if float(qpr["selected_lambda"]) != float(shrink["selected_lambda"]):
        raise RuntimeError("QPR geometry disagrees with ShrinkSel")
    if qpr["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("QPR was not selected for this ShrinkSel head")

    selected = parameters_from_mapping(deployed_qpr_choice(qpr, audit))
    variants = {
        "qpr_selected": selected + (0.0,),
        "qpr_fixed_transfer": FIXED_QPR + (0.0,),
    }
    margin_parameters = None
    margin_audit = None
    if "qpr_margin" in requested:
        if args.qpr_margin_parameters is None or args.qpr_margin_audit is None:
            raise ValueError(
                "qpr_margin requires --qpr-margin-parameters and --qpr-margin-audit"
            )
        margin_parameters = load_json(args.qpr_margin_parameters)
        margin_audit = load_json(args.qpr_margin_audit)
        for payload, name in (
            (margin_parameters, "QPR-M"),
            (margin_audit, "QPR-M audit"),
        ):
            if payload["dataset"] != data.dataset:
                raise RuntimeError(name + " dataset mismatch")
            if payload["source_feature_file_sha256"] != data.selection_source_sha256:
                raise RuntimeError(name + " feature mismatch")
        if margin_parameters["qpr_selection_result_sha256"] != qpr["selection_result_sha256"]:
            raise RuntimeError("QPR-M parent selection mismatch")
        if margin_audit["selection_result_sha256"] != margin_parameters["selection_result_sha256"]:
            raise RuntimeError("QPR-M selection/audit provenance mismatch")
        if not bool(margin_audit["passes_transfer_safety_gate"]):
            raise RuntimeError("QPR-M failed its untouched base transfer-safety gate")
        enabled = bool(margin_parameters["selected"]["enabled"])
        variants["qpr_margin"] = (
            selected[0] if enabled else 0.0,
            selected[1],
            selected[2],
            selected[3],
            float(margin_parameters["selected"]["novel_margin_threshold"]),
        )
    guard_parameters = None
    guard_audit = None
    guard_enabled = False
    guard_ratio = 1.0
    if "qpr_guard" in requested:
        if args.qpr_guard_parameters is None or args.qpr_guard_audit is None:
            raise ValueError(
                "qpr_guard requires --qpr-guard-parameters and --qpr-guard-audit"
            )
        guard_parameters = load_json(args.qpr_guard_parameters)
        guard_audit = load_json(args.qpr_guard_audit)
        for payload, name in (
            (guard_parameters, "QPR-G"),
            (guard_audit, "QPR-G audit"),
        ):
            if payload["dataset"] != data.dataset:
                raise RuntimeError(name + " dataset mismatch")
            if payload["source_feature_file_sha256"] != data.selection_source_sha256:
                raise RuntimeError(name + " feature mismatch")
        if guard_parameters["qpr_selection_result_sha256"] != qpr["selection_result_sha256"]:
            raise RuntimeError("QPR-G parent selection mismatch")
        if guard_audit["selection_result_sha256"] != guard_parameters["selection_result_sha256"]:
            raise RuntimeError("QPR-G selection/audit provenance mismatch")
        if not bool(guard_audit["passes_transfer_safety_gate"]):
            raise RuntimeError("QPR-G failed its untouched base transfer-safety gate")
        guard_enabled = bool(guard_parameters["selected"]["enabled"])
        guard_ratio = float(guard_parameters["selected"]["separation_ratio"])
    by_class = features_by_class(data)
    covariance = covariance_eigensystem(by_class, range(data.num_base))
    selected_lambda = float(shrink["selected_lambda"])
    base_rehearsal = boundary_rehearsal_coreset(
        {class_id: by_class[class_id] for class_id in range(data.num_base)},
        covariance,
        selected_lambda,
        RIDGE_GAMMA,
        REHEARSAL_PER_BASE_CLASS,
    )
    session_count = 1 + (data.num_total - data.num_base) // data.increment
    session_indices = (
        [session_count - 1]
        if args.sessions == "final"
        else list(range(session_count))
    )
    rows: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        mode: {method: [] for method in requested} for mode in modes
    }
    per_seed: List[Dict[str, object]] = []
    resources = {
        method: {
            "runtime_sec": 0.0,
            "peak_allocated_bytes": 0,
            "calls": 0,
        }
        for method in requested
    }

    for seed_position in range(args.seed_limit):
        seed_row: Dict[str, object] = {
            "support_seed": int(data.support_seeds[seed_position]),
            "sessions": [],
        }
        for session_index in session_indices:
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            query_mask = data.test_labels < end_class
            standard_query = data.test_features[query_mask]
            standard_labels = data.test_labels[query_mask]
            _, _, _, prototypes = _support_summary(
                data,
                by_class,
                seed_position,
                end_class,
                prototype_mode,
                shot_count=int(args.shot_count),
            )
            session_row: Dict[str, object] = {
                "session_index": int(session_index),
                "end_class": int(end_class),
                "modes": {},
            }
            for mode in modes:
                query = standard_query
                labels = standard_labels
                construction: Dict[str, object] = {}
                if mode == "imbalanced_full" or mode.startswith("imbalanced"):
                    suffix = mode.replace("imbalanced", "").replace("_full", "")
                    imbalance_ratio = float(suffix) if suffix else 10.0
                    query, labels, _, construction = _fixed_imbalanced_subset(
                        query,
                        labels,
                        end_class,
                        data.known_test_counts[:end_class],
                        imbalance_ratio=imbalance_ratio,
                        namespace_seed=int(args.cohort_seed_offset),
                    )
                elif mode.startswith("balanced"):
                    queries_per_class = int(mode.replace("balanced", ""))
                    query, labels, _, construction = _fixed_balanced_subset(
                        query,
                        labels,
                        end_class,
                        queries_per_class,
                        namespace_seed=int(args.cohort_seed_offset),
                    )
                batches, batching = _mode_batches(
                    mode,
                    query.shape[0],
                    PROTOCOL_SEED
                    + int(data.support_seeds[seed_position]) * 10007
                    + session_index * 1009
                    + int(args.cohort_seed_offset) * 1000003,
                    query.device,
                )
                predictions = {
                    method: torch.full(
                        (query.shape[0],),
                        -1,
                        dtype=torch.long,
                        device=query.device,
                    )
                    for method in requested
                }
                gate_calls: Dict[str, List[Dict[str, object]]] = {
                    method: [] for method in requested if method.startswith("qpr_")
                }
                guard_calls: List[Dict[str, object]] = []
                for context_index, retained in batches:
                    cohort = query[context_index]
                    cohort_labels = labels[context_index]
                    baseline_scores, elapsed, peak = _measure(
                        lambda: metric_scores(
                            cohort,
                            prototypes,
                            covariance,
                            selected_lambda,
                            RIDGE_GAMMA,
                        ),
                        args.device,
                    )
                    if "shrinksel" in requested:
                        predictions["shrinksel"][context_index[retained]] = (
                            baseline_scores[retained].argmax(1)
                        )
                        resources["shrinksel"]["runtime_sec"] += elapsed
                        resources["shrinksel"]["peak_allocated_bytes"] = max(
                            resources["shrinksel"]["peak_allocated_bytes"], peak
                        )
                        resources["shrinksel"]["calls"] += 1
                    for method, parameters in variants.items():
                        if method not in requested:
                            continue
                        if end_class == data.num_base:
                            predictions[method][context_index[retained]] = (
                                baseline_scores[retained].argmax(1)
                            )
                            resources[method]["runtime_sec"] += elapsed
                            resources[method]["peak_allocated_bytes"] = max(
                                resources[method]["peak_allocated_bytes"], peak
                            )
                            resources[method]["calls"] += 1
                            gate_calls[method].append(
                                base_only_noop_gate_diagnostics()
                            )
                            continue

                        def apply_qpr():
                            output = query_prototype_rectification(
                                cohort,
                                prototypes,
                                baseline_scores,
                                data.num_base,
                                alpha=parameters[0],
                                mass_threshold=parameters[1],
                                min_count=parameters[2],
                                logit_scale=parameters[3],
                                novel_margin_threshold=parameters[4],
                            )
                            scores = metric_scores(
                                cohort,
                                output.prototypes,
                                covariance,
                                selected_lambda,
                                RIDGE_GAMMA,
                            )
                            return output, scores

                        value, qpr_elapsed, qpr_peak = _measure(
                            apply_qpr, args.device
                        )
                        output, final_scores = value
                        predictions[method][context_index[retained]] = (
                            final_scores[retained].argmax(1)
                        )
                        resources[method]["runtime_sec"] += elapsed + qpr_elapsed
                        resources[method]["peak_allocated_bytes"] = max(
                            resources[method]["peak_allocated_bytes"],
                            peak,
                            qpr_peak,
                        )
                        resources[method]["calls"] += 1
                        gate_calls[method].append(
                            gate_diagnostics(
                                baseline_scores,
                                cohort_labels,
                                data.num_base,
                                parameters,
                                output.metadata,
                            )
                        )
                    if "qpr_guard" in requested:
                        if end_class == data.num_base:
                            predictions["qpr_guard"][context_index[retained]] = (
                                baseline_scores[retained].argmax(1)
                            )
                            resources["qpr_guard"]["runtime_sec"] += elapsed
                            resources["qpr_guard"]["peak_allocated_bytes"] = max(
                                resources["qpr_guard"]["peak_allocated_bytes"], peak
                            )
                            resources["qpr_guard"]["calls"] += 1
                            gate_calls["qpr_guard"].append(
                                base_only_noop_gate_diagnostics()
                            )
                            guard_calls.append(
                                {
                                    "proposal_class_count": 0,
                                    "accepted_class_count": 0,
                                    "full_step_class_count": 0,
                                    "mean_positive_step": 0.0,
                                    "base_rehearsal_prediction_changes": 0,
                                    "base_only_session_noop": True,
                                }
                            )
                            continue

                        def apply_qpr_guard():
                            parent_output = query_prototype_rectification(
                                cohort,
                                prototypes,
                                baseline_scores,
                                data.num_base,
                                alpha=selected[0] if guard_enabled else 0.0,
                                mass_threshold=selected[1],
                                min_count=selected[2],
                                logit_scale=selected[3],
                            )
                            guarded = base_guarded_prototypes(
                                prototypes,
                                parent_output.prototypes,
                                data.num_base,
                                base_rehearsal,
                                covariance,
                                selected_lambda,
                                RIDGE_GAMMA,
                                guard_ratio,
                            )
                            scores = metric_scores(
                                cohort,
                                guarded.prototypes,
                                covariance,
                                selected_lambda,
                                RIDGE_GAMMA,
                            )
                            return parent_output, guarded, scores

                        value, guard_elapsed, guard_peak = _measure(
                            apply_qpr_guard, args.device
                        )
                        parent_output, guarded, guard_scores = value
                        predictions["qpr_guard"][context_index[retained]] = (
                            guard_scores[retained].argmax(1)
                        )
                        resources["qpr_guard"]["runtime_sec"] += elapsed + guard_elapsed
                        resources["qpr_guard"]["peak_allocated_bytes"] = max(
                            resources["qpr_guard"]["peak_allocated_bytes"],
                            peak,
                            guard_peak,
                        )
                        resources["qpr_guard"]["calls"] += 1
                        gate_calls["qpr_guard"].append(
                            gate_diagnostics(
                                baseline_scores,
                                cohort_labels,
                                data.num_base,
                                selected + (0.0,),
                                parent_output.metadata,
                            )
                        )
                        guard_calls.append(dict(guarded.metadata))

                mode_row: Dict[str, object] = {
                    "query_count": int(query.shape[0]),
                    "batching": batching,
                    "construction": construction,
                    "methods": {},
                    "gate_diagnostics": {},
                }
                for method in requested:
                    if bool((predictions[method] < 0).any()):
                        raise RuntimeError("incomplete predictions for " + method)
                    metrics = _accuracy(
                        predictions[method], labels, data.num_base, end_class
                    )
                    mode_row["methods"][method] = metrics
                    rows[mode][method].append(metrics)
                    if method.startswith("qpr_"):
                        mode_row["gate_diagnostics"][method] = combine_gate_calls(
                            gate_calls[method]
                        )
                if "qpr_guard" in requested:
                    mode_row["guard_diagnostics"] = {
                        "calls": len(guard_calls),
                        "proposal_class_count_mean": float(
                            mean(float(row["proposal_class_count"]) for row in guard_calls)
                        ),
                        "accepted_class_count_mean": float(
                            mean(float(row["accepted_class_count"]) for row in guard_calls)
                        ),
                        "full_step_class_count_mean": float(
                            mean(float(row["full_step_class_count"]) for row in guard_calls)
                        ),
                        "mean_positive_step": float(
                            mean(float(row["mean_positive_step"]) for row in guard_calls)
                        ),
                        "base_rehearsal_prediction_changes": int(
                            sum(int(row["base_rehearsal_prediction_changes"]) for row in guard_calls)
                        ),
                        "base_rehearsal_count": int(base_rehearsal.shape[0]),
                    }
                session_row["modes"][mode] = mode_row
            seed_row["sessions"].append(session_row)
        per_seed.append(seed_row)

    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "qpr_target_evaluation_after_base_only_protocol_lock",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "information_levels": {
            "shrinksel": "I0/I1 strict inductive",
            "qpr_selected": (
                "I2 unlabeled query cohort; base-selected QPR operator "
                "on the locked ShrinkSel posterior"
            ),
            "qpr_fixed_transfer": (
                "I2 unlabeled query cohort; authors' fixed pre-commit QPR "
                "defaults applied to the ShrinkSel posterior"
            ),
            "qpr_margin": (
                "I2 unlabeled query cohort; base-selected within-novel margin "
                "gate applied to the locked QPR/ShrinkSel posterior"
            ),
            "qpr_guard": (
                "I2 unlabeled query cohort; base-only rehearsal/geometry-guarded "
                "QPR on the locked ShrinkSel posterior"
            ),
        },
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_source_sha256": data.selection_source_sha256,
        "shrinksel_lambda": selected_lambda,
        "shrinksel_prototype_mode": prototype_mode,
        "shot_count": int(args.shot_count),
        "cohort_seed_offset": int(args.cohort_seed_offset),
        "selection_provenance": {
            "shrinksel": shrink["selection_result_sha256"],
            "qpr": qpr["selection_result_sha256"],
            "qpr_audit": audit["audit_result_sha256"],
            "qpr_margin": (
                margin_parameters["selection_result_sha256"]
                if margin_parameters is not None
                else None
            ),
            "qpr_margin_audit": (
                margin_audit["audit_result_sha256"]
                if margin_audit is not None
                else None
            ),
            "qpr_guard": (
                guard_parameters["selection_result_sha256"]
                if guard_parameters is not None
                else None
            ),
            "qpr_guard_audit": (
                guard_audit["audit_result_sha256"]
                if guard_audit is not None
                else None
            ),
        },
        "selected_parameters": {
            "alpha": selected[0],
            "mass_threshold": selected[1],
            "min_count": selected[2],
            "logit_scale": selected[3],
        },
        "fixed_transfer_parameters": {
            "alpha": FIXED_QPR[0],
            "mass_threshold": FIXED_QPR[1],
            "min_count": FIXED_QPR[2],
            "logit_scale": FIXED_QPR[3],
        },
        "margin_parameters": (
            {
                "enabled": bool(margin_parameters["selected"]["enabled"]),
                "novel_margin_threshold": float(
                    margin_parameters["selected"]["novel_margin_threshold"]
                ),
            }
            if margin_parameters is not None
            else None
        ),
        "guard_parameters": (
            {
                "enabled": guard_enabled,
                "separation_ratio": guard_ratio,
                "base_rehearsal_count": int(base_rehearsal.shape[0]),
                "base_rehearsal_per_class_cap": REHEARSAL_PER_BASE_CLASS,
                "base_rehearsal_bytes": int(
                    base_rehearsal.numel() * base_rehearsal.element_size()
                ),
            }
            if guard_parameters is not None
            else None
        ),
        "methods": list(requested),
        "modes": list(modes),
        "session_scope": args.sessions,
        "support_seeds": data.support_seeds[: args.seed_limit],
        "per_seed": per_seed,
        "summary": summarize(rows),
        "resources": resources,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "summary": result["summary"],
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
