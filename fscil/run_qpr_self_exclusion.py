#!/usr/bin/env python3
"""Matched target evaluation of same-cohort and self-excluded QPR."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .qpr_calibration import CORRECTION
from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .qpr_selection import deployed_qpr_choice
from .qpr_self_exclusion import qpr_same_and_leave_one_out_scores
from .run_qpr import gate_diagnostics
from .run_transductive import _accuracy, _measure, _support_summary
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, stable_json_sha256


METHODS = ("shrinksel", "qpr_same_cohort", "qpr_leave_one_out")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260825


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parameters(values: Mapping[str, object]) -> Tuple[float, float, int, float]:
    return (
        float(values["alpha"]),
        float(values["mass_threshold"]),
        int(values["min_count"]),
        float(values["logit_scale"]),
    )


def _aggregate_seed_metrics(
    per_seed: Sequence[Mapping[str, object]], method: str, scope: str
) -> Dict[str, List[float]]:
    result = {metric: [] for metric in ("overall", "class_balanced", "base", "novel")}
    for seed_row in per_seed:
        sessions = list(seed_row["sessions"])
        selected = sessions if scope == "session_average" else sessions[-1:]
        for metric in result:
            observed = [
                float(session["methods"][method][metric])
                for session in selected
                if session["methods"][method][metric] is not None
            ]
            if observed:
                result[metric].append(float(mean(observed)))
    return result


def _summary(per_seed: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for scope in ("session_average", "final"):
        result[scope] = {}
        for method in METHODS:
            values = _aggregate_seed_metrics(per_seed, method, scope)
            result[scope][method] = {
                metric: {
                    "mean": float(mean(observed)) if observed else None,
                    "std": float(pstdev(observed)) if observed else None,
                    "support_seeds": len(observed),
                    "per_seed": observed,
                }
                for metric, observed in values.items()
            }
    return result


def _paired_bootstrap(left: Sequence[float], right: Sequence[float]) -> Dict[str, object]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, difference.shape[0], size=(BOOTSTRAP_DRAWS, difference.shape[0]))
    estimates = difference[indices].mean(1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "mean": float(difference.mean()),
        "ci95": [float(lower), float(upper)],
        "per_seed": difference.tolist(),
        "support_seeds": int(difference.shape[0]),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _comparisons(summary: Mapping[str, object]) -> Dict[str, object]:
    pairs = (
        ("qpr_same_cohort", "shrinksel"),
        ("qpr_leave_one_out", "shrinksel"),
        ("qpr_leave_one_out", "qpr_same_cohort"),
    )
    result: Dict[str, object] = {}
    for scope in ("session_average", "final"):
        result[scope] = {}
        for left, right in pairs:
            name = left + "_minus_" + right
            result[scope][name] = {}
            for metric in ("overall", "class_balanced", "base", "novel"):
                left_values = summary[scope][left][metric]["per_seed"]
                right_values = summary[scope][right][metric]["per_seed"]
                if left_values and right_values:
                    result[scope][name][metric] = _paired_bootstrap(
                        left_values, right_values
                    )
                else:
                    result[scope][name][metric] = None
    return result


def _noop_metadata() -> Dict[str, object]:
    return {
        "base_only_session_noop": True,
        "gated_query_count": 0,
        "contributing_query_count": 0,
        "affected_novel_class_count": 0,
        "corrected_query_class_score_entries": 0,
        "minimum_count_fallback_entries": 0,
        "classified_query_present_in_its_adaptation_set": False,
        "uses_query_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--qpr-parameters", required=True, type=Path)
    parser.add_argument("--qpr-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--sessions", choices=("final", "all"), default="all")
    parser.add_argument("--shot-count", type=int, default=5)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--skip-file-hash", action="store_true")
    args = parser.parse_args()
    started = time.time()

    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid seed limit")
    if int(args.shot_count) != 5:
        raise ValueError("the frozen self-exclusion experiment is five-shot")
    shrink = _load(args.shrink_parameters)
    qpr = _load(args.qpr_parameters)
    audit = _load(args.qpr_audit)
    for payload, name in ((shrink, "shrink"), (qpr, "QPR"), (audit, "QPR audit")):
        if payload["dataset"] != data.dataset:
            raise RuntimeError(name + " dataset mismatch")
    if shrink["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("shrinkage feature mismatch")
    if qpr["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("QPR feature mismatch")
    if qpr.get("selector_correction") != CORRECTION:
        raise RuntimeError("QPR parameters were not produced by the corrected selector")
    if audit.get("selector_correction") != CORRECTION:
        raise RuntimeError("QPR audit was not produced by the corrected selector")
    if audit["stage"] != "qpr_untouched_base_audit":
        raise RuntimeError("invalid QPR audit stage")
    if audit["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("QPR audit feature mismatch")
    if qpr["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("QPR was not selected for this ShrinkSel head")
    prototype_mode = str(shrink.get("selected_prototype_mode", "normalized"))
    if qpr.get("selected_prototype_mode", "normalized") != prototype_mode:
        raise RuntimeError("QPR prototype mode disagrees with ShrinkSel")
    if float(qpr["selected_lambda"]) != float(shrink["selected_lambda"]):
        raise RuntimeError("QPR geometry disagrees with ShrinkSel")
    parameters = _parameters(deployed_qpr_choice(qpr, audit))

    by_class = features_by_class(data)
    covariance = covariance_eigensystem(by_class, range(data.num_base))
    selected_lambda = float(shrink["selected_lambda"])
    session_count = 1 + (data.num_total - data.num_base) // data.increment
    session_indices = (
        [session_count - 1]
        if args.sessions == "final"
        else list(range(session_count))
    )
    per_seed: List[Dict[str, object]] = []
    resources = {
        method: {"runtime_sec": 0.0, "peak_allocated_bytes": 0, "calls": 0}
        for method in METHODS
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
            query = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            _, _, _, prototypes = _support_summary(
                data,
                by_class,
                seed_position,
                end_class,
                prototype_mode,
                shot_count=int(args.shot_count),
            )
            baseline_scores, baseline_elapsed, baseline_peak = _measure(
                lambda: metric_scores(
                    query,
                    prototypes,
                    covariance,
                    selected_lambda,
                    RIDGE_GAMMA,
                    int(args.query_chunk),
                ),
                args.device,
            )
            predictions = {"shrinksel": baseline_scores.argmax(1)}
            resources["shrinksel"]["runtime_sec"] += baseline_elapsed
            resources["shrinksel"]["peak_allocated_bytes"] = max(
                resources["shrinksel"]["peak_allocated_bytes"], baseline_peak
            )
            resources["shrinksel"]["calls"] += 1
            if end_class == data.num_base:
                predictions["qpr_same_cohort"] = predictions["shrinksel"].clone()
                predictions["qpr_leave_one_out"] = predictions["shrinksel"].clone()
                exclusion_metadata = _noop_metadata()
                gate = None
                qpr_elapsed = 0.0
                qpr_peak = baseline_peak
            else:
                value, qpr_elapsed, qpr_peak = _measure(
                    lambda: qpr_same_and_leave_one_out_scores(
                        query,
                        prototypes,
                        baseline_scores,
                        covariance,
                        selected_lambda,
                        RIDGE_GAMMA,
                        data.num_base,
                        alpha=parameters[0],
                        mass_threshold=parameters[1],
                        min_count=parameters[2],
                        logit_scale=parameters[3],
                        query_chunk=int(args.query_chunk),
                    ),
                    args.device,
                )
                predictions["qpr_same_cohort"] = value.same_cohort_scores.argmax(1)
                predictions["qpr_leave_one_out"] = value.leave_one_out_scores.argmax(1)
                exclusion_metadata = dict(value.metadata)
                gate = gate_diagnostics(
                    baseline_scores,
                    labels,
                    data.num_base,
                    parameters,
                    value.metadata,
                )
            for method in ("qpr_same_cohort", "qpr_leave_one_out"):
                resources[method]["runtime_sec"] += baseline_elapsed + qpr_elapsed
                resources[method]["peak_allocated_bytes"] = max(
                    resources[method]["peak_allocated_bytes"],
                    baseline_peak,
                    qpr_peak,
                )
                resources[method]["calls"] += 1
            session_row = {
                "session_index": int(session_index),
                "end_class": int(end_class),
                "query_count": int(query.shape[0]),
                "methods": {
                    method: _accuracy(
                        predictions[method], labels, data.num_base, end_class
                    )
                    for method in METHODS
                },
                "qpr_gate_diagnostics": gate,
                "self_exclusion_diagnostics": exclusion_metadata,
                "target_label_role": (
                    "evaluation harness instantiates the protocol-defined seen-class "
                    "cohort and computes metrics; QPR inference receives no labels"
                ),
            }
            seed_row["sessions"].append(session_row)
        per_seed.append(seed_row)
        averages = {
            method: mean(
                float(row["methods"][method]["overall"])
                for row in seed_row["sessions"]
            )
            for method in METHODS
        }
        print(
            "dataset={} seed={} {}".format(
                data.dataset,
                data.support_seeds[seed_position],
                " ".join(
                    "{}={:.3f}".format(method, averages[method]) for method in METHODS
                ),
            ),
            flush=True,
        )

    summary = _summary(per_seed)
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "qpr_same_cohort_vs_exact_query_self_exclusion",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "methods": list(METHODS),
        "information_regimes": {
            "shrinksel": "support-only",
            "qpr_same_cohort": (
                "unlabeled same-cohort batch transduction with query self-inclusion"
            ),
            "qpr_leave_one_out": (
                "unlabeled cohort adaptation with every classified query excluded "
                "from its own prototype update"
            ),
        },
        "calibration_regime": qpr["calibration_regime"],
        "selector_correction": CORRECTION,
        "audit_passed": bool(audit["passes_transfer_safety_gate"]),
        "fallback_deployed": not bool(audit["passes_transfer_safety_gate"]),
        "deployed_parameters": {
            "alpha": parameters[0],
            "mass_threshold": parameters[1],
            "min_count": parameters[2],
            "logit_scale": parameters[3],
        },
        "shrinksel_lambda": selected_lambda,
        "shrinksel_prototype_mode": prototype_mode,
        "shot_count": int(args.shot_count),
        "session_scope": args.sessions,
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "selection_source_sha256": data.selection_source_sha256,
        "selection_provenance": {
            "shrinksel": shrink["selection_result_sha256"],
            "qpr": qpr["selection_result_sha256"],
            "qpr_audit": audit["audit_result_sha256"],
        },
        "support_seeds": data.support_seeds[: args.seed_limit],
        "per_seed": per_seed,
        "summary": summary,
        "paired_comparisons": _comparisons(summary),
        "resources": resources,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "summary": summary,
                "paired_comparisons": result["paired_comparisons"],
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
