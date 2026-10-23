#!/usr/bin/env python3
"""Run strict support-only FSCIL heads on a locked feature/protocol bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .baseline_methods import (
    ULAParameters,
    class_specific_mahalanobis_scores_chunked,
    cosine_scores_chunked,
    diagonal_lda_scores_chunked,
    euclidean_scores_chunked,
    fecam_state,
    hycal_scores_chunked,
    hycal_state,
    hycal_state_bytes,
    ranpac_normal_equations,
    ranpac_scores_chunked,
    raw_prototypes,
    solve_ranpac,
    teen_calibrate,
    ula_adjust_logits,
)
from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, normalized_mean, stable_json_sha256


DEFAULT_METHODS = (
    "cosine",
    "euclidean",
    "covariance_lambda0",
    "shared_lda",
    "diagonal_lda",
    "shrinksel",
    "teen_selected",
    "teen_official",
    "ula_selected",
    "ula_official",
    "ranpac_no_projection",
    "fecam",
    "hycal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--shrink-audit", required=True, type=Path)
    parser.add_argument("--baseline-parameters", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--fecam-query-chunk", type=int, default=64)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--sessions",
        default="all",
        choices=("all", "final"),
        help="final is reserved for controlled resource benchmarks",
    )
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(
    function: Callable[[], torch.Tensor],
    device: str,
    warmup: int = 0,
    repeats: int = 1,
) -> Tuple[torch.Tensor, float, int]:
    if int(warmup) < 0 or int(repeats) < 1:
        raise ValueError("invalid timing warmup/repeat counts")
    for _ in range(int(warmup)):
        warmup_value = function()
        _synchronize(device)
        del warmup_value
    timings: List[float] = []
    peak = 0
    value = None
    for _ in range(int(repeats)):
        _synchronize(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        current = function()
        _synchronize(device)
        timings.append(time.time() - started)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
        if value is not None:
            del value
        value = current
    if value is None:
        raise RuntimeError("timing measurement produced no value")
    return value, float(median(timings)), peak


def _accuracy_row(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    num_base: int,
    end_class: int,
) -> Dict[str, object]:
    correct = prediction == labels
    base_mask = labels < int(num_base)
    novel_mask = ~base_mask
    per_class = []
    for class_id in range(int(end_class)):
        mask = labels == class_id
        per_class.append(100.0 * float(correct[mask].float().mean().item()))
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "base": 100.0 * float(correct[base_mask].float().mean().item()),
        "novel": (
            100.0 * float(correct[novel_mask].float().mean().item())
            if bool(novel_mask.any())
            else None
        ),
        "per_class": per_class,
    }


def _parameter_checks(
    data,
    shrink: Mapping[str, object],
    shrink_audit: Mapping[str, object],
    baselines: Mapping[str, object],
    baseline_audit: Mapping[str, object],
    methods: Sequence[str] = DEFAULT_METHODS,
) -> None:
    for name, payload in (("shrink", shrink), ("baselines", baselines)):
        if payload["dataset"] != data.dataset:
            raise RuntimeError("{} parameter dataset mismatch".format(name))
        if payload["source_feature_file_sha256"] != data.feature_file_sha256:
            raise RuntimeError("{} parameter feature hash mismatch".format(name))
    if shrink_audit["selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("ShrinkSel selection/audit provenance mismatch")
    if "shrinksel" in methods and not bool(
        shrink_audit["passes_non_regression_gate"]
    ):
        raise RuntimeError("ShrinkSel did not pass the untouched base audit")
    if not bool(baseline_audit["passes_sanity_gate"]):
        raise RuntimeError("one or more inductive baselines failed the base audit")
    if baseline_audit["selection_result_sha256"] != baselines["selection_result_sha256"]:
        raise RuntimeError("baseline selection/audit provenance mismatch")


def _summary(
    method_rows: Mapping[str, Sequence[Mapping[str, object]]], num_sessions: int
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for method, seed_rows in method_rows.items():
        session_average = [
            float(mean([float(row["overall"]) for row in seed_row["sessions"]]))
            for seed_row in seed_rows
        ]
        per_session = []
        for session_index in range(num_sessions):
            values = [
                float(seed_row["sessions"][session_index]["overall"])
                for seed_row in seed_rows
            ]
            per_session.append(
                {"mean": float(mean(values)), "std": float(pstdev(values))}
            )
        final_values = [
            float(seed_row["sessions"][-1]["overall"]) for seed_row in seed_rows
        ]
        result[method] = {
            "session_average_mean": float(mean(session_average)),
            "session_average_std": float(pstdev(session_average)),
            "session_average_per_seed": session_average,
            "final_accuracy_mean": float(mean(final_values)),
            "final_accuracy_std": float(pstdev(final_values)),
            "per_session": per_session,
        }
    return result


def main() -> None:
    args = parse_args()
    requested_methods = tuple(
        value.strip() for value in args.methods.split(",") if value.strip()
    )
    unknown = sorted(set(requested_methods).difference(DEFAULT_METHODS))
    if unknown:
        raise ValueError("unknown methods: {}".format(unknown))
    started = time.time()
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    dimension = int(data.train_features.shape[1])
    method_omissions: Dict[str, Dict[str, object]] = {}
    methods = requested_methods
    if "fecam" in methods and dimension > 1024:
        dense_state_bytes = int(data.num_total * dimension * dimension * 4)
        method_omissions["fecam"] = {
            "reason": (
                "official class-specific dense FeCAM exceeds the locked D<=1024 "
                "feasibility guard: O(C D^2) state, O(C D^3) covariance "
                "factorization, and O(Q C D^2) dense scoring"
            ),
            "feasibility_guard_max_dimension": 1024,
            "feature_dimension": dimension,
            "dense_inverse_covariance_state_bytes": dense_state_bytes,
            "excludes_eigensolver_workspace_and_query_tensors": True,
        }
        methods = tuple(method for method in methods if method != "fecam")
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid --seed-limit")
    shrink_parameters = _load_json(args.shrink_parameters)
    shrink_audit = _load_json(args.shrink_audit)
    baseline_parameters = _load_json(args.baseline_parameters)
    baseline_audit = _load_json(args.baseline_audit)
    _parameter_checks(
        data,
        shrink_parameters,
        shrink_audit,
        baseline_parameters,
        baseline_audit,
        methods,
    )

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    _synchronize(args.device)
    base_prototype_started = time.time()
    base_normalized = torch.stack(
        [normalized_mean(by_class[class_id]) for class_id in base_ids], 0
    )
    base_raw = raw_prototypes(by_class, base_ids)
    _synchronize(args.device)
    base_prototype_sec = time.time() - base_prototype_started

    _synchronize(args.device)
    covariance_started = time.time()
    covariance = covariance_eigensystem(by_class, base_ids)
    base_variance = (
        covariance.eigenvectors.square().matmul(covariance.eigenvalues)
    )
    _synchronize(args.device)
    covariance_construction_sec = time.time() - covariance_started
    selected_lambda = float(shrink_parameters["selected_lambda"])
    selected = baseline_parameters["selected"]
    teen_selected = selected["teen"]
    ula_selected = ULAParameters(**selected["ula"])
    ula_official = ULAParameters(**baseline_parameters["official_defaults"]["ula"])
    ranpac_ridge = float(selected["ranpac_no_projection"]["ridge"])

    resources: Dict[str, Dict[str, object]] = {
        method: {
            "head_construction_sec": 0.0,
            "support_update_sec": 0.0,
            "prediction_sec": 0.0,
            "peak_allocated_bytes": 0,
            "max_state_bytes": 0,
        }
        for method in methods
    }
    prototype_methods = {
        "cosine",
        "euclidean",
        "covariance_lambda0",
        "shared_lda",
        "diagonal_lda",
        "shrinksel",
        "teen_selected",
        "teen_official",
        "ula_selected",
        "ula_official",
    }
    covariance_methods = {
        "covariance_lambda0",
        "shared_lda",
        "diagonal_lda",
        "shrinksel",
    }
    for method in methods:
        if method in prototype_methods:
            resources[method]["head_construction_sec"] += base_prototype_sec
        if method in covariance_methods:
            resources[method]["head_construction_sec"] += (
                covariance_construction_sec
            )

    fecam_base_means = None
    fecam_base_inverse = None
    if "fecam" in methods:
        _synchronize(args.device)
        head_started = time.time()
        fecam_base_means, fecam_base_inverse = fecam_state(by_class, base_ids)
        _synchronize(args.device)
        resources["fecam"]["head_construction_sec"] += time.time() - head_started

    hycal_base_statistics = None
    if "hycal" in methods:
        _synchronize(args.device)
        head_started = time.time()
        hycal_base_statistics = hycal_state(by_class, base_ids)
        _synchronize(args.device)
        resources["hycal"]["head_construction_sec"] += time.time() - head_started

    ranpac_base_gram = None
    ranpac_base_cross = None
    if "ranpac_no_projection" in methods:
        base_mask = data.train_labels < data.num_base
        _synchronize(args.device)
        head_started = time.time()
        ranpac_base_gram, ranpac_base_cross = ranpac_normal_equations(
            data.train_features[base_mask],
            data.train_labels[base_mask],
            data.num_total,
        )
        _synchronize(args.device)
        resources["ranpac_no_projection"]["head_construction_sec"] += (
            time.time() - head_started
        )

    num_sessions = 1 + (data.num_total - data.num_base) // data.increment
    session_indices = (
        tuple(range(num_sessions))
        if args.sessions == "all"
        else (num_sessions - 1,)
    )
    rows: Dict[str, List[Dict[str, object]]] = {method: [] for method in methods}
    for seed_position in range(args.seed_limit):
        seed = data.support_seeds[seed_position]
        hycal_novel_statistics = []
        seed_rows: Dict[str, List[Dict[str, object]]] = {
            method: [] for method in methods
        }
        for session_index in session_indices:
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            query_mask = data.test_labels < end_class
            queries = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            _synchronize(args.device)
            support_statistics_started = time.time()
            if seen_novel:
                support_rows = data.standard_support_abs[
                    seed_position, :seen_novel
                ].reshape(-1)
                support_features = data.train_features[support_rows]
                support_labels = data.train_labels[support_rows]
                novel_normalized = torch.stack(
                    [
                        normalized_mean(
                            support_features[
                                support_labels == class_id
                            ]
                        )
                        for class_id in range(data.num_base, end_class)
                    ],
                    0,
                )
                novel_raw = torch.stack(
                    [
                        support_features[
                            support_labels == class_id
                        ].float().mean(0)
                        for class_id in range(data.num_base, end_class)
                    ],
                    0,
                )
            else:
                support_features = data.train_features.new_empty((0, dimension))
                support_labels = data.train_labels.new_empty((0,))
                novel_normalized = data.train_features.new_empty((0, dimension))
                novel_raw = data.train_features.new_empty((0, dimension))
            _synchronize(args.device)
            support_statistics_sec = time.time() - support_statistics_started
            for method in methods:
                if method in prototype_methods:
                    resources[method]["support_update_sec"] += (
                        support_statistics_sec
                    )

            if "hycal" in methods and seen_novel:
                # Append only the classes introduced in this session.  This
                # preserves the incremental information boundary while avoiding
                # repeated SVDs for previously seen supports.
                already_seen = len(hycal_novel_statistics)
                if already_seen < seen_novel:
                    new_map = {
                        class_id: support_features[support_labels == class_id]
                        for class_id in range(
                            data.num_base + already_seen, end_class
                        )
                    }
                    _synchronize(args.device)
                    head_started = time.time()
                    hycal_novel_statistics.extend(
                        hycal_state(new_map, sorted(new_map))
                    )
                    _synchronize(args.device)
                    resources["hycal"]["support_update_sec"] += (
                        time.time() - head_started
                    )
            normalized_centers = torch.cat([base_normalized, novel_normalized], 0)
            raw_centers = torch.cat([base_raw, novel_raw], 0)

            score_functions: Dict[str, Callable[[], torch.Tensor]] = {}
            if "cosine" in methods:
                score_functions["cosine"] = lambda q=queries, p=normalized_centers: cosine_scores_chunked(
                    q, p, args.query_chunk
                )
            if "euclidean" in methods:
                score_functions["euclidean"] = lambda q=queries, p=raw_centers: euclidean_scores_chunked(
                    q, p, args.query_chunk
                )
            if "covariance_lambda0" in methods:
                score_functions["covariance_lambda0"] = lambda q=queries, p=normalized_centers: metric_scores(
                    q, p, covariance, 0.0, RIDGE_GAMMA, args.query_chunk
                )
            if "shared_lda" in methods:
                score_functions["shared_lda"] = lambda q=queries, p=raw_centers: metric_scores(
                    q, p, covariance, 0.0, RIDGE_GAMMA, args.query_chunk
                )
            if "diagonal_lda" in methods:
                score_functions["diagonal_lda"] = lambda q=queries, p=raw_centers: diagonal_lda_scores_chunked(
                    q, p, base_variance, RIDGE_GAMMA, args.query_chunk
                )
            if "shrinksel" in methods:
                score_functions["shrinksel"] = lambda q=queries, p=normalized_centers: metric_scores(
                    q,
                    p,
                    covariance,
                    selected_lambda,
                    RIDGE_GAMMA,
                    args.query_chunk,
                )
            if "teen_selected" in methods:
                _synchronize(args.device)
                calibration_started = time.time()
                calibrated = teen_calibrate(
                    base_normalized,
                    novel_normalized,
                    float(teen_selected["shift_weight"]),
                    float(teen_selected["softmax_scale"]),
                )
                _synchronize(args.device)
                resources["teen_selected"]["support_update_sec"] += (
                    time.time() - calibration_started
                )
                teen_centers = torch.cat([base_normalized, calibrated], 0)
                score_functions["teen_selected"] = lambda q=queries, p=teen_centers: cosine_scores_chunked(
                    q, p, args.query_chunk
                )
            if "teen_official" in methods:
                official = baseline_parameters["official_defaults"]["teen"]
                _synchronize(args.device)
                calibration_started = time.time()
                calibrated = teen_calibrate(
                    base_normalized,
                    novel_normalized,
                    float(official["shift_weight"]),
                    float(official["softmax_scale"]),
                )
                _synchronize(args.device)
                resources["teen_official"]["support_update_sec"] += (
                    time.time() - calibration_started
                )
                official_centers = torch.cat([base_normalized, calibrated], 0)
                score_functions["teen_official"] = lambda q=queries, p=official_centers: cosine_scores_chunked(
                    q, p, args.query_chunk
                )
            if "ula_selected" in methods:
                score_functions["ula_selected"] = lambda q=queries, p=normalized_centers: ula_adjust_logits(
                    cosine_scores_chunked(q, p, args.query_chunk),
                    data.num_base,
                    ula_selected,
                )
            if "ula_official" in methods:
                score_functions["ula_official"] = lambda q=queries, p=normalized_centers: ula_adjust_logits(
                    cosine_scores_chunked(q, p, args.query_chunk),
                    data.num_base,
                    ula_official,
                )
            if "ranpac_no_projection" in methods:
                _synchronize(args.device)
                head_started = time.time()
                if seen_novel:
                    support_gram, support_cross = ranpac_normal_equations(
                        support_features,
                        support_labels,
                        data.num_total,
                    )
                    gram = ranpac_base_gram + support_gram
                    cross = ranpac_base_cross + support_cross
                else:
                    gram, cross = ranpac_base_gram, ranpac_base_cross
                ranpac_weights = solve_ranpac(
                    gram, cross[:, :end_class], ranpac_ridge
                )
                _synchronize(args.device)
                ranpac_resource_key = (
                    "support_update_sec" if seen_novel else "head_construction_sec"
                )
                resources["ranpac_no_projection"][ranpac_resource_key] += (
                    time.time() - head_started
                )
                score_functions["ranpac_no_projection"] = lambda q=queries, w=ranpac_weights: ranpac_scores_chunked(
                    q, w, None, args.query_chunk
                )
            if "fecam" in methods:
                if seen_novel:
                    novel_map = {
                        class_id: support_features[
                            support_labels == class_id
                        ]
                        for class_id in range(data.num_base, end_class)
                    }
                    _synchronize(args.device)
                    head_started = time.time()
                    novel_means, novel_inverse = fecam_state(
                        novel_map, range(data.num_base, end_class)
                    )
                    _synchronize(args.device)
                    resources["fecam"]["support_update_sec"] += (
                        time.time() - head_started
                    )
                    fecam_means = torch.cat(
                        [fecam_base_means, novel_means], 0
                    )
                    fecam_inverse = torch.cat(
                        [fecam_base_inverse, novel_inverse], 0
                    )
                else:
                    fecam_means, fecam_inverse = (
                        fecam_base_means,
                        fecam_base_inverse,
                    )
                score_functions["fecam"] = lambda q=queries, p=fecam_means, inv=fecam_inverse: class_specific_mahalanobis_scores_chunked(
                    q, p, inv, args.fecam_query_chunk
                )
            if "hycal" in methods:
                hycal_statistics = tuple(hycal_base_statistics) + tuple(
                    hycal_novel_statistics
                )
                score_functions["hycal"] = lambda q=queries, state=hycal_statistics: hycal_scores_chunked(
                    q, state, args.query_chunk
                )

            for method, function in score_functions.items():
                scores, prediction_sec, peak = _measure(
                    function,
                    args.device,
                    warmup=args.timing_warmup,
                    repeats=args.timing_repeats,
                )
                prediction = scores.argmax(1)
                row = _accuracy_row(
                    prediction, labels, data.num_base, end_class
                )
                row.update(
                    {
                        "session": session_index,
                        "end_class": end_class,
                        "num_queries": int(labels.shape[0]),
                        "prediction_sec": prediction_sec,
                        "peak_allocated_bytes": peak,
                    }
                )
                seed_rows[method].append(row)
                resources[method]["prediction_sec"] += prediction_sec
                resources[method]["peak_allocated_bytes"] = max(
                    int(resources[method]["peak_allocated_bytes"]), peak
                )
                if method in (
                    "cosine",
                    "euclidean",
                    "teen_selected",
                    "teen_official",
                    "ula_selected",
                    "ula_official",
                ):
                    state_bytes = end_class * dimension * 4
                elif method in (
                    "covariance_lambda0",
                    "shared_lda",
                    "diagonal_lda",
                    "shrinksel",
                ):
                    state_bytes = (dimension * dimension + end_class * dimension) * 4
                elif method == "fecam":
                    state_bytes = (
                        end_class * dimension * dimension
                        + end_class * dimension
                    ) * 4
                elif method == "hycal":
                    state_bytes = hycal_state_bytes(hycal_statistics)
                else:
                    state_bytes = (
                        dimension * dimension
                        + 2 * dimension * end_class
                    ) * 4
                resources[method]["max_state_bytes"] = max(
                    int(resources[method]["max_state_bytes"]), int(state_bytes)
                )
                del scores, prediction
        for method in methods:
            rows[method].append(
                {"support_seed": seed, "sessions": seed_rows[method]}
            )
        print(
            "dataset={} seed={} ".format(data.dataset, seed)
            + " ".join(
                "{}={:.3f}".format(
                    method,
                    mean(
                        float(item["overall"])
                        for item in seed_rows[method]
                    ),
                )
                for method in methods
            ),
            flush=True,
        )

    summary = _summary(rows, len(session_indices))
    comparisons = {}
    if "shrinksel" in summary:
        for competitor in (
            "cosine",
            "teen_selected",
            "teen_official",
            "ranpac_no_projection",
        ):
            if competitor in summary:
                paired = [
                    left - right
                    for left, right in zip(
                        summary["shrinksel"]["session_average_per_seed"],
                        summary[competitor]["session_average_per_seed"],
                    )
                ]
                comparisons["shrinksel_minus_" + competitor] = {
                    "mean": float(mean(paired)),
                    "std": float(pstdev(paired)),
                    "per_seed": paired,
                    "wins_mean": bool(mean(paired) > 0.0),
                }

    result = {
        "schema_version": 1,
        "stage": "strict_inductive_target_evaluation_after_parameter_lock",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "information_level": "I0/I1: support-only; no query-cohort statistics",
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_provenance": {
            "shrinksel": shrink_parameters["selection_result_sha256"],
            "baselines": baseline_parameters["selection_result_sha256"],
            "shrink_audit": shrink_audit.get("selection_result_sha256"),
            "baseline_audit": baseline_audit.get("audit_result_sha256"),
        },
        "parameters": {
            "shrinkage_lambda": selected_lambda,
            "shrinkage_ridge_gamma": RIDGE_GAMMA,
            "teen_selected": teen_selected,
            "ula_selected": selected["ula"],
            "ranpac_no_projection_ridge": ranpac_ridge,
            "fecam": {
                "official_vit_covariance_shrink_alpha1": 1.0,
                "alpha2": 0.0,
                "tukey": False,
            },
            "hycal": {
                "implementation": "equation-level reconstruction; no official code released",
                "alpha": 1.0,
                "beta": 0.0,
                "covariance_lambda": 1e-4,
                "covariance_gamma": 1.0,
                "text_features_used": False,
                "five_shot_reduction": (
                    "all class counts exceed alpha, so the Mahalanobis weight is one; "
                    "negative distance is argmax-equivalent to query-wise min-max similarity"
                ),
            },
        },
        "support_seeds": data.support_seeds[: args.seed_limit],
        "requested_methods": list(requested_methods),
        "methods": list(methods),
        "method_omissions": method_omissions,
        "per_seed": rows,
        "summary": summary,
        "paired_comparisons": comparisons,
        "resources": resources,
        "runtime_sec": time.time() - started,
        "timing_protocol": {
            "warmup": int(args.timing_warmup),
            "repeats": int(args.timing_repeats),
            "reported_statistic": "median synchronized wall time",
            "sessions": args.sessions,
            "resource_accounting": (
                "base-head construction, labeled-support session update, and "
                "query scoring are timed separately"
            ),
        },
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "summary": {
                    method: values["session_average_mean"]
                    for method, values in summary.items()
                },
                "paired_comparisons": comparisons,
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
