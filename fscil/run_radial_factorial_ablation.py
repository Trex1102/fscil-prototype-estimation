#!/usr/bin/env python3
"""Target-facing factorial diagnostics for the locked radial ShrinkSel head.

The complete grid and oracle are diagnostics only. They never alter the base-selected
configuration supplied through ``--radial-parameters``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Mapping, Sequence, Tuple

import torch

from .constants import LAMBDA_GRID, RIDGE_GAMMA
from .geometry import (
    covariance_eigensystem,
    metric_scores,
    metric_scores_rotated,
    rotate_to_covariance_basis,
)
from .radial_selection import PROTOTYPE_MODES, prototype_from_samples
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, stable_json_sha256


RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_base: int
) -> Dict[str, float | None]:
    correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "base": 100.0 * float(correct[base].float().mean().item()),
        "novel": (
            100.0 * float(correct[novel].float().mean().item())
            if bool(novel.any())
            else None
        ),
    }


def _summarize(values: Sequence[float]) -> Dict[str, object]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "per_seed": [float(value) for value in values],
    }


def component_arms(
    selected_mode: str, selected_lambda: float
) -> Dict[str, Tuple[str, float]]:
    return {
        "cosine": ("normalized", 1.0),
        "shared_lda": ("raw", 0.0),
        "geometry_only": ("normalized", float(selected_lambda)),
        "radius_only": (str(selected_mode), 1.0),
        "selected_radius_full_covariance": (str(selected_mode), 0.0),
        "shrinksel": (str(selected_mode), float(selected_lambda)),
    }


def _candidate_key(mode: str, shrinkage: float) -> str:
    return "{}|{}".format(mode, float(shrinkage))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--radial-parameters", required=True, type=Path)
    parser.add_argument("--radial-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--query-chunk", type=int, default=4096)
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
        raise ValueError("invalid --seed-limit")
    parameters = _load(args.radial_parameters)
    audit = _load(args.radial_audit)
    if parameters.get("stage") != "base_selection_radial_shrinksel":
        raise RuntimeError("factorial requires per-setting radial selection")
    if audit.get("stage") != "untouched_base_audit_radial_shrinksel":
        raise RuntimeError("invalid radial audit stage")
    if parameters["dataset"] != data.dataset or audit["dataset"] != data.dataset:
        raise RuntimeError("radial factorial dataset mismatch")
    if parameters["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("radial factorial feature mismatch")
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("radial selection/audit provenance mismatch")
    if not bool(audit["passes_non_regression_gate"]):
        raise RuntimeError("radial selection failed its untouched audit")

    selected_mode = str(parameters["selected_prototype_mode"])
    selected_lambda = float(parameters["selected_lambda"])
    if selected_mode not in PROTOTYPE_MODES or selected_lambda not in LAMBDA_GRID:
        raise RuntimeError("selected radial configuration is outside the locked grid")
    arms = component_arms(selected_mode, selected_lambda)
    candidates = [
        (mode, float(shrinkage))
        for mode in PROTOTYPE_MODES
        for shrinkage in LAMBDA_GRID
    ]

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    base_system = covariance_eigensystem(by_class, base_ids)
    base_centers = {
        mode: torch.stack(
            [prototype_from_samples(by_class[class_id], mode) for class_id in base_ids]
        )
        for mode in PROTOTYPE_MODES
    }
    number_sessions = 1 + (data.num_total - data.num_base) // data.increment
    component_per_seed: Dict[str, list[float]] = {name: [] for name in arms}
    component_sessions: Dict[str, list[list[float]]] = {name: [] for name in arms}
    factorial: Dict[str, list[float]] = {
        _candidate_key(mode, shrinkage): [] for mode, shrinkage in candidates
    }
    ridge: Dict[str, list[float]] = {str(value): [] for value in RIDGE_GRID}
    covariance_source: Dict[str, list[float]] = {
        "base_only": [],
        "base_plus_novel_supports": [],
    }
    per_seed = []

    for seed_position in range(args.seed_limit):
        session_values: Dict[str, list[float]] = {name: [] for name in arms}
        final_payload: Dict[str, object] = {}
        for session_index in range(number_sessions):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            query_mask = data.test_labels < end_class
            queries = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            novel_samples: Dict[int, torch.Tensor] = {}
            if seen_novel:
                rows = data.standard_support_abs[
                    seed_position, :seen_novel
                ].reshape(-1)
                features = data.train_features[rows]
                support_labels = data.train_labels[rows]
                novel_samples = {
                    class_id: features[support_labels == class_id]
                    for class_id in range(data.num_base, end_class)
                }
            centers = {}
            for mode in PROTOTYPE_MODES:
                novel = (
                    torch.stack(
                        [
                            prototype_from_samples(novel_samples[class_id], mode)
                            for class_id in range(data.num_base, end_class)
                        ]
                    )
                    if seen_novel
                    else base_centers[mode].new_empty(
                        (0, base_centers[mode].shape[1])
                    )
                )
                centers[mode] = torch.cat([base_centers[mode], novel], 0)

            rotated_queries = rotate_to_covariance_basis(queries, base_system)
            rotated_centers = {
                mode: rotate_to_covariance_basis(values, base_system)
                for mode, values in centers.items()
            }
            score_cache: Dict[Tuple[str, float, float], torch.Tensor] = {}

            def predict(
                mode: str, shrinkage: float, ridge_gamma: float = RIDGE_GAMMA
            ) -> torch.Tensor:
                key = (str(mode), float(shrinkage), float(ridge_gamma))
                if key not in score_cache:
                    score_cache[key] = metric_scores_rotated(
                        rotated_queries,
                        rotated_centers[mode],
                        base_system,
                        shrinkage,
                        ridge_gamma,
                        args.query_chunk,
                    )
                return score_cache[key].argmax(1)

            arm_metrics = {}
            for name, (mode, shrinkage) in arms.items():
                prediction = predict(mode, shrinkage)
                metrics = _group_accuracy(prediction, labels, data.num_base)
                arm_metrics[name] = metrics
                session_values[name].append(float(metrics["overall"]))

            if session_index == number_sessions - 1:
                factorial_metrics = {}
                for mode, shrinkage in candidates:
                    key = _candidate_key(mode, shrinkage)
                    prediction = predict(mode, shrinkage)
                    metrics = _group_accuracy(prediction, labels, data.num_base)
                    factorial[key].append(float(metrics["overall"]))
                    factorial_metrics[key] = metrics
                ridge_metrics = {}
                for ridge_value in RIDGE_GRID:
                    prediction = predict(
                        selected_mode, selected_lambda, ridge_value
                    )
                    metrics = _group_accuracy(prediction, labels, data.num_base)
                    ridge[str(ridge_value)].append(float(metrics["overall"]))
                    ridge_metrics[str(ridge_value)] = metrics
                covariance_source["base_only"].append(
                    float(arm_metrics["shrinksel"]["overall"])
                )
                updated_pool = {
                    class_id: by_class[class_id] for class_id in base_ids
                }
                updated_pool.update(novel_samples)
                updated_system = covariance_eigensystem(
                    updated_pool, list(range(end_class))
                )
                updated_prediction = metric_scores(
                    queries,
                    centers[selected_mode],
                    updated_system,
                    selected_lambda,
                    RIDGE_GAMMA,
                    args.query_chunk,
                ).argmax(1)
                updated_metrics = _group_accuracy(
                    updated_prediction, labels, data.num_base
                )
                covariance_source["base_plus_novel_supports"].append(
                    float(updated_metrics["overall"])
                )
                final_payload = {
                    "factorial": factorial_metrics,
                    "ridge": ridge_metrics,
                    "covariance_source": {
                        "base_only": arm_metrics["shrinksel"],
                        "base_plus_novel_supports": updated_metrics,
                        "base_only_dof": int(base_system.dof),
                        "base_plus_novel_supports_dof": int(updated_system.dof),
                    },
                }

        for name, values in session_values.items():
            component_sessions[name].append(values)
            component_per_seed[name].append(float(mean(values)))
        per_seed.append(
            {
                "support_seed": int(data.support_seeds[seed_position]),
                "component_session_overall": session_values,
                "final": final_payload,
            }
        )
        print(
            "dataset={} seed={} {}".format(
                data.dataset,
                data.support_seeds[seed_position],
                " ".join(
                    "{}={:.3f}".format(name, component_per_seed[name][-1])
                    for name in arms
                ),
            ),
            flush=True,
        )

    factorial_summary = {key: _summarize(values) for key, values in factorial.items()}
    oracle_key = max(
        factorial_summary,
        key=lambda key: float(factorial_summary[key]["mean"]),
    )
    selected_key = _candidate_key(selected_mode, selected_lambda)
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "post_lock_radial_factorial_ablation",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "audit_result_sha256": audit["audit_result_sha256"],
        "selected": {
            "prototype_mode": selected_mode,
            "lambda": selected_lambda,
            "candidate_key": selected_key,
        },
        "information_boundary": {
            "target_labels_used_for_metrics_only": True,
            "target_queries_used_for_selection": False,
            "oracle_is_diagnostic_only": True,
            "support_shots": 5,
        },
        "component_arms": {
            name: {"prototype_mode": mode, "lambda": shrinkage}
            for name, (mode, shrinkage) in arms.items()
        },
        "component_session_average": {
            name: _summarize(values) for name, values in component_per_seed.items()
        },
        "component_session_curves": component_sessions,
        "factorial_final": factorial_summary,
        "target_oracle": {
            "candidate_key": oracle_key,
            "mean": factorial_summary[oracle_key]["mean"],
            "selected_mean": factorial_summary[selected_key]["mean"],
            "oracle_minus_selected": float(
                factorial_summary[oracle_key]["mean"]
                - factorial_summary[selected_key]["mean"]
            ),
        },
        "ridge_final": {key: _summarize(values) for key, values in ridge.items()},
        "covariance_source_final": {
            key: _summarize(values) for key, values in covariance_source.items()
        },
        "support_seeds": data.support_seeds[: args.seed_limit],
        "per_seed": per_seed,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "selected": result["selected"],
                "target_oracle": result["target_oracle"],
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
