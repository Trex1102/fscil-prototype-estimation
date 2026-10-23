#!/usr/bin/env python3
"""Post-lock shrinkage sensitivity and finite-shot noise analysis.

Target labels are used only to evaluate a predeclared grid after the selector has
been frozen and audited.  The target-oracle row is diagnostic and is never fed
back into the method or a headline ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Mapping, Sequence

import torch

from .constants import LAMBDA_GRID, RIDGE_GAMMA
from .geometry import (
    covariance_eigensystem,
    inverse_eigenvalues,
    metric_scores_rotated,
    metric_trace_penalty,
    rotate_to_covariance_basis,
)
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, normalized_mean, stable_json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--shrink-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--shots", default="1,5,20")
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_base: int
) -> Dict[str, float]:
    correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "base": 100.0 * float(correct[base].float().mean().item()),
        "novel": 100.0 * float(correct[novel].float().mean().item()),
    }


def summarize(rows: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for metric in ("overall", "base", "novel"):
        values = [float(row[metric]) for row in rows]
        result[metric + "_mean"] = float(mean(values))
        result[metric + "_std"] = float(pstdev(values))
        result[metric + "_per_seed"] = values
    return result


def metric_prototype_error(
    support_mean: torch.Tensor,
    full_mean: torch.Tensor,
    eigenvectors: torch.Tensor,
    inverse: torch.Tensor,
) -> float:
    difference = (support_mean - full_mean).float().matmul(eigenvectors)
    return float((difference.square() * inverse[None, :]).sum(1).mean().item())


def main() -> None:
    args = parse_args()
    started = time.time()
    shots = tuple(int(value.strip()) for value in args.shots.split(",") if value.strip())
    if not shots or any(value not in (1, 2, 5, 10, 20) for value in shots):
        raise ValueError("--shots must be a subset of 1,2,5,10,20")
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid --seed-limit")
    parameters = load_json(args.shrink_parameters)
    audit = load_json(args.shrink_audit)
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("shrinkage parameter dataset mismatch")
    if parameters["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("shrinkage parameter feature hash mismatch")
    if not bool(audit["passes_non_regression_gate"]):
        raise RuntimeError("selected shrinkage failed its untouched base audit")
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("shrinkage selection/audit provenance mismatch")
    selected_lambda = float(parameters["selected_lambda"])
    if selected_lambda not in LAMBDA_GRID:
        raise RuntimeError("selected shrinkage is outside the committed grid")

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    novel_ids = list(range(data.num_base, data.num_total))
    system = covariance_eigensystem(by_class, base_ids)
    queries_rotated = rotate_to_covariance_basis(data.test_features, system)
    base_prototypes = torch.stack(
        [normalized_mean(by_class[class_id]) for class_id in base_ids], 0
    )
    base_rotated = rotate_to_covariance_basis(base_prototypes, system)
    base_raw = torch.stack(
        [by_class[class_id].float().mean(0) for class_id in base_ids], 0
    )
    base_raw_rotated = rotate_to_covariance_basis(base_raw, system)
    full_raw = torch.stack(
        [by_class[class_id].float().mean(0) for class_id in novel_ids], 0
    )
    full_normalized = torch.nn.functional.normalize(full_raw, dim=1)

    theoretical: Dict[str, object] = {}
    accuracy: Dict[str, object] = {}
    raw_center_accuracy: Dict[str, object] = {}
    empirical_error: Dict[str, object] = {}
    for shot in shots:
        theoretical[str(shot)] = {
            str(value): metric_trace_penalty(
                system, value, RIDGE_GAMMA, shot
            )
            for value in LAMBDA_GRID
        }
        per_lambda = {str(value): [] for value in LAMBDA_GRID}
        raw_per_lambda = (
            {str(value): [] for value in LAMBDA_GRID} if shot == 5 else None
        )
        raw_error = {str(value): [] for value in LAMBDA_GRID}
        normalized_error = {str(value): [] for value in LAMBDA_GRID}
        for seed_position in range(args.seed_limit):
            support_rows = data.nested_support_abs[seed_position, :, :shot]
            support_raw = torch.stack(
                [
                    data.train_features[support_rows[class_position]].float().mean(0)
                    for class_position in range(len(novel_ids))
                ],
                0,
            )
            support_normalized = torch.nn.functional.normalize(support_raw, dim=1)
            prototypes_rotated = torch.cat(
                [
                    base_rotated,
                    rotate_to_covariance_basis(support_normalized, system),
                ],
                0,
            )
            raw_prototypes_rotated = (
                torch.cat(
                    [
                        base_raw_rotated,
                        rotate_to_covariance_basis(support_raw, system),
                    ],
                    0,
                )
                if shot == 5
                else None
            )
            for value in LAMBDA_GRID:
                key = str(value)
                inverse = inverse_eigenvalues(system, value, RIDGE_GAMMA)
                prediction = metric_scores_rotated(
                    queries_rotated,
                    prototypes_rotated,
                    system,
                    value,
                    RIDGE_GAMMA,
                    args.query_chunk,
                ).argmax(1)
                row = group_accuracy(prediction, data.test_labels, data.num_base)
                row["support_seed"] = int(data.support_seeds[seed_position])
                per_lambda[key].append(row)
                if raw_per_lambda is not None:
                    raw_prediction = metric_scores_rotated(
                        queries_rotated,
                        raw_prototypes_rotated,
                        system,
                        value,
                        RIDGE_GAMMA,
                        args.query_chunk,
                    ).argmax(1)
                    raw_row = group_accuracy(
                        raw_prediction, data.test_labels, data.num_base
                    )
                    raw_row["support_seed"] = int(
                        data.support_seeds[seed_position]
                    )
                    raw_per_lambda[key].append(raw_row)
                raw_error[key].append(
                    metric_prototype_error(
                        support_raw,
                        full_raw,
                        system.eigenvectors,
                        inverse,
                    )
                )
                normalized_error[key].append(
                    metric_prototype_error(
                        support_normalized,
                        full_normalized,
                        system.eigenvectors,
                        inverse,
                    )
                )
        accuracy[str(shot)] = {
            key: summarize(rows) for key, rows in per_lambda.items()
        }
        if raw_per_lambda is not None:
            raw_center_accuracy[str(shot)] = {
                key: summarize(rows) for key, rows in raw_per_lambda.items()
            }
        empirical_error[str(shot)] = {
            key: {
                "raw_mean_metric_square_mean": float(mean(raw_error[key])),
                "raw_mean_metric_square_std": float(pstdev(raw_error[key])),
                "normalized_prototype_metric_square_mean": float(
                    mean(normalized_error[key])
                ),
                "normalized_prototype_metric_square_std": float(
                    pstdev(normalized_error[key])
                ),
                "raw_per_seed": raw_error[key],
                "normalized_per_seed": normalized_error[key],
            }
            for key in per_lambda
        }
        print(
            "dataset={} shot={} ".format(data.dataset, shot)
            + " ".join(
                "lambda{}={:.3f}".format(
                    value,
                    accuracy[str(shot)][str(value)]["overall_mean"],
                )
                for value in LAMBDA_GRID
            ),
            flush=True,
        )

    primary_shot = "5" if "5" in accuracy else str(shots[0])
    target_curve = {
        value: float(row["overall_mean"])
        for value, row in accuracy[primary_shot].items()
    }
    oracle_lambda = max(target_curve, key=target_curve.get)
    raw_curve = {
        value: float(row["overall_mean"])
        for value, row in raw_center_accuracy.get(primary_shot, {}).items()
    }
    raw_oracle_lambda = max(raw_curve, key=raw_curve.get) if raw_curve else None
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "post_lock_shrinkage_sensitivity_and_noise_analysis",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "selected_lambda_from_base_only_data": selected_lambda,
        "ridge_gamma": RIDGE_GAMMA,
        "lambda_grid_committed_before_target_evaluation": list(LAMBDA_GRID),
        "shots": list(shots),
        "support_seeds": data.support_seeds[: args.seed_limit],
        "target_labels_role": (
            "post-lock evaluation only; target-oracle lambda is a labelled diagnostic "
            "and is never used by the method"
        ),
        "theoretical_trace_penalty": theoretical,
        "empirical_novel_prototype_error": empirical_error,
        "final_session_accuracy_sensitivity": accuracy,
        "raw_center_final_session_accuracy_sensitivity": raw_center_accuracy,
        "prototype_normalization_ablation_role": (
            "post-lock diagnostic only; the deployed selector was calibrated with "
            "normalized prototypes and is not changed using this target curve"
        ),
        "primary_shot_diagnostic": {
            "shot": int(primary_shot),
            "selected_lambda": selected_lambda,
            "selected_accuracy": target_curve[str(selected_lambda)],
            "target_oracle_lambda_diagnostic_only": float(oracle_lambda),
            "target_oracle_accuracy_diagnostic_only": target_curve[oracle_lambda],
            "selected_regret_to_target_oracle": (
                target_curve[oracle_lambda] - target_curve[str(selected_lambda)]
            ),
            "raw_center_at_selected_lambda_diagnostic_only": (
                raw_curve.get(str(selected_lambda)) if raw_curve else None
            ),
            "raw_center_target_oracle_lambda_diagnostic_only": (
                float(raw_oracle_lambda) if raw_oracle_lambda is not None else None
            ),
            "raw_center_target_oracle_accuracy_diagnostic_only": (
                raw_curve.get(raw_oracle_lambda)
                if raw_oracle_lambda is not None
                else None
            ),
        },
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result["primary_shot_diagnostic"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
