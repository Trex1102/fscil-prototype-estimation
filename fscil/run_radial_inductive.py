#!/usr/bin/env python3
"""Matched target evaluation for a globally locked radial-ShrinkSel candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence

import torch

from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import deployed_radial_choice, prototype_from_samples
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, normalized_mean, stable_json_sha256


METHODS = ("cosine", "shared_lda", "submitted_shrinksel", "radial_shrinksel")


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _accuracy(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    num_base: int,
    end_class: int,
) -> Dict[str, object]:
    correct = prediction == labels
    base_mask = labels < int(num_base)
    novel_mask = ~base_mask
    return {
        "overall": 100.0 * float(correct.float().mean()),
        "base": 100.0 * float(correct[base_mask].float().mean()),
        "novel": (
            100.0 * float(correct[novel_mask].float().mean())
            if bool(novel_mask.any())
            else None
        ),
        "per_class": [
            100.0 * float(correct[labels == class_id].float().mean())
            for class_id in range(end_class)
        ],
    }


def _summarize(rows: Mapping[str, Sequence[Mapping[str, object]]]) -> Dict[str, object]:
    result = {}
    for method, seed_rows in rows.items():
        averages = [
            float(mean(float(item["overall"]) for item in row["sessions"]))
            for row in seed_rows
        ]
        final_overall = [float(row["sessions"][-1]["overall"]) for row in seed_rows]
        final_base = [float(row["sessions"][-1]["base"]) for row in seed_rows]
        final_novel = [float(row["sessions"][-1]["novel"]) for row in seed_rows]
        result[method] = {
            "session_average_mean": float(mean(averages)),
            "session_average_std": float(pstdev(averages)),
            "session_average_per_seed": averages,
            "final_overall_mean": float(mean(final_overall)),
            "final_base_mean": float(mean(final_base)),
            "final_novel_mean": float(mean(final_novel)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parameter_group = parser.add_mutually_exclusive_group(required=True)
    parameter_group.add_argument("--radial-parameters", type=Path)
    parameter_group.add_argument("--global-parameters", type=Path)
    parser.add_argument("--radial-audit", type=Path)
    parser.add_argument("--submitted-shrink-parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=5)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--skip-file-hash", action="store_true")
    parser.add_argument(
        "--allow-heldout-transfer",
        action="store_true",
        help="apply the globally locked configuration to an unseen feature representation",
    )
    args = parser.parse_args()

    parameters = _load(args.radial_parameters or args.global_parameters)
    submitted = _load(args.submitted_shrink_parameters)
    audit = None
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if parameters["stage"] == "base_selection_radial_shrinksel":
        if args.radial_audit is None:
            raise RuntimeError("per-dataset radial parameters require their audit")
        audit = _load(args.radial_audit)
        if audit["stage"] != "untouched_base_audit_radial_shrinksel":
            raise RuntimeError("invalid radial audit stage")
        if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
            raise RuntimeError("radial selection/audit provenance mismatch")
        if parameters["dataset"] != data.dataset:
            raise RuntimeError("radial parameter dataset mismatch")
        source_matched = (
            parameters["source_feature_file_sha256"]
            == data.selection_source_sha256
        )
        if not source_matched:
            raise RuntimeError("radial selector/target feature identity mismatch")
        parameter_transfer = "dataset_and_representation_matched_base_only_selection"
        parameter_lock_sha256 = str(parameters["selection_result_sha256"])
    elif parameters["stage"] == "global_radial_configuration_lock_before_target_evaluation":
        expected_feature = parameters["source_feature_file_sha256"].get(data.dataset)
        source_matched = data.selection_source_sha256 == expected_feature
        if not source_matched and not args.allow_heldout_transfer:
            raise RuntimeError(
                "global selector/target feature identity mismatch; use the explicit "
                "heldout-transfer flag only for an unchanged configuration"
            )
        parameter_transfer = (
            "source_matched_global_configuration"
            if source_matched
            else "unchanged_global_configuration_on_heldout_representation"
        )
        parameter_lock_sha256 = str(parameters["global_selection_sha256"])
    else:
        raise RuntimeError("radial parameters are not in a locked selection stage")
    if submitted["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("submitted selector/target feature identity mismatch")
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid seed limit")

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    system = covariance_eigensystem(by_class, base_ids)
    if audit is not None:
        mode, radial_lambda = deployed_radial_choice(parameters, audit)
    else:
        global_passed = bool(parameters["passes_untouched_base_audit"])
        mode = str(
            parameters.get(
                "deployed_prototype_mode",
                parameters["selected_prototype_mode"] if global_passed else "raw",
            )
        )
        radial_lambda = float(
            parameters.get(
                "deployed_lambda",
                parameters["selected_lambda"] if global_passed else 0.0,
            )
        )
    submitted_lambda = float(submitted["selected_lambda"])
    base_centers = {
        "cosine": torch.stack([normalized_mean(by_class[class_id]) for class_id in base_ids]),
        "shared_lda": torch.stack([by_class[class_id].float().mean(0) for class_id in base_ids]),
        "submitted_shrinksel": torch.stack(
            [normalized_mean(by_class[class_id]) for class_id in base_ids]
        ),
        "radial_shrinksel": torch.stack(
            [prototype_from_samples(by_class[class_id], mode) for class_id in base_ids]
        ),
    }
    rows: Dict[str, List[Dict[str, object]]] = {method: [] for method in METHODS}
    num_sessions = (data.num_total - data.num_base) // data.increment + 1
    for seed_position in range(args.seed_limit):
        seed_rows = {method: [] for method in METHODS}
        for session_index in range(num_sessions):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            query_mask = data.test_labels < end_class
            queries = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            novel_samples = {}
            if seen_novel:
                support_rows = data.standard_support_abs[
                    seed_position, :seen_novel
                ].reshape(-1)
                support_features = data.train_features[support_rows]
                support_labels = data.train_labels[support_rows]
                novel_samples = {
                    class_id: support_features[support_labels == class_id]
                    for class_id in range(data.num_base, end_class)
                }
            centers = {
                "cosine": torch.cat(
                    [
                        base_centers["cosine"],
                        torch.stack(
                            [normalized_mean(novel_samples[class_id]) for class_id in sorted(novel_samples)]
                        )
                        if novel_samples
                        else base_centers["cosine"].new_empty((0, base_centers["cosine"].shape[1])),
                    ]
                ),
                "shared_lda": torch.cat(
                    [
                        base_centers["shared_lda"],
                        torch.stack(
                            [novel_samples[class_id].float().mean(0) for class_id in sorted(novel_samples)]
                        )
                        if novel_samples
                        else base_centers["shared_lda"].new_empty((0, base_centers["shared_lda"].shape[1])),
                    ]
                ),
                "submitted_shrinksel": torch.cat(
                    [
                        base_centers["submitted_shrinksel"],
                        torch.stack(
                            [normalized_mean(novel_samples[class_id]) for class_id in sorted(novel_samples)]
                        )
                        if novel_samples
                        else base_centers["submitted_shrinksel"].new_empty((0, base_centers["submitted_shrinksel"].shape[1])),
                    ]
                ),
                "radial_shrinksel": torch.cat(
                    [
                        base_centers["radial_shrinksel"],
                        torch.stack(
                            [prototype_from_samples(novel_samples[class_id], mode) for class_id in sorted(novel_samples)]
                        )
                        if novel_samples
                        else base_centers["radial_shrinksel"].new_empty((0, base_centers["radial_shrinksel"].shape[1])),
                    ]
                ),
            }
            lambdas = {
                "cosine": 1.0,
                "shared_lda": 0.0,
                "submitted_shrinksel": submitted_lambda,
                "radial_shrinksel": radial_lambda,
            }
            for method in METHODS:
                prediction = metric_scores(
                    queries,
                    centers[method],
                    system,
                    lambdas[method],
                    RIDGE_GAMMA,
                    args.query_chunk,
                ).argmax(1)
                row = _accuracy(prediction, labels, data.num_base, end_class)
                row["session_index"] = session_index
                seed_rows[method].append(row)
        for method in METHODS:
            rows[method].append(
                {
                    "support_seed": int(data.support_seeds[seed_position]),
                    "sessions": seed_rows[method],
                }
            )
        print(
            "dataset={} seed={} ".format(data.dataset, data.support_seeds[seed_position])
            + " ".join(
                "{}={:.3f}".format(
                    method,
                    mean(float(item["overall"]) for item in seed_rows[method]),
                )
                for method in METHODS
            ),
            flush=True,
        )
    summary = _summarize(rows)
    radial = summary["radial_shrinksel"]["session_average_per_seed"]
    comparisons = {}
    for comparator in ("shared_lda", "submitted_shrinksel", "cosine"):
        reference = summary[comparator]["session_average_per_seed"]
        differences = [left - right for left, right in zip(radial, reference)]
        comparisons["radial_minus_" + comparator] = {
            "mean": float(mean(differences)),
            "std": float(pstdev(differences)),
            "per_seed": differences,
        }
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "matched_radial_shrinksel_target_evaluation_after_parameter_lock",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "methods": list(METHODS),
        "selected_prototype_mode": mode,
        "selected_lambda": radial_lambda,
        "fallback_deployed": bool(
            (audit is not None and not audit["passes_non_regression_gate"])
            or (
                audit is None
                and not parameters["passes_untouched_base_audit"]
            )
        ),
        "parameter_lock_sha256": parameter_lock_sha256,
        "parameter_transfer": parameter_transfer,
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "support_seeds": data.support_seeds[: args.seed_limit],
        "target_labels_role": "evaluation only; never used in parameter selection",
        "summary": summary,
        "comparisons": comparisons,
        "per_seed": [
            {
                "support_seed": int(data.support_seeds[index]),
                "methods": {method: rows[method][index] for method in METHODS},
            }
            for index in range(args.seed_limit)
        ],
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
