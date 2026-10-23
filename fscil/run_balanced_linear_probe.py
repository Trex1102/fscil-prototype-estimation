#!/usr/bin/env python3
"""Class-balanced joint linear-probe control on frozen FSCIL features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import deployed_radial_choice, prototype_from_samples
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, stable_json_sha256


STEPS = 300
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
METHODS = ("radial_shrinksel", "lp_joint_class_balanced")


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _train_balanced_linear(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Full-batch logistic regression with equal total loss weight per class."""
    counts = torch.bincount(labels, minlength=int(num_classes)).float()
    if bool((counts <= 0).any()):
        raise RuntimeError("balanced linear probe received an empty seen class")
    class_weight = counts.reciprocal()
    class_weight = class_weight / class_weight.mean()
    weight = torch.zeros(
        int(num_classes), features.shape[1], device=features.device, dtype=torch.float32
    ).requires_grad_(True)
    bias = torch.zeros(
        int(num_classes), device=features.device, dtype=torch.float32
    ).requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = features @ weight.t() + bias
        loss = F.cross_entropy(logits, labels, weight=class_weight)
        loss.backward()
        optimizer.step()
    return weight.detach(), bias.detach(), [float(value) for value in counts.tolist()]


def _accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_base: int
) -> Dict[str, Optional[float]]:
    correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    per_class = [
        float(correct[labels == class_id].float().mean().item())
        for class_id in torch.unique(labels).tolist()
    ]
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "class_balanced": 100.0 * float(mean(per_class)),
        "base": 100.0 * float(correct[base].float().mean().item()),
        "novel": (
            100.0 * float(correct[novel].float().mean().item())
            if bool(novel.any())
            else None
        ),
    }


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
    if int(args.seed_limit) != 20:
        raise ValueError("the locked balanced-probe control requires 20 support draws")
    parameters = _load(args.radial_parameters)
    audit = _load(args.radial_audit)
    if parameters.get("dataset") != data.dataset or audit.get("dataset") != data.dataset:
        raise RuntimeError("balanced-probe parameter dataset mismatch")
    if parameters.get("source_feature_file_sha256") != data.selection_source_sha256:
        raise RuntimeError("balanced-probe selector/feature identity mismatch")
    mode, shrinkage = deployed_radial_choice(parameters, audit)

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    covariance = covariance_eigensystem(by_class, base_ids)
    base_centers = torch.stack(
        [prototype_from_samples(by_class[class_id], mode) for class_id in base_ids]
    )
    base_features = data.train_features[data.train_labels < data.num_base]
    base_labels = data.train_labels[data.train_labels < data.num_base]
    session_count = 1 + (data.num_total - data.num_base) // data.increment
    per_support: List[Dict[str, object]] = []

    for support_position, support_seed in enumerate(data.support_seeds):
        sessions: List[Dict[str, object]] = []
        for session_index in range(session_count):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            query_mask = data.test_labels < end_class
            query = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            novel_samples: Dict[int, torch.Tensor] = {}
            if seen_novel:
                rows = data.standard_support_abs[
                    support_position, :seen_novel
                ].reshape(-1)
                support_features = data.train_features[rows]
                support_labels = data.train_labels[rows]
                novel_samples = {
                    class_id: support_features[support_labels == class_id]
                    for class_id in range(data.num_base, end_class)
                }

            radial_centers = torch.cat(
                [base_centers]
                + (
                    [
                        torch.stack(
                            [
                                prototype_from_samples(novel_samples[class_id], mode)
                                for class_id in sorted(novel_samples)
                            ]
                        )
                    ]
                    if novel_samples
                    else []
                ),
                0,
            )
            radial_prediction = metric_scores(
                query,
                radial_centers,
                covariance,
                float(shrinkage),
                RIDGE_GAMMA,
                int(args.query_chunk),
            ).argmax(1)

            if novel_samples:
                fit_features = torch.cat(
                    [base_features]
                    + [novel_samples[class_id] for class_id in sorted(novel_samples)],
                    0,
                )
                fit_labels = torch.cat(
                    [base_labels]
                    + [
                        torch.full(
                            (novel_samples[class_id].shape[0],),
                            class_id,
                            dtype=torch.long,
                            device=base_labels.device,
                        )
                        for class_id in sorted(novel_samples)
                    ],
                    0,
                )
            else:
                fit_features, fit_labels = base_features, base_labels
            linear_weight, linear_bias, class_counts = _train_balanced_linear(
                fit_features, fit_labels, end_class
            )
            linear_prediction = (
                query @ linear_weight.t() + linear_bias
            ).argmax(1)
            sessions.append(
                {
                    "session_index": int(session_index),
                    "fit_class_counts": class_counts,
                    "methods": {
                        "radial_shrinksel": _accuracy(
                            radial_prediction, labels, data.num_base
                        ),
                        "lp_joint_class_balanced": _accuracy(
                            linear_prediction, labels, data.num_base
                        ),
                    },
                }
            )
        per_support.append(
            {"support_seed": int(support_seed), "sessions": sessions}
        )
        print("dataset={} support={} done".format(data.dataset, support_seed), flush=True)

    summary = {}
    for method in METHODS:
        session_average = [
            float(mean(float(row["methods"][method]["overall"]) for row in seed["sessions"]))
            for seed in per_support
        ]
        final = {
            metric: [
                float(seed["sessions"][-1]["methods"][method][metric])
                for seed in per_support
            ]
            for metric in ("overall", "class_balanced", "base", "novel")
        }
        summary[method] = {
            "session_average_mean": float(mean(session_average)),
            "session_average_std": float(pstdev(session_average)),
            "session_average_per_support": session_average,
            "final": {
                metric: {
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)),
                    "per_support": values,
                }
                for metric, values in final.items()
            },
        }
    difference = [
        float(left) - float(right)
        for left, right in zip(
            summary["radial_shrinksel"]["session_average_per_support"],
            summary["lp_joint_class_balanced"]["session_average_per_support"],
        )
    ]
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "class_balanced_joint_linear_probe_control",
        "dataset": data.dataset,
        "backbone": dict(data.backbone),
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "audit_result_sha256": audit["audit_result_sha256"],
        "support_seeds": list(data.support_seeds),
        "methods": list(METHODS),
        "optimization": {
            "objective": "inverse-frequency class-balanced cross-entropy",
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "initialization": "zero weight and bias",
            "full_batch": True,
        },
        "information_budget": {
            "complete_base_training_features_revisited_each_session": True,
            "accumulated_novel_supports_used": True,
            "query_features_used_for_fitting": False,
            "query_labels_used_for_fitting": False,
            "class_names_or_text_used": False,
        },
        "summary": summary,
        "radial_minus_balanced_probe": {
            "mean": float(mean(difference)),
            "std": float(pstdev(difference)),
            "per_support": difference,
        },
        "per_support": per_support,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
