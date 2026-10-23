#!/usr/bin/env python3
"""Evaluate ShrinkSel under deterministic permutations of novel arrival order.

The published base/novel identity split, cached features, support examples, and
base-only selected head remain fixed.  Only the order in which the already
designated novel identities enter the classifier is changed.  This isolates
novel-arrival order without silently changing the base training distribution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import deployed_radial_choice, prototype_from_samples
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, stable_json_sha256


ORDER_SEEDS = (1993, 2027, 4099, 7919, 104729)
METHODS = ("shared_lda", "radial_shrinksel")


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deterministic_orders(num_base: int, num_total: int) -> Dict[str, List[int]]:
    novel = np.arange(int(num_base), int(num_total), dtype=np.int64)
    result = {"published": [int(value) for value in novel.tolist()]}
    for seed in ORDER_SEEDS:
        order = np.random.RandomState(seed).permutation(novel)
        result["perm{}".format(seed)] = [int(value) for value in order.tolist()]
    if len({tuple(values) for values in result.values()}) != len(result):
        raise RuntimeError("novel-order generator produced duplicate permutations")
    return result


def _accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_base: int
) -> Dict[str, Optional[float]]:
    correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    per_class = []
    for class_id in torch.unique(labels).tolist():
        mask = labels == int(class_id)
        per_class.append(float(correct[mask].float().mean().item()))
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


def _two_way_interval(matrix: np.ndarray, seed: int, draws: int) -> List[float]:
    if matrix.shape != (len(ORDER_SEEDS), 20):
        raise RuntimeError("two-way order/support bootstrap requires a 5x20 matrix")
    generator = np.random.RandomState(seed)
    samples = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        order_index = generator.randint(0, matrix.shape[0], size=matrix.shape[0])
        support_index = generator.randint(0, matrix.shape[1], size=matrix.shape[1])
        samples[index] = matrix[np.ix_(order_index, support_index)].mean()
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


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
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
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
        raise ValueError("the locked novel-order study requires all 20 support draws")
    parameters = _load(args.radial_parameters)
    audit = _load(args.radial_audit)
    if parameters.get("dataset") != data.dataset or audit.get("dataset") != data.dataset:
        raise RuntimeError("novel-order parameter dataset mismatch")
    if parameters.get("source_feature_file_sha256") != data.selection_source_sha256:
        raise RuntimeError("novel-order selector/feature identity mismatch")
    if audit.get("selection_result_sha256") != parameters.get("selection_result_sha256"):
        raise RuntimeError("novel-order selection/audit provenance mismatch")
    mode, shrinkage = deployed_radial_choice(parameters, audit)

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    covariance = covariance_eigensystem(by_class, base_ids)
    base_centers = {
        "shared_lda": torch.stack([by_class[c].float().mean(0) for c in base_ids]),
        "radial_shrinksel": torch.stack(
            [prototype_from_samples(by_class[c], mode) for c in base_ids]
        ),
    }
    method_lambda = {"shared_lda": 0.0, "radial_shrinksel": float(shrinkage)}
    orders = deterministic_orders(data.num_base, data.num_total)
    per_order: List[Dict[str, object]] = []
    session_count = 1 + (data.num_total - data.num_base) // data.increment

    for order_name, novel_order in orders.items():
        support_rows: List[Dict[str, object]] = []
        for support_position, support_seed in enumerate(data.support_seeds):
            sessions: List[Dict[str, object]] = []
            for session_index in range(session_count):
                seen_novel = novel_order[: session_index * data.increment]
                seen_ids = base_ids + seen_novel
                query_mask = torch.zeros_like(data.test_labels, dtype=torch.bool)
                for class_id in seen_ids:
                    query_mask |= data.test_labels == int(class_id)
                query = data.test_features[query_mask]
                labels = data.test_labels[query_mask]
                center_rows: Dict[str, List[torch.Tensor]] = {
                    method: [base_centers[method]] for method in METHODS
                }
                if seen_novel:
                    samples = {
                        class_id: data.train_features[
                            data.standard_support_abs[
                                support_position, class_id - data.num_base
                            ]
                        ]
                        for class_id in seen_novel
                    }
                    center_rows["shared_lda"].append(
                        torch.stack([samples[c].float().mean(0) for c in seen_novel])
                    )
                    center_rows["radial_shrinksel"].append(
                        torch.stack([prototype_from_samples(samples[c], mode) for c in seen_novel])
                    )
                session_methods = {}
                identity = torch.as_tensor(seen_ids, device=query.device, dtype=torch.long)
                for method in METHODS:
                    centers = torch.cat(center_rows[method], 0)
                    local_prediction = metric_scores(
                        query,
                        centers,
                        covariance,
                        method_lambda[method],
                        RIDGE_GAMMA,
                        int(args.query_chunk),
                    ).argmax(1)
                    prediction = identity[local_prediction]
                    session_methods[method] = _accuracy(
                        prediction, labels, data.num_base
                    )
                sessions.append(
                    {
                        "session_index": int(session_index),
                        "seen_novel_identity_ids": list(seen_novel),
                        "methods": session_methods,
                    }
                )
            support_rows.append(
                {
                    "support_seed": int(support_seed),
                    "sessions": sessions,
                }
            )
        summary = {}
        for method in METHODS:
            values = [
                float(mean(float(row["methods"][method]["overall"]) for row in seed["sessions"]))
                for seed in support_rows
            ]
            summary[method] = {
                "session_average_mean": float(mean(values)),
                "session_average_std": float(pstdev(values)),
                "session_average_per_support": values,
            }
        differences = [
            float(left) - float(right)
            for left, right in zip(
                summary["radial_shrinksel"]["session_average_per_support"],
                summary["shared_lda"]["session_average_per_support"],
            )
        ]
        per_order.append(
            {
                "order_name": order_name,
                "novel_identity_order": novel_order,
                "summary": summary,
                "radial_minus_shared_lda": {
                    "mean": float(mean(differences)),
                    "std": float(pstdev(differences)),
                    "per_support": differences,
                },
                "per_support": support_rows,
            }
        )
        print(
            "dataset={} order={} delta={:+.4f}".format(
                data.dataset, order_name, float(mean(differences))
            ),
            flush=True,
        )

    randomized = [row for row in per_order if row["order_name"] != "published"]
    matrix = np.asarray(
        [row["radial_minus_shared_lda"]["per_support"] for row in randomized],
        dtype=np.float64,
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "fixed_base_split_novel_arrival_order_robustness",
        "dataset": data.dataset,
        "backbone": dict(data.backbone),
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "audit_result_sha256": audit["audit_result_sha256"],
        "selected_prototype_mode": mode,
        "selected_lambda": float(shrinkage),
        "base_identity_split_fixed": True,
        "novel_arrival_order_only": True,
        "order_seeds": list(ORDER_SEEDS),
        "support_seeds": list(data.support_seeds),
        "per_order": per_order,
        "randomized_order_summary": {
            "orders": len(randomized),
            "support_draws_per_order": 20,
            "mean_delta": float(matrix.mean()),
            "minimum_order_mean_delta": float(
                min(row["radial_minus_shared_lda"]["mean"] for row in randomized)
            ),
            "maximum_order_mean_delta": float(
                max(row["radial_minus_shared_lda"]["mean"] for row in randomized)
            ),
            "positive_order_count": int(
                sum(row["radial_minus_shared_lda"]["mean"] > 0.0 for row in randomized)
            ),
            "two_way_bootstrap_95": _two_way_interval(
                matrix, 20260825 + sum(ord(value) for value in data.dataset), int(args.bootstrap_draws)
            ),
        },
        "inference_scope": (
            "conditional on the fixed base/novel split, encoder, manifests, query set, "
            "five deterministic novel permutations, and twenty support draws"
        ),
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
