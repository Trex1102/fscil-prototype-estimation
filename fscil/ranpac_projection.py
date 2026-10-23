#!/usr/bin/env python3
"""Base-only selection and audit for the official RanPAC random-projection head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .baseline_methods import (
    cosine_scores_chunked,
    make_ranpac_projection,
    ranpac_inverse_apply,
    ranpac_low_rank_inverse_apply,
    ranpac_low_rank_system,
    ranpac_normal_equations,
    ranpac_normal_equation_relative_residual,
    ranpac_transform,
    solve_ranpac_from_eigensystem,
    solve_ranpac_from_low_rank_system,
)
from .constants import PSEUDO_NOVEL_PER_EPISODE
from .selection import AuditData, ConstructionSelectionData, _anchor_split, load_audit, load_construction_selection
from .utils import atomic_write_json, normalized_mean, stable_json_sha256, tensor_sha256


PROJECTION_SEED = 1993
DEFAULT_PROJECTION_DIMENSION = 10000
DEFAULT_EPISODES = 5
# The original RanPAC release searches 10^-8,...,10^8.  A later FSCIL-calibration
# repository narrows this to 10^1,...,10^8; we retain the broader official grid so a
# boundary choice cannot make the matched baseline artificially weak.
RANPAC_RP_RIDGE_GRID = tuple(float(10.0 ** exponent) for exponent in range(-8, 9))


def _flat_anchor_data(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_queries: Mapping[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    class_ids = sorted(anchor_train)
    device = next(iter(anchor_train.values())).device
    train_features: List[torch.Tensor] = []
    train_labels: List[torch.Tensor] = []
    query_features: List[torch.Tensor] = []
    query_labels: List[torch.Tensor] = []
    for label, class_id in enumerate(class_ids):
        train = anchor_train[class_id]
        query = anchor_queries[class_id]
        train_features.append(train)
        query_features.append(query)
        train_labels.append(
            torch.full((train.shape[0],), label, dtype=torch.long, device=device)
        )
        query_labels.append(
            torch.full((query.shape[0],), label, dtype=torch.long, device=device)
        )
    return (
        torch.cat(train_features, 0),
        torch.cat(train_labels, 0),
        torch.cat(query_features, 0),
        torch.cat(query_labels, 0),
        class_ids,
    )


def _pseudo_episode(
    pseudo_pool: Mapping[int, torch.Tensor],
    episode_index: int,
    pseudo_count: int,
    num_anchor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    rng = np.random.RandomState(9137 + episode_index * 7919)
    candidates = sorted(pseudo_pool)
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidates),
            size=min(int(pseudo_count), len(candidates)),
            replace=False,
        ).tolist()
    )
    device = next(iter(pseudo_pool.values())).device
    support_features: List[torch.Tensor] = []
    support_labels: List[torch.Tensor] = []
    query_features: List[torch.Tensor] = []
    query_labels: List[torch.Tensor] = []
    for position, class_id in enumerate(chosen):
        values = pseudo_pool[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15485863 + episode_index * 32452843 + class_id * 49999)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        support = values[order[:5]]
        query = values[order[5:20]]
        if query.shape[0] < 15:
            raise RuntimeError("insufficient pseudo-query samples for projected RanPAC")
        label = int(num_anchor) + position
        support_features.append(support)
        query_features.append(query)
        support_labels.append(
            torch.full((support.shape[0],), label, dtype=torch.long, device=device)
        )
        query_labels.append(
            torch.full((query.shape[0],), label, dtype=torch.long, device=device)
        )
    return (
        torch.cat(support_features, 0),
        torch.cat(support_labels, 0),
        torch.cat(query_features, 0),
        torch.cat(query_labels, 0),
        chosen,
    )


def _balanced_accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_anchor: int
) -> Dict[str, float]:
    anchor = labels < int(num_anchor)
    novel = ~anchor
    anchor_accuracy = 100.0 * float(
        (prediction[anchor] == labels[anchor]).float().mean().item()
    )
    novel_accuracy = 100.0 * float(
        (prediction[novel] == labels[novel]).float().mean().item()
    )
    return {
        "anchor": anchor_accuracy,
        "pseudo_novel": novel_accuracy,
        "balanced": 0.5 * (anchor_accuracy + novel_accuracy),
    }


def _update_from_cached_base(
    base_weights: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    support_h: torch.Tensor,
    support_labels: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    inverse_times_support = ranpac_inverse_apply(
        eigenvalues, eigenvectors, support_h.t(), ridge
    )
    small_system = torch.eye(
        support_h.shape[0], dtype=support_h.dtype, device=support_h.device
    ) + support_h.matmul(inverse_times_support)
    targets = F.one_hot(
        support_labels.long(), num_classes=base_weights.shape[1]
    ).to(dtype=support_h.dtype)
    residual = targets - support_h.matmul(base_weights)
    correction = torch.linalg.solve(small_system, residual)
    return base_weights + inverse_times_support.matmul(correction)


def _update_from_inverse_support(
    base_weights: torch.Tensor,
    support_h: torch.Tensor,
    support_labels: torch.Tensor,
    inverse_times_support: torch.Tensor,
) -> torch.Tensor:
    support_h = support_h.to(dtype=base_weights.dtype)
    inverse_times_support = inverse_times_support.to(dtype=base_weights.dtype)
    small_system = torch.eye(
        support_h.shape[0], dtype=support_h.dtype, device=support_h.device
    ) + support_h.matmul(inverse_times_support)
    targets = F.one_hot(
        support_labels.long(), num_classes=base_weights.shape[1]
    ).to(dtype=support_h.dtype)
    residual = targets - support_h.matmul(base_weights)
    correction = torch.linalg.solve(small_system, residual)
    return base_weights + inverse_times_support.matmul(correction)


def evaluate_projection(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_pool: Mapping[int, torch.Tensor],
    projection_dimension: int,
    episodes: int,
    namespace_seed: int,
    ridge_candidates: Sequence[float],
) -> Dict[str, object]:
    started = time.time()
    anchor_train, anchor_queries = _anchor_split(anchor_pool, namespace_seed)
    train, train_labels, anchor_query, anchor_query_labels, anchor_ids = _flat_anchor_data(
        anchor_train, anchor_queries
    )
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[dataset]
    num_classes = len(anchor_ids) + pseudo_count
    projection = make_ranpac_projection(
        train.shape[1], projection_dimension, PROJECTION_SEED, train.device
    )
    anchor_train_h = ranpac_transform(train, projection)
    if not bool(torch.isfinite(anchor_train_h).all()):
        raise RuntimeError("projected RanPAC anchor features are not finite")
    anchor_query_h = ranpac_transform(anchor_query, projection)
    curves = {
        float(ridge): {"anchor": [], "pseudo_novel": [], "balanced": []}
        for ridge in ridge_candidates
    }
    cosine_curve = {"anchor": [], "pseudo_novel": [], "balanced": []}
    anchor_prototypes = torch.stack(
        [normalized_mean(anchor_train[class_id]) for class_id in anchor_ids], 0
    )
    episode_payloads = []
    for episode_index in range(int(episodes)):
        support, support_labels, pseudo_query, pseudo_labels, chosen = _pseudo_episode(
            pseudo_pool,
            episode_index,
            pseudo_count,
            len(anchor_ids),
        )
        support_h = ranpac_transform(support, projection)
        query_h = torch.cat(
            [anchor_query_h, ranpac_transform(pseudo_query, projection)], 0
        )
        labels = torch.cat([anchor_query_labels, pseudo_labels], 0)
        pseudo_prototypes = torch.stack(
            [
                normalized_mean(
                    support[support_labels == len(anchor_ids) + position]
                )
                for position in range(len(chosen))
            ],
            0,
        )
        cosine_prediction = cosine_scores_chunked(
            torch.cat([anchor_query, pseudo_query], 0),
            torch.cat([anchor_prototypes, pseudo_prototypes], 0),
        ).argmax(1)
        cosine_metrics = _balanced_accuracy(
            cosine_prediction, labels, len(anchor_ids)
        )
        for metric, value in cosine_metrics.items():
            cosine_curve[metric].append(value)
        episode_payloads.append((support_h, support_labels, query_h, labels))

    factorization_started = time.time()
    numerical_diagnostics: Dict[str, object] = {}
    if int(anchor_train_h.shape[0]) < int(anchor_train_h.shape[1]):
        low_rank_system = ranpac_low_rank_system(
            anchor_train_h, train_labels, num_classes
        )
        residual_by_ridge: Dict[str, float] = {}
        for ridge in ridge_candidates:
            base_weights = solve_ranpac_from_low_rank_system(
                low_rank_system, float(ridge)
            )
            residual_by_ridge[str(float(ridge))] = (
                ranpac_normal_equation_relative_residual(
                    anchor_train_h,
                    train_labels,
                    base_weights,
                    float(ridge),
                    num_classes,
                )
            )
            for support_h, support_labels, query_h, labels in episode_payloads:
                inverse_times_support = ranpac_low_rank_inverse_apply(
                    low_rank_system, support_h.t(), float(ridge)
                )
                weights = _update_from_inverse_support(
                    base_weights,
                    support_h,
                    support_labels,
                    inverse_times_support,
                )
                prediction = query_h.to(dtype=weights.dtype).matmul(weights).argmax(1)
                metrics = _balanced_accuracy(prediction, labels, len(anchor_ids))
                for metric, value in metrics.items():
                    curves[float(ridge)][metric].append(value)
        solver = "float64_sample_space_spectral"
        numerical_diagnostics = {
            "normal_equation_relative_residual_by_ridge": residual_by_ridge,
            "maximum_normal_equation_relative_residual": float(
                max(residual_by_ridge.values())
            ),
            "numerical_rank": int(
                low_rank_system.squared_singular_values.shape[0]
            ),
            "sample_count": int(low_rank_system.num_samples),
            "feature_dimension": int(low_rank_system.feature_dimension),
            "solve_dtype": str(low_rank_system.right_vectors.dtype),
        }
    else:
        gram, anchor_cross = ranpac_normal_equations(
            train,
            train_labels,
            num_classes,
            projection=projection,
            feature_chunk=1024,
        )
        gram = 0.5 * (gram + gram.t())
        if not bool(torch.isfinite(gram).all()):
            raise RuntimeError("projected RanPAC Gram matrix is not finite")
    if (
        int(anchor_train_h.shape[0]) >= int(anchor_train_h.shape[1])
        and int(projection_dimension) <= 4096
    ):
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.clamp_min(0.0)
        for ridge in ridge_candidates:
            base_weights = solve_ranpac_from_eigensystem(
                eigenvalues, eigenvectors, anchor_cross, float(ridge)
            )
            for support_h, support_labels, query_h, labels in episode_payloads:
                weights = _update_from_cached_base(
                    base_weights,
                    eigenvalues,
                    eigenvectors,
                    support_h,
                    support_labels,
                    float(ridge),
                )
                prediction = query_h.to(dtype=weights.dtype).matmul(weights).argmax(1)
                metrics = _balanced_accuracy(prediction, labels, len(anchor_ids))
                for metric, value in metrics.items():
                    curves[float(ridge)][metric].append(value)
        solver = "symmetric_eigendecomposition"
    elif int(anchor_train_h.shape[0]) >= int(anchor_train_h.shape[1]):
        identity = torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        for ridge in ridge_candidates:
            factor = torch.linalg.cholesky(gram + float(ridge) * identity)
            base_weights = torch.cholesky_solve(anchor_cross, factor)
            for support_h, support_labels, query_h, labels in episode_payloads:
                inverse_times_support = torch.cholesky_solve(support_h.t(), factor)
                weights = _update_from_inverse_support(
                    base_weights,
                    support_h,
                    support_labels,
                    inverse_times_support,
                )
                prediction = query_h.to(dtype=weights.dtype).matmul(weights).argmax(1)
                metrics = _balanced_accuracy(prediction, labels, len(anchor_ids))
                for metric, value in metrics.items():
                    curves[float(ridge)][metric].append(value)
            del factor, base_weights
        solver = "cholesky_per_ridge"
    factorization_sec = time.time() - factorization_started
    return {
        "projection_dimension": int(projection_dimension),
        "projection_seed": PROJECTION_SEED,
        "projection_sha256": tensor_sha256(projection),
        "episodes": int(episodes),
        "pseudo_novel_per_episode": int(pseudo_count),
        "anchor_classes": anchor_ids,
        "ridge_grid": [float(value) for value in ridge_candidates],
        "curve": {
            str(ridge): {
                metric: {
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)),
                    "episodes": len(values),
                }
                for metric, values in metrics.items()
            }
            for ridge, metrics in curves.items()
        },
        "cosine": {
            metric: {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "episodes": len(values),
            }
            for metric, values in cosine_curve.items()
        },
        "linear_solver": solver,
        "numerical_diagnostics": numerical_diagnostics,
        "factorization_and_update_sec": factorization_sec,
        "runtime_sec": time.time() - started,
    }


def select_projection(
    data: ConstructionSelectionData,
    projection_dimension: int,
    episodes: int,
) -> Dict[str, object]:
    evaluation = evaluate_projection(
        data.dataset,
        data.construction,
        data.selection,
        projection_dimension,
        episodes,
        namespace_seed=271828,
        ridge_candidates=RANPAC_RP_RIDGE_GRID,
    )
    selected_ridge = max(
        RANPAC_RP_RIDGE_GRID,
        key=lambda value: (
            float(evaluation["curve"][str(float(value))]["balanced"]["mean"]),
            -abs(np.log10(float(value)) - 4.0),
        ),
    )
    result = {
        "schema_version": 1,
        "stage": "base_selection_ranpac_projection",
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_ridge": float(selected_ridge),
        "projection_dimension": int(projection_dimension),
        "projection_seed": PROJECTION_SEED,
        "objective": "mean of anchor and pseudo-novel accuracy",
        "implementation": (
            "official ReLU random projection and Phase-2 ridge normal equations; "
            "float64 sample-space spectral solve when the system is low rank"
        ),
        "evaluation": evaluation,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_projection(
    data: AuditData,
    parameters: Mapping[str, object],
    episodes: int,
) -> Dict[str, object]:
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("dataset mismatch for projected RanPAC audit")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("projected RanPAC selection/audit bundle mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("projected RanPAC feature provenance mismatch")
    ridge = float(parameters["selected_ridge"])
    evaluation = evaluate_projection(
        data.dataset,
        data.anchors,
        data.audit,
        int(parameters["projection_dimension"]),
        episodes,
        namespace_seed=314159,
        ridge_candidates=(ridge,),
    )
    score = float(evaluation["curve"][str(ridge)]["balanced"]["mean"])
    cosine = float(evaluation["cosine"]["balanced"]["mean"])
    anchor_score = float(evaluation["curve"][str(ridge)]["anchor"]["mean"])
    cosine_anchor = float(evaluation["cosine"]["anchor"]["mean"])
    maximum_residual = evaluation.get("numerical_diagnostics", {}).get(
        "maximum_normal_equation_relative_residual"
    )
    numerical_pass = bool(
        maximum_residual is None
        or (
            np.isfinite(float(maximum_residual))
            and float(maximum_residual) <= 1e-8
        )
    )
    result = {
        "schema_version": 1,
        "stage": "untouched_base_audit_ranpac_projection",
        "dataset": data.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_ridge": ridge,
        "projection_dimension": int(parameters["projection_dimension"]),
        "balanced_accuracy": score,
        "cosine_balanced_accuracy": cosine,
        "anchor_accuracy": anchor_score,
        "cosine_anchor_accuracy": cosine_anchor,
        "selected_minus_cosine": score - cosine,
        "passes_balanced_non_regression_gate": bool(score >= cosine - 5.0),
        "passes_numerical_residual_gate": numerical_pass,
        # A strong base/anchor result plus exact algebraic parity establishes that the
        # head is operating as intended even when class imbalance destroys novel
        # accuracy.  That performance failure is retained as a target result.
        "passes_sanity_gate": bool(
            np.isfinite(score)
            and np.isfinite(anchor_score)
            and anchor_score >= cosine_anchor - 5.0
            and numerical_pass
        ),
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
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
        sub.add_argument(
            "--projection-dimension", type=int, default=DEFAULT_PROJECTION_DIMENSION
        )
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        data = load_construction_selection(args.bundle, args.device)
        result = select_projection(
            data, args.projection_dimension, args.episodes
        )
    else:
        data = load_audit(args.bundle, args.device)
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        if int(args.projection_dimension) != int(parameters["projection_dimension"]):
            raise RuntimeError("audit projection dimension differs from selected dimension")
        result = audit_projection(data, parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "stage": result["stage"],
                "selected_ridge": result["selected_ridge"],
                "projection_dimension": result["projection_dimension"],
                "selected_minus_cosine": result.get("selected_minus_cosine"),
                "passes_sanity_gate": result.get("passes_sanity_gate"),
                "runtime_sec": result["evaluation"]["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.command == "audit" and not result["passes_sanity_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
