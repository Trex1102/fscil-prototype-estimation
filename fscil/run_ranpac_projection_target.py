#!/usr/bin/env python3
"""Evaluate the audited 10k-ReLU RanPAC head on locked FSCIL target sessions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping

import torch
import torch.nn.functional as F

from .baseline_methods import (
    make_ranpac_projection,
    ranpac_low_rank_inverse_apply,
    ranpac_low_rank_system,
    ranpac_normal_equations,
    ranpac_normal_equation_relative_residual,
    ranpac_transform,
    solve_ranpac_from_low_rank_system,
)
from .ranpac_projection import PROJECTION_SEED
from .target_data import load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, stable_json_sha256, tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--feature-chunk", type=int, default=1024)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _project_chunked(
    features: torch.Tensor, projection: torch.Tensor, chunk: int
) -> torch.Tensor:
    rows = []
    for start in range(0, features.shape[0], int(chunk)):
        rows.append(
            ranpac_transform(features[start : start + int(chunk)], projection)
        )
    return torch.cat(rows, 0)


def _scores_chunked(
    projected_queries: torch.Tensor, weights: torch.Tensor, chunk: int
) -> torch.Tensor:
    rows = []
    for start in range(0, projected_queries.shape[0], int(chunk)):
        rows.append(
            projected_queries[start : start + int(chunk)]
            .to(dtype=weights.dtype)
            .matmul(weights)
        )
    return torch.cat(rows, 0)


def _accuracy(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    num_base: int,
    end_class: int,
) -> Dict[str, object]:
    correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    per_class = []
    for class_id in range(int(end_class)):
        mask = labels == class_id
        per_class.append(100.0 * float(correct[mask].float().mean().item()))
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "base": 100.0 * float(correct[base].float().mean().item()),
        "novel": (
            100.0 * float(correct[novel].float().mean().item())
            if bool(novel.any())
            else None
        ),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    parameters = _load_json(args.parameters)
    audit = _load_json(args.audit)
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid --seed-limit")
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("projected RanPAC parameter dataset mismatch")
    if parameters["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("projected RanPAC feature provenance mismatch")
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("projected RanPAC selection/audit provenance mismatch")
    if not bool(audit["passes_sanity_gate"]):
        raise RuntimeError("projected RanPAC failed its untouched base audit")

    projection_dimension = int(parameters["projection_dimension"])
    ridge = float(parameters["selected_ridge"])
    projection = make_ranpac_projection(
        data.train_features.shape[1],
        projection_dimension,
        PROJECTION_SEED,
        data.train_features.device,
    )
    projection_hash = tensor_sha256(projection)
    expected_projection_hash = parameters["evaluation"]["projection_sha256"]
    if projection_hash != expected_projection_hash:
        raise RuntimeError("projected RanPAC random matrix hash mismatch")

    _sync(args.device)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    construction_started = time.time()
    base_mask = data.train_labels < data.num_base
    base_labels = data.train_labels[base_mask]
    projected_base = _project_chunked(
        data.train_features[base_mask], projection, args.feature_chunk
    )
    low_rank_system = None
    factor = None
    if int(projected_base.shape[0]) < int(projected_base.shape[1]):
        low_rank_system = ranpac_low_rank_system(
            projected_base, base_labels, data.num_total
        )
        base_weights = solve_ranpac_from_low_rank_system(low_rank_system, ridge)
        linear_solver = "float64_sample_space_spectral"
        residual_threshold = 1e-8
        numerical_rank = int(
            low_rank_system.squared_singular_values.shape[0]
        )
    else:
        base_gram, base_cross = ranpac_normal_equations(
            data.train_features[base_mask],
            base_labels,
            data.num_total,
            projection=projection,
            feature_chunk=args.feature_chunk,
        )
        base_gram = 0.5 * (base_gram + base_gram.t())
        identity = torch.eye(
            projection_dimension, dtype=base_gram.dtype, device=base_gram.device
        )
        factor = torch.linalg.cholesky(base_gram + ridge * identity)
        base_weights = torch.cholesky_solve(base_cross, factor)
        linear_solver = "float32_primal_cholesky"
        residual_threshold = 5e-3
        numerical_rank = None
        del base_gram, base_cross, identity
    normal_equation_residual = ranpac_normal_equation_relative_residual(
        projected_base,
        base_labels,
        base_weights,
        ridge,
        data.num_total,
    )
    if (
        not torch.isfinite(torch.tensor(normal_equation_residual))
        or normal_equation_residual > residual_threshold
    ):
        raise RuntimeError(
            "projected RanPAC numerical residual {:.6g} exceeds {:.6g}".format(
                normal_equation_residual, residual_threshold
            )
        )
    del projected_base
    projected_test = _project_chunked(
        data.test_features, projection, args.feature_chunk
    )
    _sync(args.device)
    base_construction_sec = time.time() - construction_started

    num_sessions = 1 + (data.num_total - data.num_base) // data.increment
    seed_rows: List[Dict[str, object]] = []
    total_support_update_sec = 0.0
    total_prediction_sec = 0.0
    peak_allocated = (
        int(torch.cuda.max_memory_allocated())
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    for seed_position in range(args.seed_limit):
        support_h_parts: List[torch.Tensor] = []
        support_z_parts: List[torch.Tensor] = []
        support_label_parts: List[torch.Tensor] = []
        sessions = []
        for session_index in range(num_sessions):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            if session_index > 0:
                _sync(args.device)
                update_started = time.time()
            if session_index > 0:
                class_start = data.num_base + (session_index - 1) * data.increment
                class_stop = min(class_start + data.increment, data.num_total)
                relative_start = class_start - data.num_base
                relative_stop = class_stop - data.num_base
                support_rows = data.standard_support_abs[
                    seed_position, relative_start:relative_stop
                ].reshape(-1)
                new_features = data.train_features[support_rows]
                new_labels = data.train_labels[support_rows]
                new_h = ranpac_transform(new_features, projection)
                if low_rank_system is not None:
                    new_h = new_h.to(dtype=base_weights.dtype)
                    new_z = ranpac_low_rank_inverse_apply(
                        low_rank_system, new_h.t(), ridge
                    )
                else:
                    if factor is None:  # pragma: no cover - guarded by branch
                        raise RuntimeError("missing RanPAC primal factor")
                    new_z = torch.cholesky_solve(new_h.t(), factor)
                support_h_parts.append(new_h)
                support_z_parts.append(new_z)
                support_label_parts.append(new_labels)

            if support_h_parts:
                support_h = torch.cat(support_h_parts, 0)
                support_z = torch.cat(support_z_parts, 1)
                support_labels = torch.cat(support_label_parts, 0)
                small_system = torch.eye(
                    support_h.shape[0],
                    dtype=support_h.dtype,
                    device=support_h.device,
                ) + support_h.matmul(support_z)
                targets = F.one_hot(
                    support_labels.long(), num_classes=data.num_total
                ).to(dtype=support_h.dtype)
                residual = targets - support_h.matmul(base_weights)
                correction = torch.linalg.solve(small_system, residual)
                weights = base_weights + support_z.matmul(correction)
            else:
                weights = base_weights
            if session_index > 0:
                _sync(args.device)
                total_support_update_sec += time.time() - update_started

            query_mask = data.test_labels < end_class
            query_h = projected_test[query_mask]
            labels = data.test_labels[query_mask]
            _sync(args.device)
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            prediction_started = time.time()
            scores = _scores_chunked(
                query_h, weights[:, :end_class], args.query_chunk
            )
            prediction = scores.argmax(1)
            _sync(args.device)
            prediction_sec = time.time() - prediction_started
            total_prediction_sec += prediction_sec
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                peak_allocated = max(
                    peak_allocated, int(torch.cuda.max_memory_allocated())
                )
            row = _accuracy(
                prediction, labels, data.num_base, end_class
            )
            row.update(
                {
                    "session": session_index,
                    "end_class": end_class,
                    "prediction_sec": prediction_sec,
                }
            )
            sessions.append(row)
            del scores, prediction
        session_average = float(mean(float(row["overall"]) for row in sessions))
        seed_rows.append(
            {
                "support_seed": data.support_seeds[seed_position],
                "session_average": session_average,
                "sessions": sessions,
            }
        )
        print(
            "dataset={} RanPAC-M{} seed={} session_average={:.3f}".format(
                data.dataset,
                projection_dimension,
                data.support_seeds[seed_position],
                session_average,
            ),
            flush=True,
        )

    session_averages = [float(row["session_average"]) for row in seed_rows]
    per_session = []
    for session_index in range(num_sessions):
        values = [
            float(row["sessions"][session_index]["overall"]) for row in seed_rows
        ]
        per_session.append(
            {"mean": float(mean(values)), "std": float(pstdev(values))}
        )
    final_values = [float(row["sessions"][-1]["overall"]) for row in seed_rows]
    projection_bytes = int(projection.numel() * projection.element_size())
    maximum_support_count = int(data.num_total - data.num_base) * 5
    if low_rank_system is not None:
        spectral_bytes = sum(
            int(value.numel() * value.element_size())
            for value in (
                low_rank_system.squared_singular_values,
                low_rank_system.right_vectors,
                low_rank_system.rotated_targets,
                base_weights,
            )
        )
        support_bytes = (
            2
            * maximum_support_count
            * projection_dimension
            * base_weights.element_size()
        )
        state_bytes = projection_bytes + spectral_bytes + support_bytes
    else:
        state_bytes = (
            data.train_features.shape[1] * projection_dimension
            + projection_dimension * projection_dimension
            + 2 * projection_dimension * data.num_total
        ) * 4
    official_streaming_state_bytes = (
        data.train_features.shape[1] * projection_dimension
        + projection_dimension * projection_dimension
        + projection_dimension * data.num_total
    ) * 4
    result = {
        "schema_version": 2,
        "stage": "strict_inductive_projected_ranpac_target_evaluation",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "information_level": "I0: labeled supports; no query-cohort statistics",
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "audit_result_sha256": audit["audit_result_sha256"],
        "projection_dimension": projection_dimension,
        "projection_seed": PROJECTION_SEED,
        "projection_sha256": projection_hash,
        "ridge": ridge,
        "linear_solver": linear_solver,
        "numerical_diagnostics": {
            "normal_equation_relative_residual": normal_equation_residual,
            "residual_threshold": residual_threshold,
            "passes_residual_gate": True,
            "numerical_rank": numerical_rank,
            "solve_dtype": str(base_weights.dtype),
        },
        "support_seeds": data.support_seeds[: args.seed_limit],
        "per_seed": seed_rows,
        "summary": {
            "session_average_mean": float(mean(session_averages)),
            "session_average_std": float(pstdev(session_averages)),
            "session_average_per_seed": session_averages,
            "final_accuracy_mean": float(mean(final_values)),
            "final_accuracy_std": float(pstdev(final_values)),
            "per_session": per_session,
        },
        "resources": {
            "base_head_construction_sec": base_construction_sec,
            "support_update_sec": total_support_update_sec,
            "prediction_sec": total_prediction_sec,
            "peak_allocated_bytes": peak_allocated,
            "state_bytes": int(state_bytes),
            "solver_state_bytes": int(state_bytes),
            "official_streaming_state_bytes": int(
                official_streaming_state_bytes
            ),
        },
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "projection_dimension": projection_dimension,
                "ridge": ridge,
                "linear_solver": linear_solver,
                "normal_equation_relative_residual": normal_equation_residual,
                "session_average_mean": result["summary"]["session_average_mean"],
                "final_accuracy_mean": result["summary"]["final_accuracy_mean"],
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
