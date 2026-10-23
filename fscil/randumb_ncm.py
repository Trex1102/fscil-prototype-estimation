#!/usr/bin/env python3
"""RanDumb random-Fourier representation with its official NCM option.

The released RanDumb code offers an RBF sampler followed by either SLDA or cosine
nearest-centroid classification.  This module adapts the NCM branch to FSCIL without
ever fitting the kernel scale or a classifier hyperparameter on target queries.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .constants import PSEUDO_NOVEL_PER_EPISODE
from .ranpac_projection import _flat_anchor_data, _pseudo_episode
from .selection import _anchor_split, load_audit, load_construction_selection
from .target_data import load_target_evaluation_data
from .utils import (
    atomic_write_json,
    backbone_display,
    normalized_mean,
    stable_json_sha256,
    tensor_sha256,
)


RANDUMB_SEED = 1993
DEFAULT_DIMENSION = 10000


def make_rbf_projection(
    input_dimension: int,
    output_dimension: int,
    gamma: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the same Gaussian frequencies/uniform phases as RBFSampler."""
    rng = np.random.RandomState(int(seed))
    weights = (
        math.sqrt(2.0 * float(gamma))
        * rng.normal(size=(int(input_dimension), int(output_dimension)))
    ).astype(np.float32)
    offset = rng.uniform(0.0, 2.0 * math.pi, size=int(output_dimension)).astype(
        np.float32
    )
    return torch.from_numpy(weights).to(device), torch.from_numpy(offset).to(device)


def rbf_transform(
    features: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    scale = math.sqrt(2.0 / float(weights.shape[1]))
    return scale * torch.cos(features.float().matmul(weights) + offset[None, :])


def rbf_transform_chunked(
    features: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, features.shape[0], int(chunk)):
        rows.append(
            rbf_transform(features[start : start + int(chunk)], weights, offset)
        )
    return torch.cat(rows, 0)


def gamma_scale(features: torch.Tensor) -> float:
    variance = features.float().var(unbiased=False).clamp_min(1e-12)
    return 1.0 / (float(features.shape[1]) * float(variance.item()))


def projected_centroids(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int],
    weights: torch.Tensor,
    offset: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    label_to_position = {int(class_id): position for position, class_id in enumerate(class_ids)}
    sums = torch.zeros(
        (len(class_ids), weights.shape[1]), dtype=torch.float32, device=features.device
    )
    counts = torch.zeros(len(class_ids), dtype=torch.float32, device=features.device)
    for start in range(0, features.shape[0], int(chunk)):
        stop = min(start + int(chunk), features.shape[0])
        transformed = rbf_transform(features[start:stop], weights, offset)
        positions = torch.as_tensor(
            [label_to_position[int(value)] for value in labels[start:stop].tolist()],
            dtype=torch.long,
            device=features.device,
        )
        sums.index_add_(0, positions, transformed)
        counts.index_add_(0, positions, torch.ones_like(positions, dtype=torch.float32))
    return F.normalize(sums / counts[:, None].clamp_min(1.0), dim=-1)


def cosine_prediction(
    projected_queries: torch.Tensor,
    projected_prototypes: torch.Tensor,
    query_chunk: int,
) -> torch.Tensor:
    queries = F.normalize(projected_queries.float(), dim=-1)
    prototypes = F.normalize(projected_prototypes.float(), dim=-1)
    rows = []
    for start in range(0, queries.shape[0], int(query_chunk)):
        rows.append(queries[start : start + int(query_chunk)].matmul(prototypes.t()))
    return torch.cat(rows, 0).argmax(1)


def _balanced(
    prediction: torch.Tensor, labels: torch.Tensor, num_anchor: int
) -> Dict[str, float]:
    anchor = labels < int(num_anchor)
    novel = ~anchor
    anchor_value = 100.0 * float(
        (prediction[anchor] == labels[anchor]).float().mean().item()
    )
    novel_value = 100.0 * float(
        (prediction[novel] == labels[novel]).float().mean().item()
    )
    return {
        "anchor": anchor_value,
        "pseudo_novel": novel_value,
        "balanced": 0.5 * (anchor_value + novel_value),
    }


def _curve_summary(curve: Mapping[str, Sequence[float]]) -> Dict[str, object]:
    return {
        key: {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
            "episodes": len(values),
        }
        for key, values in curve.items()
    }


def lock_parameters(bundle: Path, device: str, dimension: int) -> Dict[str, object]:
    data = load_construction_selection(bundle, device)
    construction = torch.cat(
        [data.construction[class_id] for class_id in sorted(data.construction)], 0
    )
    gamma = gamma_scale(construction)
    weights, offset = make_rbf_projection(
        construction.shape[1], dimension, gamma, RANDUMB_SEED, construction.device
    )
    result = {
        "schema_version": 1,
        "stage": "base_only_randumb_parameter_lock",
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "projection_dimension": int(dimension),
        "projection_seed": RANDUMB_SEED,
        "gamma": gamma,
        "gamma_rule": "1 / (input_dimension * construction_feature_variance)",
        "weights_sha256": tensor_sha256(weights),
        "offset_sha256": tensor_sha256(offset),
        "implementation": "official RBFSampler formula followed by cosine NCM",
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_parameters(
    bundle: Path,
    parameters: Mapping[str, object],
    device: str,
    episodes: int,
    feature_chunk: int,
) -> Dict[str, object]:
    data = load_audit(bundle, device)
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("RanDumb parameter/audit dataset mismatch")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("RanDumb parameter/audit bundle mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("RanDumb parameter/audit feature mismatch")
    anchor_train, anchor_queries = _anchor_split(data.anchors, 314159)
    train, train_labels, anchor_query, anchor_labels, anchor_ids = _flat_anchor_data(
        anchor_train, anchor_queries
    )
    weights, offset = make_rbf_projection(
        train.shape[1],
        int(parameters["projection_dimension"]),
        float(parameters["gamma"]),
        int(parameters["projection_seed"]),
        train.device,
    )
    if tensor_sha256(weights) != parameters["weights_sha256"]:
        raise RuntimeError("RanDumb weight hash mismatch")
    base_projected = projected_centroids(
        train,
        train_labels,
        list(range(len(anchor_ids))),
        weights,
        offset,
        feature_chunk,
    )
    anchor_query_h = rbf_transform_chunked(
        anchor_query, weights, offset, feature_chunk
    )
    original_anchor = torch.stack(
        [normalized_mean(anchor_train[class_id]) for class_id in anchor_ids], 0
    )
    curve = {"anchor": [], "pseudo_novel": [], "balanced": []}
    cosine_curve = {"anchor": [], "pseudo_novel": [], "balanced": []}
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[data.dataset]
    for episode_index in range(int(episodes)):
        support, support_labels, pseudo_query, pseudo_labels, chosen = _pseudo_episode(
            data.audit, episode_index, pseudo_count, len(anchor_ids)
        )
        pseudo_ids = list(range(len(anchor_ids), len(anchor_ids) + len(chosen)))
        pseudo_projected = projected_centroids(
            support,
            support_labels,
            pseudo_ids,
            weights,
            offset,
            feature_chunk,
        )
        query_h = torch.cat(
            [
                anchor_query_h,
                rbf_transform_chunked(pseudo_query, weights, offset, feature_chunk),
            ],
            0,
        )
        labels = torch.cat([anchor_labels, pseudo_labels], 0)
        prediction = cosine_prediction(
            query_h, torch.cat([base_projected, pseudo_projected], 0), feature_chunk
        )
        values = _balanced(prediction, labels, len(anchor_ids))
        for metric, value in values.items():
            curve[metric].append(value)
        original_pseudo = torch.stack(
            [
                normalized_mean(
                    support[support_labels == len(anchor_ids) + position]
                )
                for position in range(len(chosen))
            ],
            0,
        )
        original_prediction = cosine_prediction(
            torch.cat([anchor_query, pseudo_query], 0),
            torch.cat([original_anchor, original_pseudo], 0),
            feature_chunk,
        )
        original_values = _balanced(
            original_prediction, labels, len(anchor_ids)
        )
        for metric, value in original_values.items():
            cosine_curve[metric].append(value)
    summary = _curve_summary(curve)
    cosine_summary = _curve_summary(cosine_curve)
    difference = summary["balanced"]["mean"] - cosine_summary["balanced"]["mean"]
    result = {
        "schema_version": 1,
        "stage": "untouched_base_audit_randumb_ncm",
        "dataset": data.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "projection_dimension": int(parameters["projection_dimension"]),
        "episodes": int(episodes),
        "randumb": summary,
        "cosine": cosine_summary,
        "selected_minus_cosine": float(difference),
        "passes_balanced_non_regression_gate": bool(difference >= -5.0),
        "passes_sanity_gate": bool(np.isfinite(difference)),
    }
    result["audit_result_sha256"] = stable_json_sha256(result)
    return result


def _target_accuracy(
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


def evaluate_target(
    feature_cache: Path,
    protocol: Path,
    parameters: Mapping[str, object],
    audit: Mapping[str, object],
    device: str,
    seed_limit: int,
    feature_chunk: int,
    verify_file: bool,
) -> Dict[str, object]:
    started = time.time()
    data = load_target_evaluation_data(
        feature_cache, protocol, device, verify_file=verify_file
    )
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("RanDumb parameter/target dataset mismatch")
    if parameters["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("RanDumb parameter/target feature mismatch")
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("RanDumb parameter/audit provenance mismatch")
    if not bool(audit["passes_sanity_gate"]):
        raise RuntimeError("RanDumb failed numerical audit")
    if seed_limit < 1 or seed_limit > len(data.support_seeds):
        raise ValueError("invalid seed limit")
    weights, offset = make_rbf_projection(
        data.train_features.shape[1],
        int(parameters["projection_dimension"]),
        float(parameters["gamma"]),
        int(parameters["projection_seed"]),
        data.train_features.device,
    )
    if tensor_sha256(weights) != parameters["weights_sha256"]:
        raise RuntimeError("RanDumb target weight hash mismatch")
    construction_started = time.time()
    base_mask = data.train_labels < data.num_base
    base_prototypes = projected_centroids(
        data.train_features[base_mask],
        data.train_labels[base_mask],
        list(range(data.num_base)),
        weights,
        offset,
        feature_chunk,
    )
    projected_test = rbf_transform_chunked(
        data.test_features, weights, offset, feature_chunk
    )
    construction_sec = time.time() - construction_started
    num_sessions = 1 + (data.num_total - data.num_base) // data.increment
    seed_rows = []
    prediction_sec = 0.0
    for seed_position in range(int(seed_limit)):
        sessions = []
        for session_index in range(num_sessions):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            if seen_novel:
                support_rows = data.standard_support_abs[
                    seed_position, :seen_novel
                ].reshape(-1)
                support = data.train_features[support_rows]
                support_labels = data.train_labels[support_rows]
                novel_prototypes = projected_centroids(
                    support,
                    support_labels,
                    list(range(data.num_base, end_class)),
                    weights,
                    offset,
                    feature_chunk,
                )
                prototypes = torch.cat([base_prototypes, novel_prototypes], 0)
            else:
                prototypes = base_prototypes
            query_mask = data.test_labels < end_class
            then = time.time()
            prediction = cosine_prediction(
                projected_test[query_mask], prototypes, feature_chunk
            )
            prediction_sec += time.time() - then
            row = _target_accuracy(
                prediction,
                data.test_labels[query_mask],
                data.num_base,
                end_class,
            )
            row.update({"session": session_index, "end_class": end_class})
            sessions.append(row)
        session_average = float(mean(float(row["overall"]) for row in sessions))
        seed_rows.append(
            {
                "support_seed": data.support_seeds[seed_position],
                "session_average": session_average,
                "sessions": sessions,
            }
        )
        print(
            "dataset={} RanDumb-NCM seed={} session_average={:.3f}".format(
                data.dataset, data.support_seeds[seed_position], session_average
            ),
            flush=True,
        )
    averages = [float(row["session_average"]) for row in seed_rows]
    final_values = [float(row["sessions"][-1]["overall"]) for row in seed_rows]
    per_session = []
    for session_index in range(num_sessions):
        values = [
            float(row["sessions"][session_index]["overall"]) for row in seed_rows
        ]
        per_session.append(
            {"mean": float(mean(values)), "std": float(pstdev(values))}
        )
    dimension = int(parameters["projection_dimension"])
    return {
        "schema_version": 1,
        "stage": "strict_inductive_randumb_ncm_target_evaluation",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "information_level": "I0/I1: support-only and base-locked kernel scale",
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "audit_result_sha256": audit["audit_result_sha256"],
        "projection_dimension": dimension,
        "projection_seed": int(parameters["projection_seed"]),
        "gamma": float(parameters["gamma"]),
        "support_seeds": data.support_seeds[:seed_limit],
        "per_seed": seed_rows,
        "summary": {
            "session_average_mean": float(mean(averages)),
            "session_average_std": float(pstdev(averages)),
            "session_average_per_seed": averages,
            "final_accuracy_mean": float(mean(final_values)),
            "final_accuracy_std": float(pstdev(final_values)),
            "per_session": per_session,
        },
        "resources": {
            "head_and_test_projection_sec": construction_sec,
            "prediction_sec": prediction_sec,
            "state_bytes": int(
                (data.train_features.shape[1] * dimension + dimension + data.num_total * dimension)
                * 4
            ),
        },
        "runtime_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--bundle", required=True, type=Path)
    lock.add_argument("--output", required=True, type=Path)
    lock.add_argument("--device", default="cuda")
    lock.add_argument("--projection-dimension", type=int, default=DEFAULT_DIMENSION)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--bundle", required=True, type=Path)
    audit_parser.add_argument("--parameters", required=True, type=Path)
    audit_parser.add_argument("--output", required=True, type=Path)
    audit_parser.add_argument("--device", default="cuda")
    audit_parser.add_argument("--episodes", type=int, default=20)
    audit_parser.add_argument("--feature-chunk", type=int, default=1024)
    target = subparsers.add_parser("target")
    target.add_argument("--feature-cache", required=True, type=Path)
    target.add_argument("--protocol", required=True, type=Path)
    target.add_argument("--parameters", required=True, type=Path)
    target.add_argument("--audit", required=True, type=Path)
    target.add_argument("--output", required=True, type=Path)
    target.add_argument("--device", default="cuda")
    target.add_argument("--seed-limit", type=int, default=20)
    target.add_argument("--feature-chunk", type=int, default=1024)
    target.add_argument("--skip-file-hash", action="store_true")
    args = parser.parse_args()
    if args.command == "lock":
        result = lock_parameters(
            args.bundle, args.device, args.projection_dimension
        )
    elif args.command == "audit":
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit_parameters(
            args.bundle,
            parameters,
            args.device,
            args.episodes,
            args.feature_chunk,
        )
    else:
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        with args.audit.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        result = evaluate_target(
            args.feature_cache,
            args.protocol,
            parameters,
            audit,
            args.device,
            args.seed_limit,
            args.feature_chunk,
            verify_file=not args.skip_file_hash,
        )
        result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    summary = {
        "dataset": result["dataset"],
        "stage": result["stage"],
        "projection_dimension": result["projection_dimension"],
        "selected_minus_cosine": result.get("selected_minus_cosine"),
        "session_average_mean": result.get("summary", {}).get("session_average_mean"),
        "runtime_sec": result.get("runtime_sec"),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
