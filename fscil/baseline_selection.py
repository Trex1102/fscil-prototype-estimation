#!/usr/bin/env python3
"""Base-only selection and untouched audit for inductive comparison heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .baseline_methods import (
    ULAParameters,
    cosine_scores_chunked,
    ranpac_normal_equations,
    ranpac_scores_chunked,
    solve_ranpac_from_eigensystem,
    teen_calibrate,
    ula_adjust_logits,
)
from .constants import LAMBDA_GRID, PSEUDO_EPISODES, PSEUDO_NOVEL_PER_EPISODE
from .selection import (
    AuditData,
    ConstructionSelectionData,
    PseudoEpisode,
    _anchor_split,
    _episode,
    _group_metrics,
    load_audit,
    load_construction_selection,
)
from .utils import atomic_write_json, stable_json_sha256


TEEN_SHIFT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75)
TEEN_SOFTMAX_SCALE_GRID = (4.0, 8.0, 16.0, 32.0)
ULA_MARGIN_GRID = (0.02, 0.04, 0.08, 0.16)
ULA_STRENGTH_GRID = (0.1, 0.2, 0.4, 0.8)
RANPAC_RIDGE_GRID = tuple(float(10.0 ** exponent) for exponent in range(-8, 9))
RANPAC_SELECTION_EPISODES = 20


def _empty_curve(keys: Sequence[object]) -> Dict[object, Dict[str, List[float]]]:
    return {
        key: {"anchor": [], "pseudo_novel": [], "balanced": []} for key in keys
    }


def _append_metrics(
    curve: Dict[object, Dict[str, List[float]]],
    key: object,
    prediction: torch.Tensor,
    episode: PseudoEpisode,
) -> None:
    metrics = _group_metrics(prediction, episode)
    for metric, value in metrics.items():
        curve[key][metric].append(float(value))


def _summarize_curve(
    curve: Mapping[object, Mapping[str, Sequence[float]]]
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, metrics in curve.items():
        result[str(key)] = {
            metric: {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "episodes": len(values),
            }
            for metric, values in metrics.items()
        }
    return result


def evaluate_teen_ula(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episodes: int,
    namespace_seed: int,
) -> Dict[str, object]:
    anchor_train, anchor_queries = _anchor_split(anchor_pool, namespace_seed)
    teen_keys = [
        (float(shift), float(scale))
        for shift in TEEN_SHIFT_GRID
        for scale in TEEN_SOFTMAX_SCALE_GRID
    ]
    ula_keys = [
        (float(threshold), float(strength))
        for threshold in ULA_MARGIN_GRID
        for strength in ULA_STRENGTH_GRID
    ]
    teen_curve = _empty_curve(teen_keys)
    ula_curve = _empty_curve(ula_keys)
    cosine_curve = _empty_curve(["cosine"])
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[dataset]
    for episode_index in range(episodes):
        episode = _episode(
            anchor_train,
            anchor_queries,
            pseudo_novel_pool,
            episode_index,
            pseudo_count,
        )
        num_anchor = episode.num_anchor_classes
        base_prototypes = episode.prototypes[:num_anchor]
        novel_prototypes = episode.prototypes[num_anchor:]
        cosine = cosine_scores_chunked(episode.queries, episode.prototypes)
        _append_metrics(cosine_curve, "cosine", cosine.argmax(1), episode)
        for key in teen_keys:
            shift, scale = key
            calibrated = teen_calibrate(
                base_prototypes, novel_prototypes, shift, scale
            )
            prototypes = torch.cat([base_prototypes, calibrated], 0)
            prediction = cosine_scores_chunked(
                episode.queries, prototypes
            ).argmax(1)
            _append_metrics(teen_curve, key, prediction, episode)
        for key in ula_keys:
            threshold, strength = key
            parameters = ULAParameters(
                margin_threshold=threshold,
                adjustment_strength=strength,
            )
            prediction = ula_adjust_logits(
                cosine, num_anchor, parameters
            ).argmax(1)
            _append_metrics(ula_curve, key, prediction, episode)
    return {
        "episodes": int(episodes),
        "pseudo_novel_per_episode": int(pseudo_count),
        "cosine": _summarize_curve(cosine_curve),
        "teen": _summarize_curve(teen_curve),
        "ula": _summarize_curve(ula_curve),
    }


def _selected_key(
    curve: Mapping[str, object], candidates: Sequence[Tuple[float, float]], default: Tuple[float, float]
) -> Tuple[float, float]:
    return max(
        candidates,
        key=lambda key: (
            float(curve[str(key)]["balanced"]["mean"]),
            -abs(float(key[0]) - float(default[0])),
            -abs(float(key[1]) - float(default[1])),
        ),
    )


def _ranpac_episode_parts(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_queries: Mapping[int, torch.Tensor],
    pseudo_pool: Mapping[int, torch.Tensor],
    episode_index: int,
    pseudo_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rng = np.random.RandomState(9137 + episode_index * 7919)
    candidate_ids = sorted(pseudo_pool)
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidate_ids),
            size=min(int(pseudo_count), len(candidate_ids)),
            replace=False,
        ).tolist()
    )
    anchor_ids = sorted(anchor_train)
    train_features: List[torch.Tensor] = []
    train_labels: List[torch.Tensor] = []
    query_features: List[torch.Tensor] = []
    query_labels: List[torch.Tensor] = []
    device = next(iter(anchor_train.values())).device
    for label, class_id in enumerate(anchor_ids):
        values = anchor_train[class_id]
        train_features.append(values)
        train_labels.append(
            torch.full((values.shape[0],), label, dtype=torch.long, device=device)
        )
        query = anchor_queries[class_id]
        query_features.append(query)
        query_labels.append(
            torch.full((query.shape[0],), label, dtype=torch.long, device=device)
        )
    num_anchor = len(anchor_ids)
    for position, class_id in enumerate(chosen):
        values = pseudo_pool[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15485863 + episode_index * 32452843 + class_id * 49999)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        support = values[order[:5]]
        query = values[order[5:20]]
        if query.shape[0] < 15:
            raise RuntimeError("insufficient pseudo-query samples for RanPAC selection")
        label = num_anchor + position
        train_features.append(support)
        train_labels.append(
            torch.full((support.shape[0],), label, dtype=torch.long, device=device)
        )
        query_features.append(query)
        query_labels.append(
            torch.full((query.shape[0],), label, dtype=torch.long, device=device)
        )
    return (
        torch.cat(train_features, 0),
        torch.cat(train_labels, 0),
        torch.cat(query_features, 0),
        torch.cat(query_labels, 0),
        num_anchor,
    )


def evaluate_ranpac_no_projection(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_pool: Mapping[int, torch.Tensor],
    episodes: int,
    namespace_seed: int,
) -> Dict[str, object]:
    anchor_train, anchor_queries = _anchor_split(anchor_pool, namespace_seed)
    curve = _empty_curve(RANPAC_RIDGE_GRID)
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[dataset]
    for episode_index in range(episodes):
        train_features, train_labels, queries, labels, num_anchor = _ranpac_episode_parts(
            anchor_train,
            anchor_queries,
            pseudo_pool,
            episode_index,
            pseudo_count,
        )
        num_classes = int(labels.max().item()) + 1
        gram, cross = ranpac_normal_equations(
            train_features, train_labels, num_classes
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.t()))
        eigenvalues = eigenvalues.clamp_min(0.0)
        synthetic = PseudoEpisode(
            queries=queries,
            labels=labels,
            prototypes=torch.empty(0, device=queries.device),
            num_anchor_classes=num_anchor,
        )
        for ridge in RANPAC_RIDGE_GRID:
            weights = solve_ranpac_from_eigensystem(
                eigenvalues, eigenvectors, cross, float(ridge)
            )
            prediction = ranpac_scores_chunked(queries, weights).argmax(1)
            _append_metrics(curve, float(ridge), prediction, synthetic)
    return {
        "episodes": int(episodes),
        "pseudo_novel_per_episode": int(pseudo_count),
        "ridge_grid": list(RANPAC_RIDGE_GRID),
        "curve": _summarize_curve(curve),
        "projection_dimension": 0,
        "selection_adaptation": (
            "official Phase-2 normal equations; ridge selected by balanced base-only "
            "pseudo-FSCIL accuracy instead of target data"
        ),
    }


def select_baselines(
    data: ConstructionSelectionData,
    episodes: int,
    ranpac_episodes: int,
) -> Dict[str, object]:
    calibration = evaluate_teen_ula(
        data.dataset,
        data.construction,
        data.selection,
        episodes,
        namespace_seed=271828,
    )
    teen_candidates = [
        (float(shift), float(scale))
        for shift in TEEN_SHIFT_GRID
        for scale in TEEN_SOFTMAX_SCALE_GRID
    ]
    ula_candidates = [
        (float(threshold), float(strength))
        for threshold in ULA_MARGIN_GRID
        for strength in ULA_STRENGTH_GRID
    ]
    teen_key = _selected_key(calibration["teen"], teen_candidates, (0.5, 16.0))
    ula_key = _selected_key(calibration["ula"], ula_candidates, (0.04, 0.4))
    ranpac = evaluate_ranpac_no_projection(
        data.dataset,
        data.construction,
        data.selection,
        ranpac_episodes,
        namespace_seed=271828,
    )
    selected_ridge = max(
        RANPAC_RIDGE_GRID,
        key=lambda value: (
            float(ranpac["curve"][str(float(value))]["balanced"]["mean"]),
            -abs(np.log10(float(value)) - 1.0),
        ),
    )
    result = {
        "schema_version": 1,
        "stage": "base_selection_inductive_baselines",
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "objective": "mean of anchor and pseudo-novel accuracy",
        "selected": {
            "teen": {
                "shift_weight": teen_key[0],
                "softmax_scale": teen_key[1],
            },
            "ula": {
                "logit_scale": 4.0,
                "margin_threshold": ula_key[0],
                "adjustment_strength": ula_key[1],
                "base_entropy_temperature": 0.25,
                "entropy_slope": 8.0,
                "entropy_threshold": 0.06,
            },
            "ranpac_no_projection": {"ridge": float(selected_ridge)},
        },
        "official_defaults": {
            "teen": {"shift_weight": 0.5, "softmax_scale": 16.0},
            "ula": {
                "logit_scale": 4.0,
                "margin_threshold": 0.04,
                "adjustment_strength": 0.4,
                "base_entropy_temperature": 0.25,
                "entropy_slope": 8.0,
                "entropy_threshold": 0.06,
            },
        },
        "calibration_evaluation": calibration,
        "ranpac_evaluation": ranpac,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_baselines(
    data: AuditData,
    parameters: Mapping[str, object],
    episodes: int,
    ranpac_episodes: int,
) -> Dict[str, object]:
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("dataset mismatch between selection and baseline audit")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("selection/audit bundle provenance mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("selection/audit feature provenance mismatch")
    calibration = evaluate_teen_ula(
        data.dataset,
        data.anchors,
        data.audit,
        episodes,
        namespace_seed=314159,
    )
    selected = parameters["selected"]
    teen = selected["teen"]
    teen_key = (float(teen["shift_weight"]), float(teen["softmax_scale"]))
    ula = selected["ula"]
    ula_key = (float(ula["margin_threshold"]), float(ula["adjustment_strength"]))
    cosine_score = float(calibration["cosine"]["cosine"]["balanced"]["mean"])
    teen_score = float(calibration["teen"][str(teen_key)]["balanced"]["mean"])
    ula_score = float(calibration["ula"][str(ula_key)]["balanced"]["mean"])
    ranpac = evaluate_ranpac_no_projection(
        data.dataset,
        data.anchors,
        data.audit,
        ranpac_episodes,
        namespace_seed=314159,
    )
    selected_ridge = float(selected["ranpac_no_projection"]["ridge"])
    ranpac_score = float(
        ranpac["curve"][str(selected_ridge)]["balanced"]["mean"]
    )
    ranpac_oracle = max(
        RANPAC_RIDGE_GRID,
        key=lambda value: float(
            ranpac["curve"][str(float(value))]["balanced"]["mean"]
        ),
    )
    result = {
        "schema_version": 1,
        "stage": "untouched_base_audit_inductive_baselines",
        "dataset": data.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "balanced_accuracy": {
            "cosine": cosine_score,
            "teen_selected": teen_score,
            "ula_selected": ula_score,
            "ranpac_no_projection_selected": ranpac_score,
        },
        "selected_minus_cosine": {
            "teen": teen_score - cosine_score,
            "ula": ula_score - cosine_score,
            "ranpac_no_projection": ranpac_score - cosine_score,
        },
        "ranpac_audit_oracle_ridge": float(ranpac_oracle),
        "ranpac_selection_regret": float(
            ranpac["curve"][str(float(ranpac_oracle))]["balanced"]["mean"]
            - ranpac_score
        ),
        # The no-projection branch is an explicit decorrelation ablation, not the
        # headline RanPAC baseline.  The separately audited 10k ReLU projection owns
        # the RanPAC sanity gate.
        "passes_sanity_gate": bool(
            teen_score >= cosine_score - 5.0
            and ula_score >= cosine_score - 5.0
            and np.isfinite(ranpac_score)
        ),
        "calibration_evaluation": calibration,
        "ranpac_evaluation": ranpac,
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
        sub.add_argument("--episodes", type=int, default=PSEUDO_EPISODES)
        sub.add_argument(
            "--ranpac-episodes", type=int, default=RANPAC_SELECTION_EPISODES
        )
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        data = load_construction_selection(args.bundle, args.device)
        result = select_baselines(data, args.episodes, args.ranpac_episodes)
    else:
        data = load_audit(args.bundle, args.device)
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit_baselines(
            data, parameters, args.episodes, args.ranpac_episodes
        )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "audit" and not result["passes_sanity_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
