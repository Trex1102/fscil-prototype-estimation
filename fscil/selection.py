"""Class-disjoint, base-training-only shrinkage selection and untouched audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .constants import (
    LAMBDA_GRID,
    PSEUDO_EPISODES,
    PSEUDO_NOVEL_PER_EPISODE,
    PSEUDO_QUERY_PER_CLASS,
    RIDGE_GAMMA,
)
from .geometry import (
    covariance_eigensystem,
    metric_scores_rotated,
    rotate_to_covariance_basis,
)
from .utils import atomic_write_json, normalized_mean, stable_json_sha256


@dataclass(frozen=True)
class ConstructionSelectionData:
    dataset: str
    construction: Mapping[int, torch.Tensor]
    selection: Mapping[int, torch.Tensor]
    bundle_sha256: str
    source_feature_file_sha256: str


@dataclass(frozen=True)
class AuditData:
    dataset: str
    anchors: Mapping[int, torch.Tensor]
    audit: Mapping[int, torch.Tensor]
    bundle_sha256: str
    source_feature_file_sha256: str


@dataclass(frozen=True)
class PseudoEpisode:
    queries: torch.Tensor
    labels: torch.Tensor
    prototypes: torch.Tensor
    num_anchor_classes: int


def _features_by_class(features: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    result = {}
    for class_id in sorted(int(value) for value in labels.unique().tolist()):
        result[class_id] = features[labels == class_id]
    return result


def load_construction_selection(path: Path, device: str) -> ConstructionSelectionData:
    payload = torch.load(str(path), map_location="cpu")
    forbidden = {key for key in payload if "test" in key.lower() or "novel" in key.lower()}
    if forbidden:
        raise RuntimeError("selection bundle exposes forbidden fields: {}".format(sorted(forbidden)))
    features = payload["base_features"].float().to(device)
    labels = payload["base_labels"].long().to(device)
    by_class = _features_by_class(features, labels)
    partition = payload["base_partition"]
    construction = {int(c): by_class[int(c)] for c in partition["construction"]}
    selection = {int(c): by_class[int(c)] for c in partition["selection"]}
    if set(construction).intersection(selection):
        raise RuntimeError("construction and selection classes overlap")
    return ConstructionSelectionData(
        dataset=str(payload["dataset"]),
        construction=construction,
        selection=selection,
        bundle_sha256=str(payload["selection_bundle_sha256"]),
        source_feature_file_sha256=str(payload["source_feature_file_sha256"]),
    )


def load_audit(path: Path, device: str) -> AuditData:
    payload = torch.load(str(path), map_location="cpu")
    forbidden = {key for key in payload if "test" in key.lower() or "novel" in key.lower()}
    if forbidden:
        raise RuntimeError("audit bundle exposes forbidden fields: {}".format(sorted(forbidden)))
    features = payload["base_features"].float().to(device)
    labels = payload["base_labels"].long().to(device)
    by_class = _features_by_class(features, labels)
    partition = payload["base_partition"]
    anchor_ids = list(partition["construction"]) + list(partition["selection"])
    audit_ids = list(partition["audit"])
    anchors = {int(c): by_class[int(c)] for c in anchor_ids}
    audit = {int(c): by_class[int(c)] for c in audit_ids}
    if set(anchors).intersection(audit):
        raise RuntimeError("anchor and audit classes overlap")
    return AuditData(
        dataset=str(payload["dataset"]),
        anchors=anchors,
        audit=audit,
        bundle_sha256=str(payload["selection_bundle_sha256"]),
        source_feature_file_sha256=str(payload["source_feature_file_sha256"]),
    )


def _anchor_split(
    features_by_class: Mapping[int, torch.Tensor], namespace_seed: int
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    construction: Dict[int, torch.Tensor] = {}
    query_pool: Dict[int, torch.Tensor] = {}
    for class_id, values in features_by_class.items():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(namespace_seed + class_id * 104729)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        query_count = min(PSEUDO_QUERY_PER_CLASS, max(1, values.shape[0] - 2))
        query_pool[class_id] = values[order[:query_count]]
        construction[class_id] = values[order[query_count:]]
        if construction[class_id].shape[0] < 2:
            raise RuntimeError("insufficient construction samples for class {}".format(class_id))
    return construction, query_pool


def _episode(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_queries: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episode_index: int,
    pseudo_novel_count: int,
) -> PseudoEpisode:
    rng = np.random.RandomState(9137 + episode_index * 7919)
    anchor_ids = sorted(anchor_train)
    candidate_ids = sorted(pseudo_novel_pool)
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidate_ids), size=min(pseudo_novel_count, len(candidate_ids)), replace=False
        ).tolist()
    )
    prototypes = [normalized_mean(anchor_train[class_id]) for class_id in anchor_ids]
    queries: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    device = next(iter(anchor_train.values())).device
    for new_label, class_id in enumerate(anchor_ids):
        values = anchor_queries[class_id]
        offset = (episode_index * PSEUDO_QUERY_PER_CLASS) % values.shape[0]
        indices = torch.arange(offset, offset + PSEUDO_QUERY_PER_CLASS, device=device) % values.shape[0]
        query = values[indices]
        queries.append(query)
        labels.append(torch.full((query.shape[0],), new_label, dtype=torch.long, device=device))
    num_anchor = len(anchor_ids)
    for position, class_id in enumerate(chosen):
        values = pseudo_novel_pool[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15485863 + episode_index * 32452843 + class_id * 49999)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        support = values[order[:5]]
        query = values[order[5:5 + PSEUDO_QUERY_PER_CLASS]]
        if query.shape[0] < PSEUDO_QUERY_PER_CLASS:
            raise RuntimeError("insufficient pseudo-query samples for class {}".format(class_id))
        prototypes.append(normalized_mean(support))
        queries.append(query)
        labels.append(
            torch.full(
                (query.shape[0],), num_anchor + position, dtype=torch.long, device=device
            )
        )
    return PseudoEpisode(
        queries=torch.cat(queries, 0),
        labels=torch.cat(labels, 0),
        prototypes=torch.stack(prototypes, 0),
        num_anchor_classes=num_anchor,
    )


def _group_metrics(prediction: torch.Tensor, episode: PseudoEpisode) -> Dict[str, float]:
    anchor_mask = episode.labels < episode.num_anchor_classes
    novel_mask = ~anchor_mask
    anchor = 100.0 * (prediction[anchor_mask] == episode.labels[anchor_mask]).float().mean().item()
    novel = 100.0 * (prediction[novel_mask] == episode.labels[novel_mask]).float().mean().item()
    return {"anchor": anchor, "pseudo_novel": novel, "balanced": 0.5 * (anchor + novel)}


def evaluate_curve(
    dataset: str,
    anchor_pool: Mapping[int, torch.Tensor],
    pseudo_novel_pool: Mapping[int, torch.Tensor],
    episodes: int,
    namespace_seed: int,
) -> Dict[str, object]:
    anchor_train, anchor_queries = _anchor_split(anchor_pool, namespace_seed)
    eigensystem = covariance_eigensystem(dict(anchor_train), sorted(anchor_train))
    curves = {
        float(value): {"anchor": [], "pseudo_novel": [], "balanced": []}
        for value in LAMBDA_GRID
    }
    pseudo_count = PSEUDO_NOVEL_PER_EPISODE[dataset]
    for episode_index in range(episodes):
        episode = _episode(
            anchor_train, anchor_queries, pseudo_novel_pool, episode_index, pseudo_count
        )
        rotated_queries = rotate_to_covariance_basis(episode.queries, eigensystem)
        rotated_prototypes = rotate_to_covariance_basis(episode.prototypes, eigensystem)
        for shrinkage in LAMBDA_GRID:
            prediction = metric_scores_rotated(
                rotated_queries,
                rotated_prototypes,
                eigensystem,
                float(shrinkage),
                RIDGE_GAMMA,
            ).argmax(1)
            values = _group_metrics(prediction, episode)
            for metric, value in values.items():
                curves[float(shrinkage)][metric].append(value)
    summary: Dict[str, object] = {}
    for shrinkage, metrics in curves.items():
        summary[str(shrinkage)] = {
            metric: {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "episodes": len(values),
            }
            for metric, values in metrics.items()
        }
    return {
        "curve": summary,
        "covariance_dof": eigensystem.dof,
        "delta": float(eigensystem.delta.item()),
        "episodes": episodes,
        "pseudo_novel_per_episode": pseudo_count,
    }


def select_shrinkage(data: ConstructionSelectionData, episodes: int) -> Dict[str, object]:
    evaluated = evaluate_curve(
        data.dataset,
        data.construction,
        data.selection,
        episodes,
        namespace_seed=271828,
    )
    curve = evaluated["curve"]
    selected = max(
        LAMBDA_GRID,
        key=lambda value: (curve[str(float(value))]["balanced"]["mean"], float(value)),
    )
    result = {
        "schema_version": 1,
        "stage": "base_selection",
        "dataset": data.dataset,
        "selection_bundle_sha256": data.bundle_sha256,
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_lambda": float(selected),
        "ridge_gamma": RIDGE_GAMMA,
        "objective": "mean of anchor and pseudo-novel accuracy",
        "candidate_grid": list(LAMBDA_GRID),
        "evaluation": evaluated,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit_shrinkage(data: AuditData, parameters: Mapping[str, object], episodes: int) -> Dict[str, object]:
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("dataset mismatch between parameters and audit bundle")
    if parameters["selection_bundle_sha256"] != data.bundle_sha256:
        raise RuntimeError("selection/audit bundle provenance mismatch")
    if parameters["source_feature_file_sha256"] != data.source_feature_file_sha256:
        raise RuntimeError("selection/audit feature provenance mismatch")
    evaluated = evaluate_curve(
        data.dataset,
        data.anchors,
        data.audit,
        episodes,
        namespace_seed=314159,
    )
    curve = evaluated["curve"]
    selected = float(parameters["selected_lambda"])
    selected_score = float(curve[str(selected)]["balanced"]["mean"])
    cosine_score = float(curve[str(1.0)]["balanced"]["mean"])
    oracle_lambda = max(
        LAMBDA_GRID,
        key=lambda value: (curve[str(float(value))]["balanced"]["mean"], float(value)),
    )
    oracle_score = float(curve[str(float(oracle_lambda))]["balanced"]["mean"])
    return {
        "schema_version": 1,
        "stage": "untouched_base_audit",
        "dataset": data.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": data.source_feature_file_sha256,
        "selected_lambda": selected,
        "selected_balanced_accuracy": selected_score,
        "cosine_balanced_accuracy": cosine_score,
        "selected_minus_cosine": selected_score - cosine_score,
        "audit_oracle_lambda": float(oracle_lambda),
        "audit_oracle_balanced_accuracy": oracle_score,
        "selection_regret_to_audit_oracle": oracle_score - selected_score,
        "passes_non_regression_gate": selected_score >= cosine_score - 0.10,
        "evaluation": evaluated,
    }


def _select_cli(args: argparse.Namespace) -> None:
    data = load_construction_selection(args.bundle, args.device)
    result = select_shrinkage(data, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _audit_cli(args: argparse.Namespace) -> None:
    data = load_audit(args.bundle, args.device)
    with args.parameters.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    result = audit_shrinkage(data, parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passes_non_regression_gate"]:
        raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "audit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--bundle", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--episodes", type=int, default=PSEUDO_EPISODES)
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        _select_cli(args)
    else:
        _audit_cli(args)


if __name__ == "__main__":
    main()
