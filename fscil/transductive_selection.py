#!/usr/bin/env python3
"""Select transductive hyperparameters using class-disjoint base data only."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .constants import PSEUDO_NOVEL_PER_EPISODE, PSEUDO_QUERY_PER_CLASS
from .selection import AuditData, ConstructionSelectionData, _anchor_split, load_audit, load_construction_selection
from .transductive_methods import (
    laplacian_shot,
    laplacian_shot_full,
    protolp_paper,
    protolp_uniform,
    pt_map,
    sparse_graph_repair,
    support_prototypes,
)
from .utils import atomic_write_json, normalized_mean, stable_json_sha256


LAPLACIAN_GRID = tuple(
    (neighbors, regularization, prototype_rectification)
    for neighbors in (3, 5, 10)
    for regularization in (0.1, 0.5, 1.0, 2.0)
    for prototype_rectification in (0, 1)
)
LAPLACIAN_CORE_GRID = tuple(
    (neighbors, regularization)
    for neighbors in (3, 5, 10)
    for regularization in (0.1, 0.5, 1.0, 2.0)
)
PTMAP_SCALE_GRID = (5.0, 10.0, 20.0)
# ProtoLP's exponent has no temperature on its original features.  Frozen CLIP
# changes the distance scale, so both the graph regularizer and an assignment
# scale are selected on class-disjoint base episodes.  Scale one retains the
# literal paper exponent; the other values prevent a deliberately diffuse
# 60--200-way adaptation from becoming an artificially weak comparator.
PROTOLP_GRID = tuple(
    (regularization, scale)
    for regularization in (0.1, 0.5, 1.0, 10.0)
    for scale in (1.0, 5.0, 10.0, 20.0)
)
GRAPH_GRID = tuple(
    (neighbors, anchor)
    for neighbors in (5, 10, 20)
    for anchor in (0.1, 0.3, 0.5)
)
ALL_FAMILIES = (
    "laplacian",
    "laplacian_core",
    "ptmap",
    "protolp_paper",
    "protolp_uniform",
    "graph",
)


@dataclass(frozen=True)
class BaseTransductivePools:
    dataset: str
    construction: Mapping[int, torch.Tensor]
    candidate: Mapping[int, torch.Tensor]
    bundle_sha256: str
    source_feature_file_sha256: str


@dataclass(frozen=True)
class CalibrationEpisode:
    support: torch.Tensor
    support_labels: torch.Tensor
    cohort: torch.Tensor
    cohort_labels: torch.Tensor
    num_anchor: int


def _from_selection(data: ConstructionSelectionData) -> BaseTransductivePools:
    return BaseTransductivePools(
        dataset=data.dataset,
        construction=data.construction,
        candidate=data.selection,
        bundle_sha256=data.bundle_sha256,
        source_feature_file_sha256=data.source_feature_file_sha256,
    )


def _from_audit(data: AuditData) -> BaseTransductivePools:
    return BaseTransductivePools(
        dataset=data.dataset,
        construction=data.anchors,
        candidate=data.audit,
        bundle_sha256=data.bundle_sha256,
        source_feature_file_sha256=data.source_feature_file_sha256,
    )


def _episode(
    pools: BaseTransductivePools,
    construction: Mapping[int, torch.Tensor],
    heldout: Mapping[int, torch.Tensor],
    episode_index: int,
    imbalanced: bool,
) -> CalibrationEpisode:
    rng = np.random.RandomState(7717 + episode_index * 104729)
    anchor_ids = sorted(construction)
    candidate_ids = sorted(heldout)
    count = min(PSEUDO_NOVEL_PER_EPISODE[pools.dataset], len(candidate_ids))
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidate_ids), size=count, replace=False
        ).tolist()
    )
    support_parts: List[torch.Tensor] = []
    support_label_parts: List[torch.Tensor] = []
    cohort_parts: List[torch.Tensor] = []
    cohort_label_parts: List[torch.Tensor] = []
    device = next(iter(construction.values())).device
    ordered = anchor_ids + chosen
    for new_label, class_id in enumerate(ordered):
        if class_id in construction:
            center = normalized_mean(construction[class_id])
            support = center[None, :].repeat(5, 1)
            values = heldout[class_id]
        else:
            values = heldout[class_id]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                32452843 + episode_index * 49999 + class_id * 15485863
            )
            order = torch.randperm(
                values.shape[0], generator=generator
            ).to(values.device)
            support = values[order[:5]]
            values = values[order[5 : 5 + PSEUDO_QUERY_PER_CLASS]]
        available = min(PSEUDO_QUERY_PER_CLASS, int(values.shape[0]))
        if available < 2:
            raise RuntimeError("insufficient calibration cohort samples")
        if imbalanced:
            take = 2 + (
                episode_index * 17 + class_id * 13 + new_label * 7
            ) % (available - 1)
        else:
            take = available
        offset = (episode_index * 3 + class_id) % available
        index = torch.arange(offset, offset + take, device=device) % available
        cohort = values[index]
        support_parts.append(support)
        support_label_parts.append(
            torch.full(
                (support.shape[0],), new_label, device=device, dtype=torch.long
            )
        )
        cohort_parts.append(cohort)
        cohort_label_parts.append(
            torch.full(
                (cohort.shape[0],), new_label, device=device, dtype=torch.long
            )
        )
    return CalibrationEpisode(
        support=torch.cat(support_parts, 0),
        support_labels=torch.cat(support_label_parts, 0),
        cohort=torch.cat(cohort_parts, 0),
        cohort_labels=torch.cat(cohort_label_parts, 0),
        num_anchor=len(anchor_ids),
    )


def _metrics(prediction: torch.Tensor, episode: CalibrationEpisode) -> Dict[str, float]:
    correct = prediction == episode.cohort_labels
    anchor = episode.cohort_labels < episode.num_anchor
    novel = ~anchor
    per_class = []
    for class_id in range(int(episode.support_labels.max().item()) + 1):
        mask = episode.cohort_labels == class_id
        per_class.append(float(correct[mask].float().mean().item()))
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "class_balanced": 100.0 * float(mean(per_class)),
        "anchor": 100.0 * float(correct[anchor].float().mean().item()),
        "pseudo_novel": 100.0 * float(correct[novel].float().mean().item()),
    }


def _key(values: Sequence[float]) -> str:
    return ",".join(str(float(value)) for value in values)


def _summarize(rows: Mapping[str, List[Dict[str, float]]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, values in rows.items():
        result[key] = {
            metric: {
                "mean": float(mean([row[metric] for row in values])),
                "std": float(pstdev([row[metric] for row in values])),
                "episodes": len(values),
            }
            for metric in ("overall", "class_balanced", "anchor", "pseudo_novel")
        }
    return result


def evaluate(
    pools: BaseTransductivePools,
    episodes: int,
    namespace_seed: int,
    families: Sequence[str] = ALL_FAMILIES,
) -> Dict[str, object]:
    families = tuple(families)
    if not families or len(set(families)) != len(families):
        raise ValueError("transductive families must be non-empty and unique")
    unknown = sorted(set(families).difference(ALL_FAMILIES))
    if unknown:
        raise ValueError("unknown transductive families {}".format(unknown))
    construction, construction_cohort = _anchor_split(
        pools.construction, namespace_seed
    )
    # LaplacianShot's CL2N statistic is constructed only from the anchor side
    # of the split.  Candidate/audit images cannot influence it.
    laplacian_train_mean = torch.cat(
        [construction[class_id] for class_id in sorted(construction)], 0
    ).mean(0)
    # Anchor class cohorts and held-out candidate images share one mapping.
    heldout: Dict[int, torch.Tensor] = dict(construction_cohort)
    heldout.update({int(key): value for key, value in pools.candidate.items()})
    rows: Dict[str, List[Dict[str, float]]] = {"cosine": []}
    if "laplacian" in families:
        for value in LAPLACIAN_GRID:
            rows["laplacian:" + _key(value)] = []
    if "laplacian_core" in families:
        for value in LAPLACIAN_CORE_GRID:
            rows["laplacian_core:" + _key(value)] = []
    if "ptmap" in families:
        for value in PTMAP_SCALE_GRID:
            rows["ptmap:" + _key((value,))] = []
    for value in PROTOLP_GRID:
        if "protolp_paper" in families:
            rows["protolp_paper:" + _key(value)] = []
        if "protolp_uniform" in families:
            rows["protolp_uniform:" + _key(value)] = []
    if "graph" in families:
        for value in GRAPH_GRID:
            rows["graph:" + _key(value)] = []
    condition_rows: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "balanced": {key: [] for key in rows},
        "imbalanced": {key: [] for key in rows},
    }
    for episode_index in range(int(episodes)):
        for condition in ("balanced", "imbalanced"):
            episode = _episode(
                pools,
                construction,
                heldout,
                episode_index,
                imbalanced=condition == "imbalanced",
            )
            num_classes = int(episode.support_labels.max().item()) + 1
            prototypes = support_prototypes(
                episode.support,
                episode.support_labels,
                num_classes,
                normalize=True,
            )
            predictions: Dict[str, torch.Tensor] = {
                "cosine": episode.cohort.matmul(prototypes.t()).argmax(1)
            }
            if "laplacian" in families:
                for neighbors, regularization, rectification in LAPLACIAN_GRID:
                    name = "laplacian:" + _key(
                        (neighbors, regularization, rectification)
                    )
                    predictions[name] = laplacian_shot_full(
                        episode.support,
                        episode.support_labels,
                        episode.cohort,
                        num_classes,
                        train_mean=laplacian_train_mean,
                        knn=int(neighbors),
                        regularization=float(regularization),
                        prototype_rectification=bool(int(rectification)),
                    ).scores.argmax(1)
            if "laplacian_core" in families:
                for neighbors, regularization in LAPLACIAN_CORE_GRID:
                    name = "laplacian_core:" + _key(
                        (neighbors, regularization)
                    )
                    predictions[name] = laplacian_shot(
                        prototypes,
                        episode.cohort,
                        neighbors=int(neighbors),
                        regularization=float(regularization),
                    ).scores.argmax(1)
            if "ptmap" in families:
                for scale in PTMAP_SCALE_GRID:
                    name = "ptmap:" + _key((scale,))
                    predictions[name] = pt_map(
                        episode.support,
                        episode.support_labels,
                        episode.cohort,
                        num_classes,
                        assignment_scale=float(scale),
                    ).scores.argmax(1)
            for regularization, scale in PROTOLP_GRID:
                if "protolp_paper" in families:
                    name = "protolp_paper:" + _key((regularization, scale))
                    predictions[name] = protolp_paper(
                        episode.support,
                        episode.support_labels,
                        episode.cohort,
                        num_classes,
                        graph_regularization=float(regularization),
                        assignment_scale=float(scale),
                    ).scores.argmax(1)
                if "protolp_uniform" in families:
                    uniform_name = "protolp_uniform:" + _key(
                        (regularization, scale)
                    )
                    predictions[uniform_name] = protolp_uniform(
                        episode.support,
                        episode.support_labels,
                        episode.cohort,
                        num_classes,
                        graph_regularization=float(regularization),
                        assignment_scale=float(scale),
                    ).scores.argmax(1)
            if "graph" in families:
                for neighbors, anchor in GRAPH_GRID:
                    name = "graph:" + _key((neighbors, anchor))
                    predictions[name] = sparse_graph_repair(
                        prototypes,
                        episode.cohort,
                        neighbors=int(neighbors),
                        prototype_anchor=float(anchor),
                    ).scores.argmax(1)
            for name, prediction in predictions.items():
                values = _metrics(prediction, episode)
                rows[name].append(values)
                condition_rows[condition][name].append(values)
    summary = _summarize(rows)
    conditions = {
        condition: _summarize(values)
        for condition, values in condition_rows.items()
    }
    return {
        "combined": summary,
        "conditions": conditions,
        "episodes_per_condition": int(episodes),
        "calibration_conditions": ["balanced", "deterministic_imbalanced"],
        "families": list(families),
    }


def _select_family(
    curve: Mapping[str, object], family: str
) -> Tuple[str, Mapping[str, object]]:
    candidates = {
        key: value
        for key, value in curve.items()
        if key.startswith(family + ":")
    }
    selected = max(
        candidates,
        key=lambda key: (
            float(candidates[key]["class_balanced"]["mean"]), key
        ),
    )
    return selected, candidates[selected]


def select(
    pools: BaseTransductivePools,
    episodes: int,
    families: Sequence[str] = ALL_FAMILIES,
) -> Dict[str, object]:
    evaluation = evaluate(
        pools, episodes, namespace_seed=161803, families=families
    )
    curve = evaluation["combined"]
    selected = {}
    for family in families:
        name, metrics = _select_family(curve, family)
        values = [float(value) for value in name.split(":", 1)[1].split(",")]
        selected[family] = {"key": name, "values": values, "metrics": metrics}
    result: Dict[str, object] = {
        "schema_version": 7,
        "stage": "transductive_base_selection",
        "dataset": pools.dataset,
        "selection_bundle_sha256": pools.bundle_sha256,
        "source_feature_file_sha256": pools.source_feature_file_sha256,
        "objective": (
            "class-balanced accuracy averaged over balanced and deterministic "
            "imbalanced unlabeled cohorts"
        ),
        "tim": {
            "source": "official TIM-GD defaults",
            "temperature": 15.0,
            "learning_rate": 1e-4,
            "iterations": 150,
            "loss_weights": [0.1, 1.0, 0.1],
        },
        "laplacian_shot": {
            "source": "full released inference pipeline",
            "components": [
                "CL2N",
                "prototype_rectification",
                "binary_directed_Euclidean_kNN",
                "bound_update",
            ],
            "train_mean_boundary": "construction anchors only",
            "prototype_rectification_selection": (
                "official optional switch selected on class-disjoint base episodes"
            ),
            "missing_rectification_class_fallback": "retain_CL2N_support_mean",
        },
        "protolp_variants": {
            "protolp_uniform": (
                "paper Eqs. 4 and 9--18 with the optional uniform Sinkhorn "
                "prior; graph regularization and CLIP assignment scale are "
                "selected on class-disjoint base episodes"
            ),
            "protolp_paper": (
                "primary paper Algorithm 1 / Eqs. 4 and 9--17 without a "
                "class marginal; support assignments follow Eq. 4 and graph "
                "regularization plus CLIP assignment scale are selected on "
                "class-disjoint base episodes"
            ),
        },
        "selected": selected,
        "evaluation": evaluation,
    }
    result["selection_result_sha256"] = stable_json_sha256(result)
    return result


def audit(
    pools: BaseTransductivePools,
    parameters: Mapping[str, object],
    episodes: int,
) -> Dict[str, object]:
    if int(parameters.get("schema_version", 0)) != 7:
        raise RuntimeError("transductive audit requires schema-v7 selection")
    if parameters["dataset"] != pools.dataset:
        raise RuntimeError("transductive parameter dataset mismatch")
    if parameters["selection_bundle_sha256"] != pools.bundle_sha256:
        raise RuntimeError("transductive selection/audit bundle mismatch")
    if parameters["source_feature_file_sha256"] != pools.source_feature_file_sha256:
        raise RuntimeError("transductive selection/audit feature mismatch")
    evaluation = evaluate(
        pools,
        episodes,
        namespace_seed=141421,
        families=tuple(parameters["selected"]),
    )
    combined = evaluation["combined"]
    cosine = float(combined["cosine"]["class_balanced"]["mean"])
    selected_rows = {}
    for family, item in parameters["selected"].items():
        key = item["key"]
        score = float(combined[key]["class_balanced"]["mean"])
        selected_rows[family] = {
            "key": key,
            "class_balanced_accuracy": score,
            "minus_cosine": score - cosine,
        }
    result: Dict[str, object] = {
        "schema_version": 7,
        "stage": "transductive_untouched_base_audit",
        "dataset": pools.dataset,
        "selection_result_sha256": parameters["selection_result_sha256"],
        "source_feature_file_sha256": pools.source_feature_file_sha256,
        "cosine_class_balanced_accuracy": cosine,
        "selected": selected_rows,
        "passes_numerical_sanity_gate": bool(
            all(np.isfinite(row["class_balanced_accuracy"]) for row in selected_rows.values())
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
        sub.add_argument("--episodes", type=int, default=20)
        sub.add_argument("--device", default="cuda")
        if command == "select":
            sub.add_argument("--families", default=",".join(ALL_FAMILIES))
        if command == "audit":
            sub.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        pools = _from_selection(load_construction_selection(args.bundle, args.device))
        families = tuple(
            value.strip() for value in args.families.split(",") if value.strip()
        )
        result = select(pools, args.episodes, families=families)
    else:
        pools = _from_audit(load_audit(args.bundle, args.device))
        with args.parameters.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
        result = audit(pools, parameters, args.episodes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "audit" and not result["passes_numerical_sanity_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
