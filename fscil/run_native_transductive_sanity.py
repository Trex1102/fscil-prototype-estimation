#!/usr/bin/env python3
"""Run a native-format 5-way/5-shot/15-query comparator sanity check.

This is not a reproduction of any comparator paper's published accuracy: all
methods receive the same frozen CLIP image embeddings and the FSCIL train/test
split.  Its purpose is narrower and auditable--verify that an implementation
behaves sensibly in the low-way balanced episodic regime for which the original
transductive algorithms were designed before adapting it to all-seen FSCIL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import torch

from .constants import PROTOCOL_SEED
from .target_data import load_target_evaluation_data
from .transductive_methods import (
    TransductiveOutput,
    laplacian_shot,
    laplacian_shot_full,
    protolp_paper,
    pt_map,
    query_prototype_rectification,
    support_prototypes,
    tim_gd,
)
from .utils import atomic_write_json, stable_json_sha256


METHODS = (
    "cosine",
    "tim_gd",
    "laplacian_shot",
    "laplacian_shot_core",
    "protolp_paper",
    "ptmap_uniform",
    "qpr_gates_inactive",
)


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected(parameters: Mapping[str, object], family: str) -> List[float]:
    return [float(value) for value in parameters["selected"][family]["values"]]


def _episode(
    data,
    episode_index: int,
    ways: int,
    queries_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], int]:
    novel = np.arange(data.num_base, data.num_total, dtype=np.int64)
    rng = np.random.RandomState(PROTOCOL_SEED + 104729 * int(episode_index))
    classes = sorted(
        int(value) for value in rng.choice(novel, size=int(ways), replace=False)
    )
    seed_position = int(episode_index) % len(data.support_seeds)
    support_parts = []
    support_labels = []
    query_parts = []
    query_labels = []
    for local_label, class_id in enumerate(classes):
        rows = data.standard_support_abs[
            seed_position, class_id - data.num_base
        ]
        support_parts.append(data.train_features[rows])
        support_labels.append(
            torch.full(
                (rows.numel(),),
                local_label,
                dtype=torch.long,
                device=data.train_features.device,
            )
        )
        candidates = torch.where(data.test_labels == class_id)[0]
        if candidates.numel() < int(queries_per_class):
            raise RuntimeError(
                "class {} has fewer than {} test queries".format(
                    class_id, queries_per_class
                )
            )
        query_generator = torch.Generator(device="cpu")
        query_generator.manual_seed(
            PROTOCOL_SEED + 15485863 * episode_index + 32452843 * class_id
        )
        order = torch.randperm(
            candidates.numel(), generator=query_generator
        ).to(candidates.device)
        chosen = candidates[order[: int(queries_per_class)]]
        query_parts.append(data.test_features[chosen])
        query_labels.append(
            torch.full(
                (int(queries_per_class),),
                local_label,
                dtype=torch.long,
                device=data.test_features.device,
            )
        )
    support = torch.cat(support_parts, 0)
    support_label = torch.cat(support_labels, 0)
    query = torch.cat(query_parts, 0)
    query_label = torch.cat(query_labels, 0)
    # Remove class-block ordering without consulting feature values.
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(PROTOCOL_SEED + 49979687 * episode_index)
    order = torch.randperm(query.shape[0], generator=shuffle_generator).to(
        query.device
    )
    return (
        support,
        support_label,
        query[order],
        query_label[order],
        classes,
        data.support_seeds[seed_position],
    )


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    average = float(mean(values))
    deviation = float(pstdev(values)) if len(values) > 1 else 0.0
    return {
        "mean": average,
        "std": deviation,
        "ci95_half_width": (
            1.96 * deviation / math.sqrt(len(values)) if values else float("nan")
        ),
        "episodes": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--transductive-parameters", required=True, type=Path)
    parser.add_argument("--qpr-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--queries-per-class", type=int, default=15)
    parser.add_argument("--graph-chunk", type=int, default=512)
    parser.add_argument("--skip-file-hash", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.ways < 2 or args.queries_per_class < 1:
        raise ValueError("invalid episodic dimensions")

    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    parameters = _load_json(args.transductive_parameters)
    if int(parameters.get("schema_version", 0)) != 7:
        raise RuntimeError("native sanity check requires schema-v7 parameters")
    if parameters["dataset"] != data.dataset:
        raise RuntimeError("transductive parameter dataset mismatch")
    if parameters["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("transductive parameter feature mismatch")
    laplacian_values = _selected(parameters, "laplacian")
    laplacian_core_values = _selected(parameters, "laplacian_core")
    ptmap_values = _selected(parameters, "ptmap")
    protolp_values = _selected(parameters, "protolp_paper")
    tim_parameters = parameters["tim"]
    train_mean = data.train_features[data.train_labels < data.num_base].mean(0)

    # The degenerate QPR row transfers this dataset's locked base-selected
    # tuple unchanged; nothing about it is retuned for the episodic format.
    qpr_selection = _load_json(args.qpr_selection)
    if qpr_selection.get("dataset") != data.dataset:
        raise RuntimeError("QPR selection dataset mismatch")
    qpr_selected = qpr_selection["selected"]
    qpr_alpha = float(qpr_selected["alpha"])
    qpr_mass_threshold = float(qpr_selected["mass_threshold"])
    qpr_min_count = int(qpr_selected["min_count"])
    qpr_logit_scale = float(qpr_selected["logit_scale"])

    scores: Dict[str, List[float]] = {method: [] for method in METHODS}
    runtimes: Dict[str, List[float]] = {method: [] for method in METHODS}
    records = []
    last_metadata: Dict[str, object] = {}
    for local_index in range(int(args.episodes)):
        episode_index = int(args.episode_offset) + local_index
        support, support_labels, query, labels, classes, support_seed = _episode(
            data, episode_index, args.ways, args.queries_per_class
        )
        prototypes = support_prototypes(
            support, support_labels, args.ways, normalize=True
        )
        functions: Dict[str, Callable[[], TransductiveOutput]] = {
            "tim_gd": lambda: tim_gd(
                support,
                support_labels,
                query,
                args.ways,
                iterations=int(tim_parameters["iterations"]),
                temperature=float(tim_parameters["temperature"]),
                learning_rate=float(tim_parameters["learning_rate"]),
                cross_entropy_weight=float(tim_parameters["loss_weights"][0]),
                marginal_entropy_weight=float(tim_parameters["loss_weights"][1]),
                conditional_entropy_weight=float(tim_parameters["loss_weights"][2]),
            ),
            "laplacian_shot": lambda: laplacian_shot_full(
                support,
                support_labels,
                query,
                args.ways,
                train_mean=train_mean,
                knn=int(laplacian_values[0]),
                regularization=float(laplacian_values[1]),
                # Exercise the complete released configuration in its native
                # low-way regime, independently of the all-seen selector.
                prototype_rectification=True,
                chunk_size=args.graph_chunk,
            ),
            "laplacian_shot_core": lambda: laplacian_shot(
                prototypes,
                query,
                neighbors=int(laplacian_core_values[0]),
                regularization=float(laplacian_core_values[1]),
                chunk_size=args.graph_chunk,
            ),
            "protolp_paper": lambda: protolp_paper(
                support,
                support_labels,
                query,
                args.ways,
                graph_regularization=float(protolp_values[0]),
                assignment_scale=float(protolp_values[1]),
            ),
            "ptmap_uniform": lambda: pt_map(
                support,
                support_labels,
                query,
                args.ways,
                assignment_scale=float(ptmap_values[0]),
            ),
            # Degenerate QPR: the reported operator has no base block to gate
            # against here, so its novel-mass and predicted-novel gates admit
            # every query and only the within-novel margin survives.  Reported
            # as a diagnostic, never as the paper's ShrinkSel+QPR method.
            "qpr_gates_inactive": lambda: query_prototype_rectification(
                query,
                prototypes,
                query.matmul(prototypes.t()),
                num_base_classes=0,
                alpha=qpr_alpha,
                mass_threshold=qpr_mass_threshold,
                min_count=qpr_min_count,
                logit_scale=qpr_logit_scale,
                allow_all_novel=True,
            ),
        }
        episode_scores: Dict[str, float] = {}
        started = time.perf_counter()
        cosine_scores = query.matmul(prototypes.t())
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prediction = cosine_scores.argmax(1)
        accuracy = 100.0 * float((prediction == labels).float().mean().item())
        scores["cosine"].append(accuracy)
        runtimes["cosine"].append(elapsed)
        episode_scores["cosine"] = accuracy
        for method, function in functions.items():
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = function()
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not bool(torch.isfinite(output.scores).all()):
                raise RuntimeError(method + " produced non-finite scores")
            accuracy = 100.0 * float(
                (output.scores.argmax(1) == labels).float().mean().item()
            )
            scores[method].append(accuracy)
            runtimes[method].append(elapsed)
            episode_scores[method] = accuracy
            last_metadata[method] = output.metadata
        records.append(
            {
                "episode_index": episode_index,
                "support_seed": support_seed,
                "global_class_ids": classes,
                "accuracy": episode_scores,
            }
        )

    summaries = {method: _summarize(values) for method, values in scores.items()}
    runtime_summaries = {
        method: _summarize(values) for method, values in runtimes.items()
    }
    cosine = np.asarray(scores["cosine"], dtype=np.float64)
    paired = {
        method: _summarize(
            (np.asarray(values, dtype=np.float64) - cosine).tolist()
        )
        for method, values in scores.items()
        if method != "cosine"
    }
    random_accuracy = 100.0 / float(args.ways)
    passes = bool(
        all(
            math.isfinite(float(summaries[method]["mean"]))
            and float(summaries[method]["mean"]) > random_accuracy
            for method in METHODS
        )
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "native_format_transductive_sanity",
        "dataset": data.dataset,
        "backbone": dict(data.backbone),
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "episode_protocol": {
            "ways": int(args.ways),
            "shots": 5,
            "queries_per_class": int(args.queries_per_class),
            "episodes": int(args.episodes),
            "episode_offset": int(args.episode_offset),
            "class_pool": "FSCIL novel classes only",
            "support_source": "exact BiMC five-shot support draws",
            "query_source": "FSCIL test split",
            "balanced_query_cohort": True,
        },
        "interpretation_boundary": (
            "Native-format matched-feature implementation sanity check; not a "
            "reproduction of published accuracies or original backbones."
        ),
        "qpr_omission": (
            "QPR is an all-seen FSCIL base-versus-novel repair operator and has "
            "no defined base/novel boundary in a pure novel 5-way episode."
        ),
        "method_labels": {
            "ptmap_uniform": (
                "MAP-OT_u (PT-MAP core); published power transform is invalid "
                "for signed CLIP embeddings"
            ),
            "laplacian_shot": "full released LaplacianShot inference pipeline",
            "laplacian_shot_core": (
                "graph/bound-update core on matched frozen normalized features"
            ),
            "protolp_paper": "paper-equation reconstruction",
            "tim_gd": "official TIM-GD objective and defaults",
        },
        "transductive_selection_sha256": parameters["selection_result_sha256"],
        "selected_parameters": {
            "laplacian": laplacian_values,
            "laplacian_core": laplacian_core_values,
            "native_laplacian_forces_official_rectification_on": True,
            "ptmap_core": ptmap_values,
            "protolp_paper": protolp_values,
            "tim": tim_parameters,
        },
        "accuracy": summaries,
        "paired_delta_vs_cosine": paired,
        "runtime_seconds_per_episode": runtime_summaries,
        "last_metadata": last_metadata,
        "random_accuracy": random_accuracy,
        "passes_above_random_sanity_gate": passes,
        "episodes": records,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result["accuracy"], indent=2, sort_keys=True))
    if not passes:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
