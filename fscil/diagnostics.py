#!/usr/bin/env python3
"""Matched-feature error budget, shot scaling, and prototype-error diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .constants import SHOTS
from .geometry import cosine_scores
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, backbone_display, normalized_mean, stable_json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def predict_cosine_chunked(
    queries: torch.Tensor, prototypes: torch.Tensor, query_chunk: int
) -> torch.Tensor:
    predictions = []
    for start in range(0, queries.shape[0], query_chunk):
        predictions.append(
            cosine_scores(queries[start:start + query_chunk], prototypes).argmax(1)
        )
    return torch.cat(predictions, 0)


def class_accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int]
) -> np.ndarray:
    values = []
    for class_id in class_ids:
        mask = labels == int(class_id)
        values.append(float((prediction[mask] == labels[mask]).float().mean().item()))
    return np.asarray(values, dtype=np.float64)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def percentile_interval(values: np.ndarray) -> List[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def two_way_bootstrap(
    k_class_accuracy: np.ndarray,
    full_class_accuracy: np.ndarray,
    lp_class_accuracy: np.ndarray,
    draws: int,
    seed: int = 20260812,
) -> Dict[str, object]:
    """Independently resample both axes of the crossed seed-by-class design."""
    rng = np.random.RandomState(seed)
    num_seeds, num_classes = k_class_accuracy.shape
    samples = {
        "A_n_K": np.empty(draws),
        "A_n_full": np.empty(draws),
        "A_n_LP": np.empty(draws),
        "estimation_term": np.empty(draws),
        "form_term": np.empty(draws),
        "total_LP_minus_K": np.empty(draws),
        "normalized_estimation_fraction": np.empty(draws),
    }
    for draw in range(draws):
        sampled_seeds = rng.randint(0, num_seeds, size=num_seeds)
        sampled_classes = rng.randint(0, num_classes, size=num_classes)
        k_value = float(k_class_accuracy[np.ix_(sampled_seeds, sampled_classes)].mean())
        full_value = float(full_class_accuracy[sampled_classes].mean())
        lp_value = float(lp_class_accuracy[sampled_classes].mean())
        estimation = full_value - k_value
        form = lp_value - full_value
        denominator = lp_value - k_value
        ratio = estimation / denominator if abs(denominator) > 1e-12 else float("nan")
        samples["A_n_K"][draw] = k_value
        samples["A_n_full"][draw] = full_value
        samples["A_n_LP"][draw] = lp_value
        samples["estimation_term"][draw] = estimation
        samples["form_term"][draw] = form
        samples["total_LP_minus_K"][draw] = denominator
        samples["normalized_estimation_fraction"][draw] = ratio
    return {
        key: {
            "mean": 100.0 * float(np.nanmean(value)) if key != "normalized_estimation_fraction" else float(np.nanmean(value)),
            "ci95": (
                [100.0 * endpoint for endpoint in percentile_interval(value[~np.isnan(value)])]
                if key != "normalized_estimation_fraction"
                else percentile_interval(value[~np.isnan(value)])
            ),
        }
        for key, value in samples.items()
    }


def hierarchical_bootstrap(
    k_class_accuracy: np.ndarray,
    full_class_accuracy: np.ndarray,
    lp_class_accuracy: np.ndarray,
    draws: int,
    seed: int = 20260812,
) -> Dict[str, object]:
    """Backward-compatible alias for the two-way crossed-design bootstrap."""
    return two_way_bootstrap(
        k_class_accuracy,
        full_class_accuracy,
        lp_class_accuracy,
        draws,
        seed,
    )


def fit_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    started = time.time()
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        penalty="l2",
        fit_intercept=True,
        class_weight=None,
        max_iter=2000,
        tol=1e-6,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=0,
    )
    caught: List[str] = []
    with warnings.catch_warnings(record=True) as warning_rows:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(train_features.detach().cpu().numpy(), train_labels.detach().cpu().numpy())
        for row in warning_rows:
            caught.append(str(row.message))
    prediction = classifier.predict(test_features.detach().cpu().numpy())
    metadata = {
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
        "fit_intercept": True,
        "class_weight": None,
        "tol": 1e-6,
        "max_iter": 2000,
        "n_iter_per_class": classifier.n_iter_.tolist(),
        "convergence_warnings": caught,
        "runtime_sec": time.time() - started,
    }
    return torch.as_tensor(prediction, dtype=torch.long, device=test_features.device), metadata


def main() -> None:
    args = parse_args()
    started = time.time()
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid --seed-limit")
    by_class = features_by_class(data)
    class_ids = list(range(data.num_total))
    novel_ids = list(range(data.num_base, data.num_total))
    full_prototypes = torch.stack([normalized_mean(by_class[class_id]) for class_id in class_ids], 0)
    full_prediction = predict_cosine_chunked(
        data.test_features, full_prototypes, args.query_chunk
    )
    full_class = class_accuracy(full_prediction, data.test_labels, novel_ids)
    linear_prediction, linear_metadata = fit_linear_probe(
        data.train_features, data.train_labels, data.test_features
    )
    linear_class = class_accuracy(linear_prediction, data.test_labels, novel_ids)

    base_prototypes = full_prototypes[:data.num_base]
    shot_results: Dict[str, object] = {}
    five_shot_predictions: List[torch.Tensor] = []
    five_shot_angles: List[np.ndarray] = []
    five_shot_class_accuracy: List[np.ndarray] = []
    for shot in SHOTS:
        seed_class_accuracy: List[np.ndarray] = []
        seed_angles: List[np.ndarray] = []
        seed_accuracy: List[float] = []
        for seed_position in range(args.seed_limit):
            support_rows = data.nested_support_abs[seed_position, :, :shot]
            novel_prototypes = torch.stack(
                [normalized_mean(data.train_features[support_rows[row]]) for row in range(len(novel_ids))],
                0,
            )
            prototypes = torch.cat([base_prototypes, novel_prototypes], 0)
            prediction = predict_cosine_chunked(
                data.test_features, prototypes, args.query_chunk
            )
            accuracy_by_class = class_accuracy(prediction, data.test_labels, novel_ids)
            seed_class_accuracy.append(accuracy_by_class)
            seed_accuracy.append(100.0 * float(accuracy_by_class.mean()))
            cosine = (novel_prototypes * full_prototypes[data.num_base:]).sum(1).clamp(-1.0, 1.0)
            angles = torch.rad2deg(torch.acos(cosine)).detach().cpu().numpy()
            seed_angles.append(angles)
            if shot == 5:
                standard_rows = data.standard_support_abs[seed_position]
                if not torch.equal(support_rows, standard_rows):
                    raise RuntimeError("nested K=5 support does not equal the standard support")
                five_shot_predictions.append(prediction)
        accuracy_matrix = np.stack(seed_class_accuracy, 0)
        angle_matrix = np.stack(seed_angles, 0)
        shot_results[str(shot)] = {
            "novel_accuracy_mean": float(np.mean(seed_accuracy)),
            "novel_accuracy_std_over_support_seeds": float(np.std(seed_accuracy)),
            "prototype_angle_degrees_mean": float(angle_matrix.mean()),
            "prototype_angle_degrees_std": float(angle_matrix.std()),
            "per_seed_novel_accuracy": seed_accuracy,
            "per_seed_per_class_accuracy": accuracy_matrix.tolist(),
            "per_seed_per_class_angle_degrees": angle_matrix.tolist(),
        }
        if shot == 5:
            five_shot_class_accuracy = seed_class_accuracy
            five_shot_angles = seed_angles

    k_class = np.stack(five_shot_class_accuracy, 0)
    angle_values = np.stack(five_shot_angles, 0)
    loss_values = full_class[None, :] - k_class
    classwise_relation = correlation(angle_values.reshape(-1), loss_values.reshape(-1))

    repaired_harmed = []
    novel_mask = data.test_labels >= data.num_base
    full_correct = full_prediction[novel_mask] == data.test_labels[novel_mask]
    for seed_position, prediction in enumerate(five_shot_predictions):
        k_correct = prediction[novel_mask] == data.test_labels[novel_mask]
        repaired_harmed.append(
            {
                "support_seed": data.support_seeds[seed_position],
                "repaired_by_full_prototype": int((~k_correct & full_correct).sum().item()),
                "harmed_by_full_prototype": int((k_correct & ~full_correct).sum().item()),
                "correct_under_both": int((k_correct & full_correct).sum().item()),
                "wrong_under_both": int((~k_correct & ~full_correct).sum().item()),
            }
        )

    a_k = 100.0 * float(k_class.mean())
    a_full = 100.0 * float(full_class.mean())
    a_lp = 100.0 * float(linear_class.mean())
    estimation = a_full - a_k
    form = a_lp - a_full
    denominator = a_lp - a_k
    ratio = estimation / denominator if abs(denominator) > 1e-12 else float("nan")
    bootstrap = two_way_bootstrap(
        k_class,
        full_class,
        linear_class,
        args.bootstrap,
    )
    denominator_interval = bootstrap["total_LP_minus_K"]["ci95"]
    denominator_stable = bool(
        float(denominator_interval[0]) > 0.0
        or float(denominator_interval[1]) < 0.0
    )
    diagnosis_pass = bool(
        denominator_stable
        and ratio > 0.5
        and bootstrap["normalized_estimation_fraction"]["ci95"][0] > 0.5
    )
    result = {
        "schema_version": 1,
        "stage": "target_diagnostic_after_protocol_lock",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "protocol_sha256": data.protocol_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "support_seeds": data.support_seeds[:args.seed_limit],
        "linear_probe": linear_metadata,
        "accounting_decomposition_pct": {
            "A_n_K": a_k,
            "A_n_full": a_full,
            "A_n_LP": a_lp,
            "estimation_term": estimation,
            "classifier_form_term": form,
            "total_LP_minus_K": denominator,
            "normalized_estimation_fraction": ratio,
        },
        "bootstrap": bootstrap,
        "ratio_denominator_stable": denominator_stable,
        "diagnosis_majority_gate_for_dataset": diagnosis_pass,
        "shot_scaling": shot_results,
        "prototype_angle_vs_accuracy_loss": classwise_relation,
        "full_prototype_repaired_harmed_counts": repaired_harmed,
        "per_class_full_cosine_accuracy": (100.0 * full_class).tolist(),
        "per_class_linear_probe_accuracy": (100.0 * linear_class).tolist(),
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps({
        "dataset": data.dataset,
        "decomposition": result["accounting_decomposition_pct"],
        "bootstrap_ratio": bootstrap["normalized_estimation_fraction"],
        "diagnosis_gate": diagnosis_pass,
        "runtime_sec": result["runtime_sec"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
