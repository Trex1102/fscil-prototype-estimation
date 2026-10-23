#!/usr/bin/env python3
"""Evaluate every table method from both locked support-only heads.

This is a parameter-transfer ablation: existing base-selected hyperparameters
are frozen before loading target queries and transferred unchanged to both
heads. No target results are used to select a setting. The native benchmark
remains a separate experiment; here every method explicitly receives its
parent's prototypes and geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .common_head_transduction import METHODS, common_head_method, head_transform, initial_scores
from .constants import RIDGE_GAMMA
from .geometry import CovarianceEigensystem, covariance_eigensystem, inverse_eigenvalues, metric_scores
from .qpr_selection import deployed_qpr_choice
from .qpr_self_exclusion import qpr_same_and_leave_one_out_scores
from .radial_selection import deployed_radial_choice
from .run_transductive import _accuracy, _support_summary
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, stable_json_sha256


HEADS = ("cosine", "shrinksel")
ALL_METHODS = ("baseline", "qpr_same_cohort", "qpr_leave_one_out") + METHODS


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cub200", "cifar100", "miniimagenet"))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-root", type=Path, required=True,
                        help="Dataset parameter directory produced by reproduce.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--sessions", choices=("all", "final"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.time()
    ds = args.dataset
    parameters = args.parameter_root.resolve()
    shrink = load(parameters / "shrinksel_selection.json")
    shrink_audit = load(parameters / "shrinksel_audit.json")
    qpr = load(parameters / "qpr_selection.json")
    qpr_audit = load(parameters / "qpr_audit.json")
    comparator = load(parameters / "transductive_selection.json")
    comparator_audit = load(parameters / "transductive_audit.json")
    for payload in (shrink, shrink_audit, qpr, qpr_audit, comparator, comparator_audit):
        if payload["dataset"] != ds:
            raise RuntimeError("Parameter dataset mismatch")
    if comparator_audit["selection_result_sha256"] != comparator["selection_result_sha256"]:
        raise RuntimeError("Comparator selection/audit mismatch")
    if not comparator_audit["passes_numerical_sanity_gate"]:
        raise RuntimeError("Comparator numerical audit failed")
    if qpr["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("QPR/ShrinkSel provenance mismatch")
    mode, lam = deployed_radial_choice(shrink, shrink_audit)
    qpr_parameters = deployed_qpr_choice(qpr, qpr_audit)
    data = load_target_evaluation_data(args.feature_cache, args.protocol, args.device)
    if data.dataset != ds:
        raise RuntimeError("Protocol dataset mismatch")
    for payload in (shrink, qpr):
        if payload["source_feature_file_sha256"] != data.selection_source_sha256:
            raise RuntimeError("Parameter/target source mismatch")
    if comparator["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("Comparator/target source mismatch")
    if not 1 <= args.seed_limit <= len(data.support_seeds):
        raise ValueError("Invalid support draw count")
    by_class = features_by_class(data)
    covariance = covariance_eigensystem(by_class, range(data.num_base))
    train_mean = data.train_features[data.train_labels < data.num_base].mean(0)
    transform = head_transform(covariance, lam, RIDGE_GAMMA)
    dimension = data.train_features.shape[1]
    identity = CovarianceEigensystem(
        torch.ones(dimension, device=args.device),
        torch.eye(dimension, device=args.device),
        torch.tensor(0.5, device=args.device), 1)
    sessions = list(range(1 + (data.num_total - data.num_base) // data.increment))
    if args.sessions == "final":
        sessions = sessions[-1:]
    result = {
        "schema_version": 1, "stage": "two_parent_common_head_parameter_transfer",
        "state": "running", "dataset": ds, "backbone": data.backbone,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_source_sha256": data.selection_source_sha256,
        "support_seeds": data.support_seeds[:args.seed_limit],
        "session_indices": sessions, "heads": list(HEADS),
        "methods": list(ALL_METHODS),
        "parameter_policy": "existing base-selected settings transferred unchanged between heads; no target tuning",
        "comparator_geometry": "parent whitening normalized by mean metric eigenvalue; no per-row renormalization",
        "laplacian_full_adaptation": "native CL2N retained using transformed base-training mean; parent directions centered and normalized with initial radii retained; query shift, optional prototype rectification and Euclidean graph retained",
        "qpr_geometry": "original parent coordinates and scores; cosine dot-product logits; original ShrinkSel Mahalanobis logits",
        "shrinksel_lambda": lam, "shrinksel_prototype_mode": mode,
        "qpr_parameters": qpr_parameters, "comparator_parameters": comparator,
        "selection_hashes": {"shrinksel": shrink["selection_result_sha256"],
                             "qpr": qpr["selection_result_sha256"],
                             "comparators": comparator["selection_result_sha256"]},
        "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("common_head_transduction.py", "run_common_head_benchmark.py",
                                       "transductive_methods.py", "qpr_self_exclusion.py")},
        "per_seed": [], "initialization_maximum_mismatches": 0,
        "initialization_maximum_non_tie_mismatches": 0,
        "initialization_maximum_tie_gap": 0.0,
        "initialization_tolerance": 5e-6,
        "initialization_maximum_score_error": 0.0,
    }
    base_cache = None
    for seed_index in range(args.seed_limit):
        seed_row = {"support_seed": data.support_seeds[seed_index], "sessions": []}
        for session in sessions:
            if session == 0 and base_cache is not None:
                seed_row["sessions"].append(base_cache)
                continue
            end = data.num_base + session * data.increment
            mask = data.test_labels < end
            queries, labels = data.test_features[mask], data.test_labels[mask]
            support, support_labels, cosine_p, shrink_p = _support_summary(
                data, by_class, seed_index, end, mode)
            row = {"session_index": session, "end_class": end, "heads": {}}
            for head in HEADS:
                p = cosine_p if head == "cosine" else shrink_p
                baseline = queries.matmul(p.t()) if head == "cosine" else metric_scores(
                    queries, p, covariance, lam, RIDGE_GAMMA)
                q = queries if head == "cosine" else queries.matmul(transform)
                s = support if head == "cosine" else support.matmul(transform)
                initial = p if head == "cosine" else p.matmul(transform)
                common_initial = initial_scores(q, initial)
                mismatch = int((common_initial.argmax(1) != baseline.argmax(1)).sum().item())
                result["initialization_maximum_mismatches"] = max(
                    result["initialization_maximum_mismatches"], mismatch)
                expected = baseline if head == "cosine" else baseline / float(
                    inverse_eigenvalues(covariance, lam, RIDGE_GAMMA).mean().item())
                bad = common_initial.argmax(1) != baseline.argmax(1)
                if mismatch:
                    index = bad.nonzero(as_tuple=False).flatten()
                    chosen = common_initial.argmax(1)[index]
                    gaps = expected[index].max(1).values - expected[index, chosen]
                    result["initialization_maximum_tie_gap"] = max(
                        result["initialization_maximum_tie_gap"], float(gaps.max().item()))
                    non_ties = int((gaps > result["initialization_tolerance"]).sum().item())
                    result["initialization_maximum_non_tie_mismatches"] = max(
                        result["initialization_maximum_non_tie_mismatches"], non_ties)
                    if non_ties:
                        raise RuntimeError("Common-head initialization changes a non-tied parent prediction")
                centered_error = (common_initial - common_initial[:, :1]) - (expected - expected[:, :1])
                result["initialization_maximum_score_error"] = max(
                    result["initialization_maximum_score_error"], float(centered_error.abs().max().item()))
                if float(centered_error.abs().max().item()) > result["initialization_tolerance"]:
                    raise RuntimeError("Common-head initialization does not preserve parent score differences")
                predictions = {"baseline": baseline.argmax(1)}
                metadata = {}
                if session == 0:
                    predictions["qpr_same_cohort"] = predictions["baseline"]
                    predictions["qpr_leave_one_out"] = predictions["baseline"]
                else:
                    qpr_output = qpr_same_and_leave_one_out_scores(
                        queries, p, baseline, identity if head == "cosine" else covariance,
                        1.0 if head == "cosine" else lam, RIDGE_GAMMA,
                        data.num_base, **qpr_parameters)
                    predictions["qpr_same_cohort"] = qpr_output.same_cohort_scores.argmax(1)
                    predictions["qpr_leave_one_out"] = qpr_output.leave_one_out_scores.argmax(1)
                for method in METHODS:
                    output = common_head_method(
                        method, s, support_labels, q, initial, comparator,
                        train_mean=train_mean if head == "cosine" else train_mean.matmul(transform))
                    predictions[method] = output.scores.argmax(1)
                    metadata[method] = output.metadata
                row["heads"][head] = {
                    "methods": {method: _accuracy(prediction, labels, data.num_base, end)
                                for method, prediction in predictions.items()},
                    "metadata": metadata,
                }
            seed_row["sessions"].append(row)
            if session == 0:
                base_cache = row
        result["per_seed"].append(seed_row)
        result["completed_support_draws"] = len(result["per_seed"])
        result["runtime_sec"] = time.time() - start
        atomic_write_json(args.output, result)
        print(json.dumps({"dataset": ds, "completed_support_draws": len(result["per_seed"]),
                          "runtime_sec": result["runtime_sec"]}), flush=True)
    result["state"] = "complete"
    result["runtime_sec"] = time.time() - start
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
