#!/usr/bin/env python3
"""Evaluate I2 transduction and I3 marginal diagnostics on a locked protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from .constants import KNOWN_TEST_COUNTS, PROTOCOL_SEED, RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .qpr_selection import deployed_qpr_choice
from .radial_selection import (
    PROTOTYPE_MODES,
    deployed_radial_choice,
    prototype_from_samples,
)
from .target_data import features_by_class, load_target_evaluation_data
from .transductive_methods import (
    TransductiveOutput,
    laplacian_shot,
    laplacian_shot_full,
    marginal_constrained_scores,
    protolp_paper,
    protolp_uniform,
    pt_map,
    query_prototype_rectification,
    sparse_graph_repair,
    support_prototypes,
    tim_gd,
)
from .utils import atomic_write_json, backbone_display, normalized_mean, stable_json_sha256


DEFAULT_METHODS = (
    "cosine",
    "shrinksel",
    "tim_gd",
    "laplacian_shot",
    "laplacian_shot_core",
    "ptmap_uniform",
    "protolp_paper",
    "protolp_uniform",
    "sparse_graph_repair",
    "shrinksel_known_marginal",
    "protolp_paper_known_marginal",
    "ptmap_known_marginal",
)

QPR_METHOD = "qpr"
ALL_METHODS = DEFAULT_METHODS + (QPR_METHOD,)
INFERENCE_MARGINAL_METHODS = frozenset(
    {
        "shrinksel_known_marginal",
        "protolp_paper_known_marginal",
        "ptmap_known_marginal",
    }
)


def _requires_inference_marginal(methods: Sequence[str]) -> bool:
    return any(method in INFERENCE_MARGINAL_METHODS for method in methods)


def _requires_protocol_counts(methods: Sequence[str], modes: Sequence[str]) -> bool:
    return _requires_inference_marginal(methods) or any(
        mode == "imbalanced_full" or mode.startswith("imbalanced")
        for mode in modes
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--radial-audit", type=Path)
    parser.add_argument("--transductive-parameters", required=True, type=Path)
    parser.add_argument("--transductive-audit", required=True, type=Path)
    parser.add_argument("--qpr-parameters", type=Path)
    parser.add_argument("--qpr-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=1)
    parser.add_argument("--sessions", choices=("final", "all"), default="final")
    parser.add_argument("--modes", default="full")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--graph-chunk", type=int, default=512)
    parser.add_argument("--shot-count", type=int, default=5)
    parser.add_argument("--cohort-seed-offset", type=int, default=0)
    parser.add_argument("--skip-file-hash", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(
    function: Callable[[], object], device: str
) -> Tuple[object, float, int]:
    _sync(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    value = function()
    _sync(device)
    elapsed = time.time() - started
    peak = (
        int(torch.cuda.max_memory_allocated())
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    return value, elapsed, peak


def _accuracy(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    num_base: int,
    end_class: int,
) -> Dict[str, object]:
    correct = prediction == labels
    base_mask = labels < int(num_base)
    novel_mask = ~base_mask
    per_class = []
    for class_id in range(int(end_class)):
        mask = labels == class_id
        per_class.append(
            100.0 * float(correct[mask].float().mean().item())
            if bool(mask.any())
            else None
        )
    present = [value for value in per_class if value is not None]
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "class_balanced": float(mean(present)),
        "base": 100.0 * float(correct[base_mask].float().mean().item()),
        "novel": (
            100.0 * float(correct[novel_mask].float().mean().item())
            if bool(novel_mask.any())
            else None
        ),
        "per_class": per_class,
    }


def _support_summary(
    data,
    by_class: Mapping[int, torch.Tensor],
    seed_position: int,
    end_class: int,
    prototype_mode: str,
    shot_count: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(shot_count) not in (1, 2, 5, 10, 20):
        raise ValueError("shot_count must be one of 1,2,5,10,20")
    parts: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for class_id in range(data.num_base):
        center = normalized_mean(by_class[class_id])
        part = center[None, :].repeat(int(shot_count), 1)
        parts.append(part)
        labels.append(
            torch.full(
                (int(shot_count),),
                class_id,
                dtype=torch.long,
                device=data.train_features.device,
            )
        )
    seen_novel = end_class - data.num_base
    if seen_novel:
        rows = (
            data.standard_support_abs[seed_position, :seen_novel].reshape(-1)
            if int(shot_count) == 5
            else data.nested_support_abs[
                seed_position, :seen_novel, : int(shot_count)
            ].reshape(-1)
        )
        novel_features = data.train_features[rows]
        novel_labels = data.train_labels[rows]
        parts.append(novel_features)
        labels.append(novel_labels)
    support = torch.cat(parts, 0)
    support_labels = torch.cat(labels, 0)
    comparison_prototypes = support_prototypes(
        support, support_labels, end_class, normalize=True
    )
    if prototype_mode == "normalized":
        shrink_prototypes = comparison_prototypes
    else:
        if prototype_mode not in PROTOTYPE_MODES:
            raise ValueError("unknown prototype mode: {}".format(prototype_mode))
        centers = [
            prototype_from_samples(by_class[class_id], prototype_mode)
            for class_id in range(data.num_base)
        ]
        if seen_novel:
            centers.extend(
                prototype_from_samples(
                    novel_features[novel_labels == class_id], prototype_mode
                )
                for class_id in range(data.num_base, end_class)
            )
        shrink_prototypes = torch.stack(centers, 0)
    return support, support_labels, comparison_prototypes, shrink_prototypes


def _fixed_imbalanced_subset(
    queries: torch.Tensor,
    labels: torch.Tensor,
    end_class: int,
    protocol_counts: torch.Tensor,
    imbalance_ratio: float = 10.0,
    namespace_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    if float(imbalance_ratio) < 1.0:
        raise ValueError("imbalance_ratio must be at least one")
    rng = np.random.RandomState(
        PROTOCOL_SEED + end_class * 1009 + int(namespace_seed) * 1000003
    )
    ranks = rng.permutation(end_class)
    rank_of = np.empty(end_class, dtype=np.int64)
    rank_of[ranks] = np.arange(end_class)
    maximum = int(protocol_counts[:end_class].max().item())
    desired = []
    selected = []
    for class_id in range(end_class):
        fraction = (1.0 / float(imbalance_ratio)) ** (
            float(rank_of[class_id]) / max(end_class - 1, 1)
        )
        count = max(2, int(round(maximum * fraction)))
        count = min(count, int(protocol_counts[class_id].item()))
        desired.append(count)
        rows = torch.where(labels == class_id)[0]
        if rows.numel() < count:
            raise RuntimeError("protocol imbalance count exceeds available rows")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            PROTOCOL_SEED
            + class_id * 9176
            + end_class
            + int(namespace_seed) * 1000003
        )
        order = torch.randperm(rows.numel(), generator=generator).to(rows.device)
        selected.append(rows[order[:count]])
    index = torch.cat(selected, 0)
    return (
        queries[index],
        labels[index],
        torch.as_tensor(desired, dtype=torch.float32, device=queries.device),
        {
            "construction": "fixed class-id long-tail schedule",
            "imbalance_ratio": float(imbalance_ratio),
            "counts": desired,
            "index_order_uses_labels_for_benchmark_construction_only": True,
        },
    )


def _fixed_balanced_subset(
    queries: torch.Tensor,
    labels: torch.Tensor,
    end_class: int,
    queries_per_class: int,
    namespace_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Construct a label-stratified benchmark; labels never reach inference."""

    if int(queries_per_class) < 1:
        raise ValueError("queries_per_class must be positive")
    selected = []
    counts = []
    for class_id in range(int(end_class)):
        rows = torch.where(labels == class_id)[0]
        count = min(int(queries_per_class), int(rows.numel()))
        if count < 1:
            raise RuntimeError("balanced benchmark is missing class {}".format(class_id))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            PROTOCOL_SEED
            + class_id * 9176
            + int(end_class)
            + int(namespace_seed) * 1000003
        )
        order = torch.randperm(rows.numel(), generator=generator).to(rows.device)
        selected.append(rows[order[:count]])
        counts.append(count)
    index = torch.cat(selected, 0)
    return (
        queries[index],
        labels[index],
        torch.as_tensor(counts, dtype=torch.float32, device=queries.device),
        {
            "construction": "fixed label-stratified evaluation cohort",
            "queries_per_class_cap": int(queries_per_class),
            "counts": counts,
            "labels_used_for_benchmark_construction_only": True,
        },
    )


def _mode_batches(
    mode: str,
    size: int,
    session_seed: int,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, object]]:
    """Return (context indices, retained positions within context) batches."""

    if (
        mode == "full"
        or mode == "imbalanced_full"
        or mode.startswith("imbalanced")
        or mode.startswith("balanced")
    ):
        index = torch.arange(size, device=device)
        return [(index, index)], {"kind": "whole_query_cohort", "batches": 1}
    if mode.startswith("random"):
        batch_size = int(mode.replace("random", ""))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(session_seed)
        order = torch.randperm(size, generator=generator).to(device)
        batches = []
        for start in range(0, size, batch_size):
            context = order[start : start + batch_size]
            retained = torch.arange(context.numel(), device=device)
            batches.append((context, retained))
        return batches, {
            "kind": "feature-independent random non-overlapping batches",
            "batch_size": batch_size,
            "batches": len(batches),
        }
    if mode.startswith("causal"):
        pieces = mode.replace("causal", "").split("x")
        window, stride = int(pieces[0]), int(pieces[1])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(session_seed)
        order = torch.randperm(size, generator=generator).to(device)
        batches = []
        for start in range(0, size, stride):
            stop = min(size, start + stride)
            context_start = max(0, stop - window)
            context = order[context_start:stop]
            retained_count = stop - start
            retained = torch.arange(
                context.numel() - retained_count,
                context.numel(),
                device=device,
            )
            batches.append((context, retained))
        return batches, {
            "kind": "causal sliding cohort",
            "window": window,
            "stride": stride,
            "batches": len(batches),
        }
    raise ValueError("unknown cohort mode {}".format(mode))


def _selected_values(parameters: Mapping[str, object], family: str) -> List[float]:
    return [float(value) for value in parameters["selected"][family]["values"]]


def _add_resource(
    resources: MutableMapping[str, Dict[str, object]],
    method: str,
    elapsed: float,
    peak: int,
    metadata: Mapping[str, object],
    cohort_size: int,
) -> None:
    row = resources[method]
    row["runtime_sec"] = float(row["runtime_sec"]) + float(elapsed)
    row["peak_allocated_bytes"] = max(int(row["peak_allocated_bytes"]), int(peak))
    row["calls"] = int(row["calls"]) + 1
    row["largest_cohort"] = max(int(row["largest_cohort"]), int(cohort_size))
    row["last_metadata"] = dict(metadata)


def _summary(
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for mode, method_rows in rows.items():
        result[mode] = {}
        for method, values in method_rows.items():
            summary: Dict[str, object] = {"evaluations": len(values)}
            for metric in ("overall", "class_balanced", "base", "novel"):
                observed = [
                    float(value[metric])
                    for value in values
                    if value[metric] is not None
                ]
                summary[metric + "_mean"] = (
                    float(mean(observed)) if observed else None
                )
                summary[metric + "_std"] = (
                    float(pstdev(observed)) if observed else None
                )
            result[mode][method] = summary
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods).difference(ALL_METHODS))
    if unknown:
        raise ValueError("unknown methods {}".format(unknown))
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid seed limit")
    if int(args.shot_count) not in (1, 2, 5, 10, 20):
        raise ValueError("--shot-count must be one of 1,2,5,10,20")
    shrink = _load_json(args.shrink_parameters)
    prototype_mode = str(
        shrink.get(
            "deployed_prototype_mode",
            shrink.get("selected_prototype_mode", "normalized"),
        )
    )
    selected_lambda = float(
        shrink.get("deployed_lambda", shrink["selected_lambda"])
    )
    radial_audit = None
    if shrink.get("stage") == "base_selection_radial_shrinksel":
        if args.radial_audit is None:
            raise ValueError("revised ShrinkSel requires --radial-audit")
        radial_audit = _load_json(args.radial_audit)
        if radial_audit["stage"] != "untouched_base_audit_radial_shrinksel":
            raise RuntimeError("invalid revised ShrinkSel audit stage")
        if radial_audit["selection_result_sha256"] != shrink["selection_result_sha256"]:
            raise RuntimeError("revised ShrinkSel selection/audit provenance mismatch")
        prototype_mode, selected_lambda = deployed_radial_choice(
            shrink, radial_audit
        )
    parameters = _load_json(args.transductive_parameters)
    audit = _load_json(args.transductive_audit)
    if shrink["dataset"] != data.dataset or parameters["dataset"] != data.dataset:
        raise RuntimeError("parameter dataset mismatch")
    if shrink["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("shrinkage parameter feature mismatch")
    if parameters["source_feature_file_sha256"] != data.feature_file_sha256:
        raise RuntimeError("transductive parameter feature mismatch")
    if audit["selection_result_sha256"] != parameters["selection_result_sha256"]:
        raise RuntimeError("transductive selection/audit provenance mismatch")
    if not bool(audit["passes_numerical_sanity_gate"]):
        raise RuntimeError("transductive base audit failed")
    if (
        "protolp_paper" in methods
        or "protolp_uniform" in methods
        or "protolp_paper_known_marginal" in methods
    ):
        if int(parameters.get("schema_version", 0)) < 4:
            raise RuntimeError(
                "ProtoLP requires regenerated schema-v4 base selection/audit"
            )
        required_protolp_families = []
        if (
            "protolp_paper" in methods
            or "protolp_paper_known_marginal" in methods
        ):
            required_protolp_families.append("protolp_paper")
        if "protolp_uniform" in methods:
            required_protolp_families.append("protolp_uniform")
        for family in required_protolp_families:
            values = _selected_values(parameters, family)
            if len(values) != 2:
                raise RuntimeError(
                    family
                    + " must transfer base-selected graph regularization and "
                    "CLIP assignment scale"
                )
    if (
        "laplacian_shot" in methods or "laplacian_shot_core" in methods
    ) and int(parameters.get("schema_version", 0)) < 7:
        raise RuntimeError(
            "LaplacianShot variants require regenerated schema-v7 base selection/audit"
        )
    qpr_parameters = None
    qpr_audit = None
    if QPR_METHOD in methods:
        if args.qpr_parameters is None or args.qpr_audit is None:
            raise ValueError("qpr requires --qpr-parameters and --qpr-audit")
        qpr_parameters = _load_json(args.qpr_parameters)
        qpr_audit = _load_json(args.qpr_audit)
        if qpr_parameters["dataset"] != data.dataset:
            raise RuntimeError("QPR parameter dataset mismatch")
        if qpr_audit["dataset"] != data.dataset:
            raise RuntimeError("QPR audit dataset mismatch")
        if qpr_audit["stage"] != "qpr_untouched_base_audit":
            raise RuntimeError("invalid QPR audit stage")
        if qpr_parameters["source_feature_file_sha256"] != data.selection_source_sha256:
            raise RuntimeError("QPR parameter feature mismatch")
        if qpr_parameters["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
            raise RuntimeError("QPR was not selected for this ShrinkSel head")
        parent_mode = str(shrink.get("selected_prototype_mode", "normalized"))
        parent_lambda = float(shrink["selected_lambda"])
        if qpr_parameters.get("selected_prototype_mode", "normalized") != parent_mode:
            raise RuntimeError("QPR prototype mode disagrees with ShrinkSel")
        if float(qpr_parameters["selected_lambda"]) != parent_lambda:
            raise RuntimeError("QPR geometry disagrees with ShrinkSel")
        if qpr_audit["source_feature_file_sha256"] != data.selection_source_sha256:
            raise RuntimeError("QPR audit feature mismatch")
        deployed_qpr_choice(qpr_parameters, qpr_audit)
    needs_inference_marginal = _requires_inference_marginal(methods)
    needs_protocol_counts = _requires_protocol_counts(methods, modes)
    if needs_protocol_counts and data.known_test_counts is None:
        raise RuntimeError(
            "controlled imbalance or prior-assisted diagnostics need a protocol vector"
        )

    by_class = features_by_class(data)
    laplacian_train_mean = data.train_features[
        data.train_labels < data.num_base
    ].mean(0)
    covariance = covariance_eigensystem(by_class, range(data.num_base))
    laplacian_values = (
        _selected_values(parameters, "laplacian")
        if "laplacian_shot" in methods
        else ()
    )
    if "laplacian_shot" in methods and len(laplacian_values) != 3:
        raise RuntimeError(
            "LaplacianShot must transfer base-selected k, lambda, and optional "
            "prototype-rectification switch"
        )
    laplacian_core_values = (
        _selected_values(parameters, "laplacian_core")
        if "laplacian_shot_core" in methods
        else ()
    )
    if "laplacian_shot_core" in methods and len(laplacian_core_values) != 2:
        raise RuntimeError(
            "LaplacianShot-core must transfer base-selected k and lambda"
        )
    ptmap_values = (
        _selected_values(parameters, "ptmap")
        if "ptmap_uniform" in methods or "ptmap_known_marginal" in methods
        else ()
    )
    protolp_paper_values = (
        _selected_values(parameters, "protolp_paper")
        if "protolp_paper" in methods
        or "protolp_paper_known_marginal" in methods
        else ()
    )
    protolp_uniform_values = (
        _selected_values(parameters, "protolp_uniform")
        if "protolp_uniform" in methods
        else ()
    )
    graph_values = (
        _selected_values(parameters, "graph")
        if "sparse_graph_repair" in methods
        else ()
    )
    tim_parameters = parameters["tim"]
    number_sessions = 1 + (data.num_total - data.num_base) // data.increment
    session_indices = (
        [number_sessions - 1]
        if args.sessions == "final"
        else list(range(number_sessions))
    )
    evaluation_rows: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        mode: {method: [] for method in methods} for mode in modes
    }
    per_seed: List[Dict[str, object]] = []
    resources: Dict[str, Dict[str, object]] = {
        method: {
            "runtime_sec": 0.0,
            "peak_allocated_bytes": 0,
            "calls": 0,
            "largest_cohort": 0,
            "last_metadata": {},
        }
        for method in methods
    }

    for seed_position in range(args.seed_limit):
        seed_row: Dict[str, object] = {
            "support_seed": data.support_seeds[seed_position],
            "sessions": [],
        }
        for session_index in session_indices:
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            query_mask = data.test_labels < end_class
            standard_queries = data.test_features[query_mask]
            standard_labels = data.test_labels[query_mask]
            support, support_labels, comparison_prototypes, shrink_prototypes = _support_summary(
                data,
                by_class,
                seed_position,
                end_class,
                prototype_mode,
                shot_count=int(args.shot_count),
            )
            session_row: Dict[str, object] = {
                "session_index": session_index,
                "end_class": end_class,
                "modes": {},
            }
            for mode in modes:
                queries = standard_queries
                labels = standard_labels
                inference_mass = (
                    data.known_test_counts[:end_class].clone()
                    if needs_inference_marginal
                    else torch.empty(0, device=queries.device)
                )
                mode_metadata: Dict[str, object] = {}
                if mode == "imbalanced_full" or mode.startswith("imbalanced"):
                    assert data.known_test_counts is not None
                    suffix = mode.replace("imbalanced", "").replace("_full", "")
                    imbalance_ratio = float(suffix) if suffix else 10.0
                    queries, labels, constructed_mass, mode_metadata = _fixed_imbalanced_subset(
                        queries,
                        labels,
                        end_class,
                        data.known_test_counts,
                        imbalance_ratio=imbalance_ratio,
                        namespace_seed=int(args.cohort_seed_offset),
                    )
                    if needs_inference_marginal:
                        inference_mass = constructed_mass
                elif mode.startswith("balanced"):
                    queries_per_class = int(mode.replace("balanced", ""))
                    queries, labels, constructed_mass, mode_metadata = _fixed_balanced_subset(
                        queries,
                        labels,
                        end_class,
                        queries_per_class,
                        namespace_seed=int(args.cohort_seed_offset),
                    )
                    if needs_inference_marginal:
                        inference_mass = constructed_mass
                batches, batching = _mode_batches(
                    mode,
                    queries.shape[0],
                    PROTOCOL_SEED
                    + data.support_seeds[seed_position] * 10007
                    + session_index * 1009
                    + int(args.cohort_seed_offset) * 1000003,
                    queries.device,
                )
                predictions = {
                    method: torch.full(
                        (queries.shape[0],), -1, dtype=torch.long, device=queries.device
                    )
                    for method in methods
                }
                fallbacks = {method: 0 for method in methods}
                for context_index, retained in batches:
                    cohort = queries[context_index]
                    output_positions = context_index[retained]

                    def record(
                        method: str,
                        function: Callable[[], TransductiveOutput],
                    ) -> TransductiveOutput:
                        value, elapsed, peak = _measure(function, args.device)
                        assert isinstance(value, TransductiveOutput)
                        _add_resource(
                            resources,
                            method,
                            elapsed,
                            peak,
                            value.metadata,
                            cohort.shape[0],
                        )
                        predictions[method][output_positions] = value.scores[
                            retained
                        ].argmax(1)
                        return value

                    cosine_scores = None
                    if "cosine" in methods:
                        value, elapsed, peak = _measure(
                            lambda: cohort.matmul(comparison_prototypes.t()), args.device
                        )
                        cosine_scores = value
                        _add_resource(
                            resources,
                            "cosine",
                            elapsed,
                            peak,
                            {"method": "cosine-NCM", "batch_dependent": False},
                            cohort.shape[0],
                        )
                        predictions["cosine"][output_positions] = value[
                            retained
                        ].argmax(1)
                    shrink_scores = None
                    shrink_elapsed = 0.0
                    shrink_peak = 0
                    if (
                        "shrinksel" in methods
                        or "shrinksel_known_marginal" in methods
                        or QPR_METHOD in methods
                    ):
                        value, elapsed, peak = _measure(
                            lambda: metric_scores(
                                cohort,
                                shrink_prototypes,
                                covariance,
                                selected_lambda,
                                RIDGE_GAMMA,
                            ),
                            args.device,
                        )
                        shrink_scores = value
                        shrink_elapsed = elapsed
                        shrink_peak = peak
                        if "shrinksel" in methods:
                            _add_resource(
                                resources,
                                "shrinksel",
                                elapsed,
                                peak,
                                {"method": "ShrinkSel", "batch_dependent": False},
                                cohort.shape[0],
                            )
                            predictions["shrinksel"][output_positions] = value[
                                retained
                            ].argmax(1)
                    if QPR_METHOD in methods:
                        assert shrink_scores is not None
                        if end_class == data.num_base:
                            predictions[QPR_METHOD][output_positions] = shrink_scores[
                                retained
                            ].argmax(1)
                            _add_resource(
                                resources,
                                QPR_METHOD,
                                shrink_elapsed,
                                shrink_peak,
                                {
                                    "method": "ShrinkSel + QPR",
                                    "base_session_identity": True,
                                    "uses_query_labels": False,
                                    "uses_query_marginal_constraint": False,
                                },
                                cohort.shape[0],
                            )
                        else:
                            assert qpr_parameters is not None
                            assert qpr_audit is not None
                            selected_qpr = deployed_qpr_choice(
                                qpr_parameters, qpr_audit
                            )
                            if (
                                radial_audit is not None
                                and not radial_audit["passes_non_regression_gate"]
                            ):
                                selected_qpr = {
                                    **selected_qpr,
                                    "alpha": 0.0,
                                }

                            def apply_qpr() -> TransductiveOutput:
                                proposal = query_prototype_rectification(
                                    cohort,
                                    shrink_prototypes,
                                    shrink_scores,
                                    data.num_base,
                                    alpha=float(selected_qpr["alpha"]),
                                    mass_threshold=float(
                                        selected_qpr["mass_threshold"]
                                    ),
                                    min_count=int(selected_qpr["min_count"]),
                                    logit_scale=float(selected_qpr["logit_scale"]),
                                )
                                return TransductiveOutput(
                                    scores=metric_scores(
                                        cohort,
                                        proposal.prototypes,
                                        covariance,
                                        selected_lambda,
                                        RIDGE_GAMMA,
                                    ),
                                    prototypes=proposal.prototypes,
                                    metadata={
                                        **proposal.metadata,
                                        "method": "ShrinkSel + QPR",
                                        "rescored_with_locked_shrinksel_geometry": True,
                                    },
                                )

                            qpr_value, qpr_elapsed, qpr_peak = _measure(
                                apply_qpr, args.device
                            )
                            assert isinstance(qpr_value, TransductiveOutput)
                            predictions[QPR_METHOD][output_positions] = qpr_value.scores[
                                retained
                            ].argmax(1)
                            _add_resource(
                                resources,
                                QPR_METHOD,
                                shrink_elapsed + qpr_elapsed,
                                max(shrink_peak, qpr_peak),
                                qpr_value.metadata,
                                cohort.shape[0],
                            )
                    if "tim_gd" in methods:
                        record(
                            "tim_gd",
                            lambda: tim_gd(
                                support,
                                support_labels,
                                cohort,
                                end_class,
                                iterations=int(tim_parameters["iterations"]),
                                temperature=float(tim_parameters["temperature"]),
                                learning_rate=float(tim_parameters["learning_rate"]),
                                cross_entropy_weight=float(tim_parameters["loss_weights"][0]),
                                marginal_entropy_weight=float(tim_parameters["loss_weights"][1]),
                                conditional_entropy_weight=float(tim_parameters["loss_weights"][2]),
                            ),
                        )
                    if "laplacian_shot" in methods:
                        if cohort.shape[0] < 2:
                            if cosine_scores is None:
                                cosine_scores = cohort.matmul(comparison_prototypes.t())
                            predictions["laplacian_shot"][output_positions] = cosine_scores[
                                retained
                            ].argmax(1)
                            fallbacks["laplacian_shot"] += 1
                        else:
                            record(
                                "laplacian_shot",
                                lambda: laplacian_shot_full(
                                    support,
                                    support_labels,
                                    cohort,
                                    end_class,
                                    train_mean=laplacian_train_mean,
                                    knn=int(laplacian_values[0]),
                                    regularization=float(laplacian_values[1]),
                                    prototype_rectification=bool(
                                        int(laplacian_values[2])
                                    ),
                                    chunk_size=args.graph_chunk,
                                ),
                            )
                    if "laplacian_shot_core" in methods:
                        if cohort.shape[0] < 2:
                            if cosine_scores is None:
                                cosine_scores = cohort.matmul(
                                    comparison_prototypes.t()
                                )
                            predictions["laplacian_shot_core"][
                                output_positions
                            ] = cosine_scores[retained].argmax(1)
                            fallbacks["laplacian_shot_core"] += 1
                        else:
                            record(
                                "laplacian_shot_core",
                                lambda: laplacian_shot(
                                    comparison_prototypes,
                                    cohort,
                                    neighbors=int(laplacian_core_values[0]),
                                    regularization=float(
                                        laplacian_core_values[1]
                                    ),
                                    chunk_size=args.graph_chunk,
                                ),
                            )
                    if "ptmap_uniform" in methods:
                        record(
                            "ptmap_uniform",
                            lambda: pt_map(
                                support,
                                support_labels,
                                cohort,
                                end_class,
                                assignment_scale=float(ptmap_values[0]),
                            ),
                        )
                    proto_paper_output = None
                    proto_paper_elapsed = 0.0
                    proto_paper_peak = 0
                    if (
                        "protolp_paper" in methods
                        or "protolp_paper_known_marginal" in methods
                    ):
                        value, elapsed, peak = _measure(
                            lambda: protolp_paper(
                                support,
                                support_labels,
                                cohort,
                                end_class,
                                graph_regularization=float(
                                    protolp_paper_values[0]
                                ),
                                assignment_scale=float(
                                    protolp_paper_values[1]
                                ),
                            ),
                            args.device,
                        )
                        proto_paper_output = value
                        proto_paper_elapsed = elapsed
                        proto_paper_peak = peak
                        if "protolp_paper" in methods:
                            _add_resource(
                                resources,
                                "protolp_paper",
                                elapsed,
                                peak,
                                value.metadata,
                                cohort.shape[0],
                            )
                            predictions["protolp_paper"][
                                output_positions
                            ] = value.scores[retained].argmax(1)
                    if "protolp_uniform" in methods:
                        record(
                            "protolp_uniform",
                            lambda: protolp_uniform(
                                support,
                                support_labels,
                                cohort,
                                end_class,
                                graph_regularization=float(
                                    protolp_uniform_values[0]
                                ),
                                assignment_scale=float(
                                    protolp_uniform_values[1]
                                ),
                            ),
                        )
                    if "sparse_graph_repair" in methods:
                        record(
                            "sparse_graph_repair",
                            lambda: sparse_graph_repair(
                                comparison_prototypes,
                                cohort,
                                neighbors=int(graph_values[0]),
                                prototype_anchor=float(graph_values[1]),
                                chunk_size=args.graph_chunk,
                            ),
                        )
                    if "shrinksel_known_marginal" in methods:
                        assert shrink_scores is not None
                        value, elapsed, peak = _measure(
                            lambda: marginal_constrained_scores(
                                shrink_scores, inference_mass
                            ),
                            args.device,
                        )
                        _add_resource(
                            resources,
                            "shrinksel_known_marginal",
                            shrink_elapsed + elapsed,
                            max(shrink_peak, peak),
                            {
                                "method": "ShrinkSel + protocol-marginal decoder",
                                "information_level": "I3",
                                "runtime_includes_base_shrinksel": True,
                            },
                            cohort.shape[0],
                        )
                        predictions["shrinksel_known_marginal"][output_positions] = value[
                            retained
                        ].argmax(1)
                    if "protolp_paper_known_marginal" in methods:
                        assert proto_paper_output is not None
                        value, elapsed, peak = _measure(
                            lambda: marginal_constrained_scores(
                                proto_paper_output.scores, inference_mass
                            ),
                            args.device,
                        )
                        _add_resource(
                            resources,
                            "protolp_paper_known_marginal",
                            proto_paper_elapsed + elapsed,
                            max(proto_paper_peak, peak),
                            {
                                "method": (
                                    "ProtoLP paper equations + "
                                    "protocol-marginal decoder"
                                ),
                                "information_level": "I3",
                                "runtime_includes_base_protolp": True,
                            },
                            cohort.shape[0],
                        )
                        predictions["protolp_paper_known_marginal"][
                            output_positions
                        ] = value[retained].argmax(1)
                    if "ptmap_known_marginal" in methods:
                        record(
                            "ptmap_known_marginal",
                            lambda: pt_map(
                                support,
                                support_labels,
                                cohort,
                                end_class,
                                assignment_scale=float(ptmap_values[0]),
                                class_mass=inference_mass,
                            ),
                        )
                mode_result: Dict[str, object] = {
                    "query_count": int(queries.shape[0]),
                    "batching": batching,
                    "inference_marginal_source": (
                        "committed protocol constants; never constructed from runtime labels"
                        if needs_inference_marginal
                        else None
                    ),
                    "inference_marginal": (
                        inference_mass.tolist() if needs_inference_marginal else None
                    ),
                    "evaluation_counts_are_not_inference_inputs": True,
                    "construction": mode_metadata,
                    "fallback_calls": fallbacks,
                    "methods": {},
                }
                for method in methods:
                    if bool((predictions[method] < 0).any()):
                        raise RuntimeError("incomplete predictions for {}".format(method))
                    values = _accuracy(
                        predictions[method], labels, data.num_base, end_class
                    )
                    mode_result["methods"][method] = values
                    evaluation_rows[mode][method].append(values)
                session_row["modes"][mode] = mode_result
            seed_row["sessions"].append(session_row)
        per_seed.append(seed_row)

    result: Dict[str, object] = {
        "schema_version": 3,
        "stage": "target_transductive_evaluation_after_protocol_lock",
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "information_levels": {
            "cosine": "I0/I1 strict inductive",
            "shrinksel": "I0/I1 strict inductive",
            "tim_gd": "I2 unlabeled query cohort",
            "laplacian_shot": "I2 unlabeled query cohort",
            "laplacian_shot_core": (
                "I2 unlabeled query cohort; graph/bound-update core on the "
                "matched frozen normalized features"
            ),
            "ptmap_uniform": "I2 plus assumed uniform class marginal",
            "protolp_paper": (
                "I2 unlabeled query cohort; primary ProtoLP Eqs. 4 and 9--17 "
                "without a class marginal"
            ),
            "protolp_uniform": (
                "I2 plus ProtoLP's optional uniform-Sinkhorn prior (Eq. 18)"
            ),
            "sparse_graph_repair": "I2 unlabeled query cohort",
            "qpr": (
                "I2 unlabeled query cohort; QPR refines only novel prototypes "
                "from the locked ShrinkSel posterior"
            ),
            "shrinksel_known_marginal": "I3 committed class marginal",
            "protolp_paper_known_marginal": (
                "I3 committed class marginal"
            ),
            "ptmap_known_marginal": "I3 committed class marginal",
        },
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "shrinksel_prototype_mode": prototype_mode,
        "shrinksel_lambda": selected_lambda,
        "shrinksel_fallback_deployed": bool(
            radial_audit is not None
            and not radial_audit["passes_non_regression_gate"]
        ),
        "shot_count": int(args.shot_count),
        "cohort_seed_offset": int(args.cohort_seed_offset),
        "selection_provenance": {
            "shrinksel": shrink["selection_result_sha256"],
            "shrinksel_audit": (
                radial_audit["audit_result_sha256"]
                if radial_audit is not None
                else None
            ),
            "transductive": parameters["selection_result_sha256"],
            "transductive_audit": audit["audit_result_sha256"],
            "qpr": (
                qpr_parameters["selection_result_sha256"]
                if qpr_parameters is not None
                else None
            ),
            "qpr_audit": (
                qpr_audit["audit_result_sha256"]
                if qpr_audit is not None
                else None
            ),
        },
        "methods": list(methods),
        "modes": list(modes),
        "session_scope": args.sessions,
        "support_seeds": data.support_seeds[: args.seed_limit],
        "per_seed": per_seed,
        "summary": _summary(evaluation_rows),
        "resources": resources,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "summary": result["summary"],
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
