#!/usr/bin/env python3
"""Component, sensitivity, and failure diagnostics for locked QPR.

All variants are post-lock removals or one-at-a-time diagnostics. Target labels and
complete novel means are consumed only after inference to calculate failure metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from .constants import RIDGE_GAMMA
from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import prototype_from_samples
from .run_qpr import parameters_from_mapping
from .run_transductive import _accuracy, _measure, _support_summary
from .target_data import features_by_class, load_target_evaluation_data
from .transductive_methods import query_prototype_rectification
from .utils import atomic_write_json, backbone_display, stable_json_sha256


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config(
    alpha: float,
    mass_threshold: float,
    min_count: int,
    logit_scale: float,
    *,
    posterior_weighting: bool = True,
    preserve_radius: bool = True,
    require_predicted_novel: bool = True,
) -> Dict[str, object]:
    return {
        "alpha": float(alpha),
        "mass_threshold": float(mass_threshold),
        "min_count": int(min_count),
        "logit_scale": float(logit_scale),
        "posterior_weighting": bool(posterior_weighting),
        "preserve_radius": bool(preserve_radius),
        "require_predicted_novel": bool(require_predicted_novel),
    }


def component_configurations(
    selected: Sequence[float],
) -> Dict[str, Dict[str, object]]:
    alpha, mass, minimum, scale = selected
    return {
        "qpr_selected": _config(alpha, mass, int(minimum), scale),
        "qpr_uniform_weights": _config(
            alpha,
            mass,
            int(minimum),
            scale,
            posterior_weighting=False,
        ),
        "qpr_no_mass_gate": _config(alpha, 0.0, int(minimum), scale),
        "qpr_min_count_one": _config(alpha, mass, 1, scale),
        "qpr_no_radius_preservation": _config(
            alpha,
            mass,
            int(minimum),
            scale,
            preserve_radius=False,
        ),
        "qpr_no_predicted_novel_gate": _config(
            alpha,
            mass,
            int(minimum),
            scale,
            require_predicted_novel=False,
        ),
        "qpr_full_step": _config(1.0, mass, int(minimum), scale),
        "qpr_aggressive_no_conservative_gates": _config(
            1.0,
            0.0,
            1,
            scale,
            posterior_weighting=False,
            preserve_radius=False,
            require_predicted_novel=False,
        ),
    }


def sensitivity_configurations(
    selected: Sequence[float],
) -> Dict[str, Dict[str, object]]:
    alpha, mass, minimum, scale = selected
    result: Dict[str, Dict[str, object]] = {}
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        result["alpha={}".format(value)] = _config(
            value, mass, int(minimum), scale
        )
    for value in (0.0, 0.3, 0.6, 0.9):
        result["mass={}".format(value)] = _config(
            alpha, value, int(minimum), scale
        )
    for value in (1, 5, 10, 20):
        result["min_count={}".format(value)] = _config(alpha, mass, value, scale)
    for value in (0.01, 0.03, 0.1, 0.3, 1.0):
        result["scale={}".format(value)] = _config(
            alpha, mass, int(minimum), value
        )
    return result


def _gate_mask(
    scores: torch.Tensor,
    num_base: int,
    config: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.softmax(
        scores.float() * float(config["logit_scale"]), dim=1
    )
    prediction = probability.argmax(1)
    novel = probability[:, int(num_base) :]
    novel_mass = novel.sum(1)
    assigned = novel.argmax(1) + int(num_base)
    conditional = novel / novel_mass[:, None].clamp_min(1e-12)
    if conditional.shape[1] > 1:
        top_two = conditional.topk(2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
    else:
        margin = torch.ones_like(novel_mass)
    gate = novel_mass >= float(config["mass_threshold"])
    if bool(config["require_predicted_novel"]):
        gate = gate & (prediction >= int(num_base))
    return gate, assigned, probability, margin


def _angle_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = (F.normalize(left.float(), dim=1) * F.normalize(right.float(), dim=1)).sum(1)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _diagnostics(
    baseline_prediction: torch.Tensor,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    original: torch.Tensor,
    refined: torch.Tensor,
    full_reference: torch.Tensor,
    num_base: int,
    config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    baseline_correct = baseline_prediction == labels
    refined_correct = prediction == labels
    base = labels < int(num_base)
    novel = ~base
    repaired = (~baseline_correct) & refined_correct
    harmed = baseline_correct & (~refined_correct)
    gate, assigned, _, margin = _gate_mask(scores, num_base, config)
    gated = int(gate.sum().item())
    gated_novel = int((gate & novel).sum().item())
    assignment_correct = int((gate & (assigned == labels)).sum().item())
    before_error = _angle_degrees(
        original[int(num_base) :], full_reference[int(num_base) :]
    )
    after_error = _angle_degrees(
        refined[int(num_base) :], full_reference[int(num_base) :]
    )
    displacement = _angle_degrees(
        original[int(num_base) :], refined[int(num_base) :]
    )
    updated_ids = [int(value) for value in metadata["updated_class_ids"]]

    # Base-side attribution of the operator's damage.  Every base example that
    # was correct before rectification and is now claimed by a novel prototype
    # is recorded as a (true base class -> capturing novel class) defection, so
    # the harm can be traced to specific prototypes rather than only counted.
    # Labels are read here only after inference, as elsewhere in this function.
    class_total = int(original.shape[0])
    captured_mask = base & baseline_correct & (prediction >= int(num_base))
    base_defections: List[Dict[str, int]] = []
    base_loss_per_class: List[Dict[str, int]] = []
    novel_capture_per_class: List[Dict[str, int]] = []
    if bool(captured_mask.any()):
        source = labels[captured_mask].to(torch.int64)
        destination = prediction[captured_mask].to(torch.int64)
        pair_key = source * class_total + destination
        keys, counts = torch.unique(pair_key, return_counts=True)
        for key, count in zip(keys.tolist(), counts.tolist()):
            base_defections.append(
                {
                    "base_class": int(key) // class_total,
                    "novel_class": int(key) % class_total,
                    "count": int(count),
                }
            )
        source_ids, source_counts = torch.unique(source, return_counts=True)
        for class_id, count in zip(source_ids.tolist(), source_counts.tolist()):
            base_loss_per_class.append(
                {"class_id": int(class_id), "captured": int(count)}
            )
        target_ids, target_counts = torch.unique(destination, return_counts=True)
        for class_id, count in zip(target_ids.tolist(), target_counts.tolist()):
            novel_capture_per_class.append(
                {"class_id": int(class_id), "captured": int(count)}
            )

    per_class = []
    for class_id in range(int(num_base), int(original.shape[0])):
        class_mask = labels == class_id
        admitted = gate & (assigned == class_id)
        class_position = class_id - int(num_base)
        per_class.append(
            {
                "class_id": class_id,
                "query_count": int(class_mask.sum().item()),
                "accuracy_before": 100.0
                * float(baseline_correct[class_mask].float().mean().item()),
                "accuracy_after": 100.0
                * float(refined_correct[class_mask].float().mean().item()),
                "repaired": int((repaired & class_mask).sum().item()),
                "harmed": int((harmed & class_mask).sum().item()),
                "admitted": int(admitted.sum().item()),
                "admission_purity": (
                    100.0
                    * float((labels[admitted] == class_id).float().mean().item())
                    if bool(admitted.any())
                    else None
                ),
                "prototype_angle_before": float(before_error[class_position].item()),
                "prototype_angle_after": float(after_error[class_position].item()),
                "prototype_displacement": float(displacement[class_position].item()),
                "updated": class_id in updated_ids,
            }
        )
    return {
        "repaired": int(repaired.sum().item()),
        "harmed": int(harmed.sum().item()),
        "base_repaired": int((repaired & base).sum().item()),
        "base_harmed": int((harmed & base).sum().item()),
        "novel_repaired": int((repaired & novel).sum().item()),
        "novel_harmed": int((harmed & novel).sum().item()),
        "base_correct_captured_by_novel": int(
            (base & baseline_correct & (prediction >= int(num_base))).sum().item()
        ),
        "gated_query_count": gated,
        "gate_group_purity": 100.0 * gated_novel / gated if gated else 100.0,
        "gate_assignment_purity": (
            100.0 * assignment_correct / gated if gated else 100.0
        ),
        "gated_novel_margin_mean": (
            float(margin[gate].mean().item()) if gated else 1.0
        ),
        "eligible_class_count": int(metadata["eligible_class_count"]),
        "updated_class_count": int(metadata["updated_class_count"]),
        "prototype_angle_before_mean": float(before_error.mean().item()),
        "prototype_angle_after_mean": float(after_error.mean().item()),
        "prototype_displacement_mean": float(displacement.mean().item()),
        "per_class": per_class,
        "base_defections": base_defections,
        "base_loss_per_class": base_loss_per_class,
        "novel_capture_per_class": novel_capture_per_class,
        "labels_and_full_means_used_after_inference_only": True,
    }


def _aggregate(
    per_seed: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for method in methods:
        session_average = []
        final = {metric: [] for metric in ("overall", "base", "novel")}
        for seed in per_seed:
            sessions = seed["sessions"]
            session_average.append(
                float(mean(float(row["methods"][method]["overall"]) for row in sessions))
            )
            for metric in final:
                final[metric].append(float(sessions[-1]["methods"][method][metric]))
        result[method] = {
            "session_average_mean": float(mean(session_average)),
            "session_average_std": float(pstdev(session_average)),
            "session_average_per_seed": session_average,
            "final": {
                metric: {
                    "mean": float(mean(values)),
                    "std": float(pstdev(values)),
                    "per_seed": values,
                }
                for metric, values in final.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shrink-parameters", required=True, type=Path)
    parser.add_argument("--qpr-parameters", required=True, type=Path)
    parser.add_argument("--qpr-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", choices=("components", "sensitivity"), required=True)
    parser.add_argument("--sessions", choices=("final", "all"), default="final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--skip-file-hash", action="store_true")
    args = parser.parse_args()

    started = time.time()
    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    if args.seed_limit < 1 or args.seed_limit > len(data.support_seeds):
        raise ValueError("invalid --seed-limit")
    shrink = _load(args.shrink_parameters)
    qpr = _load(args.qpr_parameters)
    audit = _load(args.qpr_audit)
    for payload, name in ((shrink, "ShrinkSel"), (qpr, "QPR"), (audit, "QPR audit")):
        if payload["dataset"] != data.dataset:
            raise RuntimeError(name + " dataset mismatch")
    if shrink["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("ShrinkSel feature mismatch")
    if qpr["source_feature_file_sha256"] != data.selection_source_sha256:
        raise RuntimeError("QPR feature mismatch")
    if audit["selection_result_sha256"] != qpr["selection_result_sha256"]:
        raise RuntimeError("QPR selection/audit provenance mismatch")
    if not bool(audit["passes_transfer_safety_gate"]):
        raise RuntimeError("QPR failed its untouched base audit")
    if qpr["shrink_selection_result_sha256"] != shrink["selection_result_sha256"]:
        raise RuntimeError("QPR was not selected for this ShrinkSel head")

    selected = parameters_from_mapping(qpr["selected"])
    configurations = (
        component_configurations(selected)
        if args.suite == "components"
        else sensitivity_configurations(selected)
    )
    methods = ("shrinksel",) + tuple(configurations)
    prototype_mode = str(shrink.get("selected_prototype_mode", "normalized"))
    selected_lambda = float(shrink["selected_lambda"])
    by_class = features_by_class(data)
    covariance = covariance_eigensystem(by_class, range(data.num_base))
    session_count = 1 + (data.num_total - data.num_base) // data.increment
    session_indices = (
        [session_count - 1] if args.sessions == "final" else list(range(session_count))
    )
    per_seed = []
    resources: MutableMapping[str, Dict[str, object]] = {
        method: {"runtime_sec": 0.0, "peak_allocated_bytes": 0, "calls": 0}
        for method in methods
    }

    for seed_position in range(args.seed_limit):
        seed_row: Dict[str, object] = {
            "support_seed": int(data.support_seeds[seed_position]),
            "sessions": [],
        }
        for session_index in session_indices:
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            mask = data.test_labels < end_class
            query = data.test_features[mask]
            labels = data.test_labels[mask]
            _, _, _, prototypes = _support_summary(
                data,
                by_class,
                seed_position,
                end_class,
                prototype_mode,
                shot_count=5,
            )
            full_reference = torch.stack(
                [
                    prototype_from_samples(by_class[class_id], prototype_mode)
                    for class_id in range(end_class)
                ]
            )
            baseline_scores, elapsed, peak = _measure(
                lambda: metric_scores(
                    query,
                    prototypes,
                    covariance,
                    selected_lambda,
                    RIDGE_GAMMA,
                    args.query_chunk,
                ),
                args.device,
            )
            baseline_prediction = baseline_scores.argmax(1)
            resources["shrinksel"]["runtime_sec"] += elapsed
            resources["shrinksel"]["peak_allocated_bytes"] = max(
                int(resources["shrinksel"]["peak_allocated_bytes"]), int(peak)
            )
            resources["shrinksel"]["calls"] += 1
            session_row: Dict[str, object] = {
                "session_index": int(session_index),
                "end_class": int(end_class),
                "query_count": int(query.shape[0]),
                "methods": {
                    "shrinksel": _accuracy(
                        baseline_prediction, labels, data.num_base, end_class
                    )
                },
                "diagnostics": {},
            }
            for method, config in configurations.items():
                if end_class == data.num_base or float(config["alpha"]) == 0.0:
                    prediction = baseline_prediction
                    metrics = _accuracy(prediction, labels, data.num_base, end_class)
                    session_row["methods"][method] = metrics
                    session_row["diagnostics"][method] = {
                        "base_session_or_zero_alpha_noop": True,
                        "repaired": 0,
                        "harmed": 0,
                        "updated_class_count": 0,
                    }
                    resources[method]["runtime_sec"] += elapsed
                    resources[method]["peak_allocated_bytes"] = max(
                        int(resources[method]["peak_allocated_bytes"]), int(peak)
                    )
                    resources[method]["calls"] += 1
                    continue

                def apply():
                    proposal = query_prototype_rectification(
                        query,
                        prototypes,
                        baseline_scores,
                        data.num_base,
                        alpha=float(config["alpha"]),
                        mass_threshold=float(config["mass_threshold"]),
                        min_count=int(config["min_count"]),
                        logit_scale=float(config["logit_scale"]),
                        posterior_weighting=bool(config["posterior_weighting"]),
                        preserve_radius=bool(config["preserve_radius"]),
                        require_predicted_novel=bool(
                            config["require_predicted_novel"]
                        ),
                    )
                    final_scores = metric_scores(
                        query,
                        proposal.prototypes,
                        covariance,
                        selected_lambda,
                        RIDGE_GAMMA,
                        args.query_chunk,
                    )
                    return proposal, final_scores

                value, qpr_elapsed, qpr_peak = _measure(apply, args.device)
                proposal, final_scores = value
                prediction = final_scores.argmax(1)
                session_row["methods"][method] = _accuracy(
                    prediction, labels, data.num_base, end_class
                )
                diagnostics = _diagnostics(
                    baseline_prediction,
                    prediction,
                    labels,
                    baseline_scores,
                    prototypes,
                    proposal.prototypes,
                    full_reference,
                    data.num_base,
                    config,
                    proposal.metadata,
                )
                if args.suite == "sensitivity":
                    diagnostics.pop("per_class")
                session_row["diagnostics"][method] = diagnostics
                resources[method]["runtime_sec"] += elapsed + qpr_elapsed
                resources[method]["peak_allocated_bytes"] = max(
                    int(resources[method]["peak_allocated_bytes"]),
                    int(peak),
                    int(qpr_peak),
                )
                resources[method]["calls"] += 1
            seed_row["sessions"].append(session_row)
        per_seed.append(seed_row)
        print(
            "dataset={} suite={} seed={} shrinksel={:.3f} selected={:.3f}".format(
                data.dataset,
                args.suite,
                data.support_seeds[seed_position],
                mean(
                    float(row["methods"]["shrinksel"]["overall"])
                    for row in seed_row["sessions"]
                ),
                mean(
                    float(row["methods"]["qpr_selected"]["overall"])
                    for row in seed_row["sessions"]
                )
                if "qpr_selected" in configurations
                else float("nan"),
            ),
            flush=True,
        )

    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "post_lock_qpr_{}_ablation".format(args.suite),
        "dataset": data.dataset,
        "backbone": backbone_display(dict(data.backbone)),
        "backbone_metadata": dict(data.backbone),
        "feature_file_sha256": data.feature_file_sha256,
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "selection_provenance": {
            "shrinksel": shrink["selection_result_sha256"],
            "qpr": qpr["selection_result_sha256"],
            "qpr_audit": audit["audit_result_sha256"],
        },
        "information_boundary": {
            "query_labels_available_to_inference": False,
            "query_marginal_available_to_inference": False,
            "labels_and_full_novel_means_used_for_posthoc_diagnostics": True,
            "target_results_used_to_select_reported_qpr": False,
        },
        "suite": args.suite,
        "session_scope": args.sessions,
        "selected_parameters": {
            "alpha": selected[0],
            "mass_threshold": selected[1],
            "min_count": selected[2],
            "logit_scale": selected[3],
        },
        "configurations": configurations,
        "methods": list(methods),
        "support_seeds": data.support_seeds[: args.seed_limit],
        "summary": _aggregate(per_seed, methods),
        "per_seed": per_seed,
        "resources": resources,
        "runtime_sec": time.time() - started,
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "dataset": data.dataset,
                "suite": args.suite,
                "runtime_sec": result["runtime_sec"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
