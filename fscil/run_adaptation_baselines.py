#!/usr/bin/env python3
"""Adaptation-budget FSCIL comparators on the identical frozen features.

The reported heads in this paper are training-free: they read frozen features and
form prototypes. This runner compares them with methods that spend an
optimization budget. The adaptation
comparators train parameters while reading exactly the same frozen features, the
same BiMC sessions, and the same twenty support draws as the training-free rows.

Information boundary: no method reads query features, query labels, a class
marginal, class names, or CLIP text.  Hyperparameters are chosen on
class-disjoint base pseudo-episodes and audited on untouched base classes before
any target session is evaluated.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import covariance_eigensystem, metric_scores
from .radial_selection import prototype_from_samples
from .selection import _anchor_split, load_audit, load_construction_selection
from .target_data import features_by_class, load_target_evaluation_data
from .utils import atomic_write_json, stable_json_sha256


METHODS = (
    "cosine",
    "radial_shrinksel",
    "clip_adapter_visual",
    "adam_concat",
    "lp_joint_replay",
)
ADAPTATION_METHODS = ("clip_adapter_visual", "adam_concat", "lp_joint_replay")

# Imprinted weights (Qi et al. 2018) are deliberately excluded from the reported
# block. Imprinting appends a unit class prototype as a row of a trained cosine
# classifier, which is commensurate only when the embedding was trained jointly
# with that classifier. On a frozen foundation model the learned base rows are
# discriminative directions that sit far from their own class means, while an
# imprinted row sits exactly on one, so imprinted classes win almost every
# comparison. Measured on CIFAR-100 this yields 0.00 base accuracy against 62.78
# novel accuracy: a calibration artifact rather than a fair rendering of the
# method. The diagnosis is reported in the supplement instead of tabling a
# strawman row.
IMPRINT_EXCLUSION = {
    "method": "imprinted weights (Qi et al. 2018)",
    "reason": "trained cosine rows and unit prototypes are not commensurate on a frozen backbone",
    "measured_cifar100_final_base": 0.0,
    "measured_cifar100_final_novel": 62.78,
}

BASE_STEPS = 300
BASE_LR = 1e-2
WEIGHT_DECAY = 1e-4
ADAPTER_REDUCTION = 4
FT_LR_GRID = (1e-3, 1e-2, 1e-1)
FT_STEP_GRID = (50, 200)
ADAPTER_ALPHA_GRID = (0.2, 0.5)
JOINT_STEP_GRID = (100, 300)


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=1)


def _train_linear(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic full-batch multinomial logistic regression."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = torch.zeros(
        num_classes, features.shape[1], device=features.device, dtype=torch.float32
    )
    bias = torch.zeros(num_classes, device=features.device, dtype=torch.float32)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = features @ weight.t() + bias
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    return weight.detach(), bias.detach()


def _train_cosine_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train a normalized (cosine) classifier with a learnable logit scale.

    Imprinting is only defined against a cosine classifier: it appends a unit
    prototype as a new weight row, which is commensurate with the existing rows
    only when those rows are themselves normalized (Qi et al. 2018). Training an
    unnormalized linear head and then imprinting produces rows whose norms are
    arbitrary relative to the learned ones, which silently destroys the base
    block rather than measuring the method.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = (
        torch.randn(num_classes, features.shape[1], generator=generator).to(
            features.device
        )
        / features.shape[1] ** 0.5
    ).requires_grad_(True)
    log_scale = torch.tensor(
        float(np.log(10.0)), device=features.device
    ).requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [weight, log_scale], lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = log_scale.exp() * (features @ _normalize(weight).t())
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    return _normalize(weight.detach()), log_scale.detach().exp()


def _imprint(
    weight: torch.Tensor,
    scale: torch.Tensor,
    novel_samples: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Append unit prototype rows for novel classes (Qi et al. 2018)."""
    if not novel_samples:
        return weight
    rows = [
        _normalize(novel_samples[class_id].mean(0, keepdim=True)).squeeze(0)
        for class_id in sorted(novel_samples)
    ]
    return torch.cat([weight, torch.stack(rows, 0)], 0)


def _finetune_novel_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    novel_samples: Mapping[int, torch.Tensor],
    num_base: int,
    steps: int,
    learning_rate: float,
) -> torch.Tensor:
    """Update only the appended rows on the five-shot supports.

    Base rows stay frozen, so no base training data is revisited and no base
    prototype can drift; only the newly created classes are optimized. Rows are
    renormalized so the head remains the cosine classifier it was trained as.
    """
    if not novel_samples or int(steps) <= 0:
        return weight
    support = torch.cat([novel_samples[c] for c in sorted(novel_samples)], 0)
    support_labels = torch.cat(
        [
            torch.full(
                (novel_samples[c].shape[0],),
                num_base + position,
                dtype=torch.long,
                device=weight.device,
            )
            for position, c in enumerate(sorted(novel_samples))
        ],
        0,
    )
    frozen_weight = weight[:num_base].detach()
    novel_weight = weight[num_base:].detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [novel_weight], lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        full_weight = torch.cat([frozen_weight, _normalize(novel_weight)], 0)
        logits = scale * (support @ full_weight.t())
        loss = F.cross_entropy(logits, support_labels)
        loss.backward()
        optimizer.step()
    return torch.cat([frozen_weight, _normalize(novel_weight.detach())], 0)


@dataclass
class Adapter:
    down: torch.Tensor
    up: torch.Tensor
    alpha: float

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(values @ self.down.t())
        residual = hidden @ self.up.t()
        return _normalize((1.0 - self.alpha) * values + self.alpha * residual)


def _train_adapter(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Adapter:
    """Image-only reduction of CLIP-Adapter's residual visual adapter.

    The published method trains this adapter against CLIP's text classifier; the
    text branch is unavailable under this paper's image-only budget, so the
    adapter is trained against a base-class linear head instead. The adapter
    form, residual mixing, and bottleneck ratio are the published ones.
    """
    dimension = features.shape[1]
    hidden = max(1, dimension // ADAPTER_REDUCTION)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    down = (
        torch.randn(hidden, dimension, generator=generator).to(features.device)
        / dimension**0.5
    ).requires_grad_(True)
    up = (
        torch.randn(dimension, hidden, generator=generator).to(features.device)
        / hidden**0.5
    ).requires_grad_(True)
    weight = torch.zeros(
        num_classes, dimension, device=features.device, dtype=torch.float32
    ).requires_grad_(True)
    bias = torch.zeros(num_classes, device=features.device).requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [down, up, weight, bias], lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        adapted = Adapter(down, up, alpha).apply(features)
        logits = adapted @ weight.t() + bias
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    return Adapter(down.detach(), up.detach(), alpha)


def _adam_features(values: torch.Tensor, adapter: "Adapter") -> torch.Tensor:
    """ADAM-style concatenation of the frozen and adapted embeddings.

    Both halves are unit-normalized before concatenation so neither dominates
    the cosine comparison, following the released formulation.
    """
    return torch.cat([_normalize(values), _normalize(adapter.apply(values))], dim=1)


def _prototype_scores(
    queries: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    return _normalize(queries) @ _normalize(prototypes).t()


def _accuracy(
    prediction: torch.Tensor, labels: torch.Tensor, num_base: int
) -> Dict[str, float]:
    correct = prediction == labels
    base_mask = labels < int(num_base)
    novel_mask = ~base_mask
    return {
        "overall": 100.0 * float(correct.float().mean().item()),
        "base": (
            100.0 * float(correct[base_mask].float().mean().item())
            if bool(base_mask.any())
            else float("nan")
        ),
        "novel": (
            100.0 * float(correct[novel_mask].float().mean().item())
            if bool(novel_mask.any())
            else float("nan")
        ),
    }


def _base_episode_pool(
    by_class: Mapping[int, torch.Tensor], seed: int
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    return _anchor_split(dict(by_class), seed)


def _pseudo_episode_novel(
    pool: Mapping[int, torch.Tensor], episode_index: int, count: int
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Draw pseudo-novel supports and queries from class-disjoint base classes."""
    rng = np.random.RandomState(9137 + episode_index * 7919)
    candidates = sorted(pool)
    chosen = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(candidates), size=min(count, len(candidates)), replace=False
        ).tolist()
    )
    supports: Dict[int, torch.Tensor] = {}
    queries: Dict[int, torch.Tensor] = {}
    for class_id in chosen:
        values = pool[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(15485863 + episode_index * 32452843 + class_id * 49999)
        order = torch.randperm(values.shape[0], generator=generator).to(values.device)
        supports[class_id] = values[order[:5]]
        queries[class_id] = values[order[5:35]]
    return supports, queries


def _episode_balanced(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_query: Mapping[int, torch.Tensor],
    novel_support: Mapping[int, torch.Tensor],
    novel_query: Mapping[int, torch.Tensor],
    scorer,
) -> float:
    anchor_ids = sorted(anchor_train)
    novel_ids = sorted(novel_support)
    queries = []
    labels = []
    for position, class_id in enumerate(anchor_ids):
        values = anchor_query[class_id]
        queries.append(values)
        labels.append(
            torch.full((values.shape[0],), position, dtype=torch.long, device=values.device)
        )
    for position, class_id in enumerate(novel_ids):
        values = novel_query[class_id]
        queries.append(values)
        labels.append(
            torch.full(
                (values.shape[0],),
                len(anchor_ids) + position,
                dtype=torch.long,
                device=values.device,
            )
        )
    query = torch.cat(queries, 0)
    label = torch.cat(labels, 0)
    prediction = scorer(query, novel_support).argmax(1)
    correct = prediction == label
    anchor_mask = label < len(anchor_ids)
    anchor_accuracy = 100.0 * float(correct[anchor_mask].float().mean().item())
    novel_accuracy = 100.0 * float(correct[~anchor_mask].float().mean().item())
    return 0.5 * (anchor_accuracy + novel_accuracy)


def _select(
    anchor_train: Mapping[int, torch.Tensor],
    anchor_query: Mapping[int, torch.Tensor],
    pool: Mapping[int, torch.Tensor],
    episodes: int,
    novel_count: int,
    device: str,
) -> Dict[str, object]:
    """Choose the reported variable method settings on class-disjoint base classes."""
    anchor_ids = sorted(anchor_train)
    train_features = _normalize(torch.cat([anchor_train[c] for c in anchor_ids], 0))
    train_labels = torch.cat(
        [
            torch.full(
                (anchor_train[c].shape[0],), position, dtype=torch.long, device=device
            )
            for position, c in enumerate(anchor_ids)
        ],
        0,
    )
    num_anchor = len(anchor_ids)
    base_weight, base_scale = _train_cosine_head(
        train_features, train_labels, num_anchor, BASE_STEPS, BASE_LR, seed=17
    )
    episode_inputs = [
        _pseudo_episode_novel(pool, index, novel_count) for index in range(episodes)
    ]

    def evaluate(scorer_factory) -> float:
        values = []
        for supports, queries in episode_inputs:
            supports = {k: _normalize(v) for k, v in supports.items()}
            queries = {k: _normalize(v) for k, v in queries.items()}
            values.append(
                _episode_balanced(
                    {c: _normalize(anchor_train[c]) for c in anchor_ids},
                    {c: _normalize(anchor_query[c]) for c in anchor_ids},
                    supports,
                    queries,
                    scorer_factory,
                )
            )
        return float(mean(values))

    curves: Dict[str, Dict[str, float]] = {}

    def imprint_scorer(steps: int, learning_rate: float):
        def scorer(query: torch.Tensor, supports: Mapping[int, torch.Tensor]):
            weight = _imprint(base_weight, base_scale, supports)
            weight = _finetune_novel_rows(
                weight, base_scale, supports, num_anchor, steps, learning_rate
            )
            return base_scale * (query @ weight.t())

        return scorer

    # Retained as a diagnostic only; see IMPRINT_EXCLUSION.
    imprint_diagnostic = {
        "imprint_only": evaluate(imprint_scorer(0, 0.0)),
        "imprint_ft_lr0.1_steps200": evaluate(imprint_scorer(200, 0.1)),
    }

    adapter_curve = {}
    adapters = {}
    for alpha in ADAPTER_ALPHA_GRID:
        adapter = _train_adapter(
            train_features, train_labels, num_anchor, alpha, BASE_STEPS, BASE_LR, seed=23
        )
        adapters[alpha] = adapter
        anchor_prototypes = torch.stack(
            [
                adapter.apply(_normalize(anchor_train[c])).mean(0)
                for c in anchor_ids
            ]
        )

        def scorer(query: torch.Tensor, supports: Mapping[int, torch.Tensor], _a=adapter, _p=anchor_prototypes):
            novel = torch.stack(
                [_a.apply(supports[c]).mean(0) for c in sorted(supports)]
            )
            return _prototype_scores(_a.apply(query), torch.cat([_p, novel], 0))

        adapter_curve["alpha{}".format(alpha)] = evaluate(scorer)

        def adam_scorer(query: torch.Tensor, supports: Mapping[int, torch.Tensor], _a=adapter):
            anchor_concat = torch.stack(
                [_adam_features(_normalize(anchor_train[c]), _a).mean(0) for c in anchor_ids]
            )
            novel_concat = torch.stack(
                [_adam_features(supports[c], _a).mean(0) for c in sorted(supports)]
            )
            return _prototype_scores(
                _adam_features(query, _a), torch.cat([anchor_concat, novel_concat], 0)
            )

        curves.setdefault("adam_concat", {})["alpha{}".format(alpha)] = evaluate(
            adam_scorer
        )
    curves["clip_adapter_visual"] = adapter_curve

    joint_curve = {}
    for steps in JOINT_STEP_GRID:
        def scorer(query: torch.Tensor, supports: Mapping[int, torch.Tensor], _s=steps):
            novel_ids = sorted(supports)
            features = torch.cat(
                [train_features] + [supports[c] for c in novel_ids], 0
            )
            labels = torch.cat(
                [train_labels]
                + [
                    torch.full(
                        (supports[c].shape[0],),
                        num_anchor + position,
                        dtype=torch.long,
                        device=device,
                    )
                    for position, c in enumerate(novel_ids)
                ],
                0,
            )
            weight, bias = _train_linear(
                features, labels, num_anchor + len(novel_ids), _s, BASE_LR, seed=29
            )
            return query @ weight.t() + bias

        joint_curve["steps{}".format(steps)] = evaluate(scorer)
    curves["lp_joint_replay"] = joint_curve

    selected = {
        method: max(curve.items(), key=lambda item: item[1])[0]
        for method, curve in curves.items()
    }
    return {
        "curves": curves,
        "selected": selected,
        "imprint_diagnostic": imprint_diagnostic,
        "excluded": IMPRINT_EXCLUSION,
    }


def _parse_selected(selected: Mapping[str, str]) -> Dict[str, object]:
    return {
        "adapter_alpha": float(str(selected["clip_adapter_visual"])[5:]),
        "adam_alpha": float(str(selected["adam_concat"])[5:]),
        "joint_steps": int(str(selected["lp_joint_replay"])[5:]),
    }


def _audit_locked_hyperparameters(
    selection_result: Mapping[str, object],
    audit_result: Mapping[str, object],
) -> Dict[str, object]:
    """Evaluate construction-selected choices on the untouched audit curves.

    The audit split must not select a replacement configuration. Its role is to
    report the score and regret of the already locked construction/selection
    choice. Keeping this calculation in the experiment artifact prevents prose
    from calling an independent second selection pass an audit.
    """
    rows: Dict[str, object] = {}
    selected = selection_result["selected"]
    audit_curves = audit_result["curves"]
    for method in ADAPTATION_METHODS:
        key = str(selected[method])
        curve = audit_curves[method]
        if key not in curve:
            raise RuntimeError("locked adaptation key absent from audit: " + method)
        best_key, best_score = max(curve.items(), key=lambda item: float(item[1]))
        locked_score = float(curve[key])
        rows[method] = {
            "locked_key": key,
            "locked_score": locked_score,
            "audit_best_key": str(best_key),
            "audit_best_score": float(best_score),
            "audit_regret": float(best_score) - locked_score,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--selection-bundle", required=True, type=Path)
    parser.add_argument("--audit-bundle", required=True, type=Path)
    parser.add_argument("--radial-parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--selection-episodes", type=int, default=20)
    parser.add_argument("--pseudo-novel", type=int, default=5)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--skip-file-hash", action="store_true")
    args = parser.parse_args()

    started = time.time()
    selection_data = load_construction_selection(args.selection_bundle, args.device)
    construction_split, construction_query = _base_episode_pool(
        selection_data.construction, 7717
    )
    selection_result = _select(
        construction_split,
        construction_query,
        selection_data.selection,
        args.selection_episodes,
        args.pseudo_novel,
        args.device,
    )
    chosen = _parse_selected(selection_result["selected"])

    audit_data = load_audit(args.audit_bundle, args.device)
    anchor_split, anchor_query = _base_episode_pool(audit_data.anchors, 9931)
    audit_result = _select(
        anchor_split,
        anchor_query,
        audit_data.audit,
        args.selection_episodes,
        args.pseudo_novel,
        args.device,
    )
    locked_audit = _audit_locked_hyperparameters(selection_result, audit_result)

    data = load_target_evaluation_data(
        args.feature_cache,
        args.protocol,
        args.device,
        verify_file=not args.skip_file_hash,
    )
    with args.radial_parameters.open("r", encoding="utf-8") as handle:
        radial = json.load(handle)
    if radial["dataset"] != data.dataset:
        raise RuntimeError("radial parameter dataset mismatch")
    mode = str(radial["selected_prototype_mode"])
    radial_lambda = float(radial["selected_lambda"])

    by_class = features_by_class(data)
    base_ids = list(range(data.num_base))
    system = covariance_eigensystem(by_class, base_ids)
    base_train = _normalize(data.train_features[data.train_labels < data.num_base])
    base_train_labels = data.train_labels[data.train_labels < data.num_base]
    base_weight, base_scale = _train_cosine_head(
        base_train, base_train_labels, data.num_base, BASE_STEPS, BASE_LR, seed=17
    )
    adapter = _train_adapter(
        base_train,
        base_train_labels,
        data.num_base,
        chosen["adapter_alpha"],
        BASE_STEPS,
        BASE_LR,
        seed=23,
    )
    base_cosine = torch.stack([_normalize(by_class[c]).mean(0) for c in base_ids])
    base_radial = torch.stack([prototype_from_samples(by_class[c], mode) for c in base_ids])
    base_adapted = torch.stack(
        [adapter.apply(_normalize(by_class[c])).mean(0) for c in base_ids]
    )
    adam_adapter = (
        adapter
        if float(chosen["adam_alpha"]) == float(chosen["adapter_alpha"])
        else _train_adapter(
            base_train,
            base_train_labels,
            data.num_base,
            float(chosen["adam_alpha"]),
            BASE_STEPS,
            BASE_LR,
            seed=23,
        )
    )
    base_adam = torch.stack(
        [_adam_features(_normalize(by_class[c]), adam_adapter).mean(0) for c in base_ids]
    )

    rows: Dict[str, List[Dict[str, object]]] = {method: [] for method in METHODS}
    num_sessions = (data.num_total - data.num_base) // data.increment + 1
    for seed_position in range(int(args.seed_limit)):
        seed_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in METHODS}
        for session_index in range(num_sessions):
            end_class = min(
                data.num_base + session_index * data.increment, data.num_total
            )
            seen_novel = end_class - data.num_base
            query_mask = data.test_labels < end_class
            queries = data.test_features[query_mask]
            labels = data.test_labels[query_mask]
            normalized_queries = _normalize(queries)
            novel_samples: Dict[int, torch.Tensor] = {}
            if seen_novel:
                support_rows = data.standard_support_abs[
                    seed_position, :seen_novel
                ].reshape(-1)
                support_features = data.train_features[support_rows]
                support_labels = data.train_labels[support_rows]
                novel_samples = {
                    int(class_id): support_features[support_labels == class_id]
                    for class_id in range(data.num_base, end_class)
                }
            normalized_novel = {k: _normalize(v) for k, v in novel_samples.items()}

            predictions: Dict[str, torch.Tensor] = {}
            cosine_centers = torch.cat(
                [base_cosine]
                + (
                    [torch.stack([normalized_novel[c].mean(0) for c in sorted(normalized_novel)])]
                    if normalized_novel
                    else []
                )
            )
            predictions["cosine"] = _prototype_scores(queries, cosine_centers).argmax(1)
            radial_centers = torch.cat(
                [base_radial]
                + (
                    [
                        torch.stack(
                            [
                                prototype_from_samples(novel_samples[c], mode)
                                for c in sorted(novel_samples)
                            ]
                        )
                    ]
                    if novel_samples
                    else []
                )
            )
            predictions["radial_shrinksel"] = metric_scores(
                queries, radial_centers, system, radial_lambda, 1.0, args.query_chunk
            ).argmax(1)

            adapted_centers = torch.cat(
                [base_adapted]
                + (
                    [
                        torch.stack(
                            [
                                adapter.apply(normalized_novel[c]).mean(0)
                                for c in sorted(normalized_novel)
                            ]
                        )
                    ]
                    if normalized_novel
                    else []
                )
            )
            predictions["clip_adapter_visual"] = _prototype_scores(
                adapter.apply(normalized_queries), adapted_centers
            ).argmax(1)

            adam_centers = torch.cat(
                [base_adam]
                + (
                    [
                        torch.stack(
                            [
                                _adam_features(normalized_novel[c], adam_adapter).mean(0)
                                for c in sorted(normalized_novel)
                            ]
                        )
                    ]
                    if normalized_novel
                    else []
                )
            )
            predictions["adam_concat"] = _prototype_scores(
                _adam_features(normalized_queries, adam_adapter), adam_centers
            ).argmax(1)

            if normalized_novel:
                joint_features = torch.cat(
                    [base_train] + [normalized_novel[c] for c in sorted(normalized_novel)], 0
                )
                joint_labels = torch.cat(
                    [base_train_labels]
                    + [
                        torch.full(
                            (normalized_novel[c].shape[0],),
                            int(c),
                            dtype=torch.long,
                            device=base_train_labels.device,
                        )
                        for c in sorted(normalized_novel)
                    ],
                    0,
                )
            else:
                joint_features, joint_labels = base_train, base_train_labels
            joint_weight, joint_bias = _train_linear(
                joint_features,
                joint_labels,
                end_class,
                chosen["joint_steps"],
                BASE_LR,
                seed=29,
            )
            predictions["lp_joint_replay"] = (
                normalized_queries @ joint_weight.t() + joint_bias
            ).argmax(1)

            for method in METHODS:
                row = _accuracy(predictions[method], labels, data.num_base)
                row["session_index"] = session_index
                seed_rows[method].append(row)
        for method in METHODS:
            rows[method].append(
                {
                    "support_seed": int(data.support_seeds[seed_position]),
                    "sessions": seed_rows[method],
                }
            )
        print("seed {} done".format(int(data.support_seeds[seed_position])), flush=True)

    summary = {}
    for method in METHODS:
        session_average = [
            float(mean(float(item["overall"]) for item in row["sessions"]))
            for row in rows[method]
        ]
        final = rows[method][0]["sessions"][-1]
        summary[method] = {
            "session_average_mean": float(mean(session_average)),
            "session_average_std": float(pstdev(session_average)),
            "session_average_per_seed": session_average,
            "final": {
                key: float(
                    mean(float(row["sessions"][-1][key]) for row in rows[method])
                )
                for key in ("overall", "base", "novel")
            },
        }

    payload: Dict[str, object] = {
        "schema_version": 1,
        "stage": "adaptation_budget_comparators",
        "dataset": data.dataset,
        "backbone": dict(data.backbone),
        "protocol_sha256": data.protocol_sha256,
        "support_sha256": data.support_sha256,
        "feature_file_sha256": data.feature_file_sha256,
        "selection_bundle_sha256": selection_data.bundle_sha256,
        "audit_bundle_sha256": audit_data.bundle_sha256,
        "methods": list(METHODS),
        "adaptation_methods": list(ADAPTATION_METHODS),
        "base_only_selection": selection_result,
        "untouched_base_audit": audit_result,
        "locked_hyperparameter_audit": locked_audit,
        "selected_hyperparameters": chosen,
        "fixed_hyperparameters": {
            "base_steps": BASE_STEPS,
            "base_learning_rate": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "adapter_reduction": ADAPTER_REDUCTION,
        },
        "information_budget": {
            "uses_query_features_for_training": False,
            "uses_query_labels": False,
            "uses_class_marginal": False,
            "uses_class_names_or_text": False,
            "trains_parameters": True,
            "base_training_data_access": {
                "clip_adapter_visual": "base_session_once",
                "adam_concat": "base_session_once",
                "lp_joint_replay": "every_incremental_session",
            },
            "revisits_base_training_data": {
                "clip_adapter_visual": True,
                "adam_concat": True,
                "lp_joint_replay": True,
            },
        },
        "per_seed": rows,
        "summary": summary,
        "runtime_sec": time.time() - started,
    }
    payload["result_sha256"] = stable_json_sha256(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps({m: summary[m]["session_average_mean"] for m in METHODS}, indent=1))


if __name__ == "__main__":
    main()
