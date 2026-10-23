"""Base-rehearsal and base-geometry guard for QPR prototype proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from .geometry import CovarianceEigensystem, metric_scores


STEP_GRID = (1.0, 0.75, 0.5, 0.25, 0.0)
REHEARSAL_PER_BASE_CLASS = 32


@dataclass(frozen=True)
class GuardedPrototypeOutput:
    prototypes: torch.Tensor
    metadata: Dict[str, object]


def boundary_rehearsal_coreset(
    features_by_class: Mapping[int, torch.Tensor],
    covariance: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
    per_class: int = REHEARSAL_PER_BASE_CLASS,
) -> torch.Tensor:
    """Select a deterministic, bounded base decision-boundary coreset.

    For coverage, every base class contributes up to ``per_class`` embeddings.
    Within a class, the fixed base head's lowest top-one/top-two score margins
    are retained.  Class labels are used only to partition the authorized base
    training data while constructing the coreset; the returned bank contains
    embeddings only and the incremental guard never receives labels.
    """

    class_ids = sorted(int(value) for value in features_by_class)
    if len(class_ids) < 2:
        raise ValueError("boundary coreset requires at least two base classes")
    limit = int(per_class)
    if limit < 1:
        raise ValueError("per_class must be positive")
    prototypes = torch.stack(
        [F.normalize(features_by_class[class_id].float().mean(0), dim=0)
         for class_id in class_ids],
        dim=0,
    )
    parts = []
    for class_id in class_ids:
        values = features_by_class[class_id].float()
        scores = metric_scores(
            values,
            prototypes,
            covariance,
            float(shrinkage),
            float(ridge_gamma),
        )
        top_two = scores.topk(2, dim=1).values
        winner_margin = top_two[:, 0] - top_two[:, 1]
        # Stable tie-breaking by source order makes the bank byte-reproducible.
        order = torch.argsort(winner_margin, stable=True)
        parts.append(values[order[: min(limit, values.shape[0])]])
    return torch.cat(parts, dim=0)


def base_guarded_prototypes(
    original_prototypes: torch.Tensor,
    proposed_prototypes: torch.Tensor,
    num_base_classes: int,
    base_rehearsal: torch.Tensor,
    covariance: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
    separation_ratio: float,
    step_grid: Sequence[float] = STEP_GRID,
) -> GuardedPrototypeOutput:
    """Shrink each novel proposal until it is safe on stored base geometry.

    For every novel class independently, the largest committed interpolation
    step is accepted only when (i) its score does not overtake the original
    head winner on any base-rehearsal embedding newly captured by that class,
    and (ii) its nearest-base Mahalanobis separation is at least the selected
    fraction of the original separation.  Updates are accepted sequentially
    after rescoring the complete candidate head, so the combined head exactly
    preserves every base-bank prediction under the inference computation.
    """

    if original_prototypes.ndim != 2 or proposed_prototypes.ndim != 2:
        raise ValueError("prototype inputs must be matrices")
    if original_prototypes.shape != proposed_prototypes.shape:
        raise ValueError("original/proposed prototype shapes differ")
    if base_rehearsal.ndim != 2:
        raise ValueError("base_rehearsal must be a matrix")
    if base_rehearsal.shape[1] != original_prototypes.shape[1]:
        raise ValueError("base rehearsal/prototype dimensions differ")
    class_count = int(original_prototypes.shape[0])
    base_count = int(num_base_classes)
    if base_count < 1 or base_count >= class_count:
        raise ValueError("guard requires at least one base and one novel class")
    ratio = float(separation_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("separation_ratio must be in [0, 1]")
    steps = tuple(float(value) for value in step_grid)
    if not steps or any(value < 0.0 or value > 1.0 for value in steps):
        raise ValueError("step_grid must contain values in [0, 1]")
    if 0.0 not in steps:
        raise ValueError("step_grid must include the no-update step zero")
    steps = tuple(sorted(set(steps), reverse=True))

    original = original_prototypes.float()
    proposed = proposed_prototypes.float()
    rehearsal = base_rehearsal.float()
    original_scores = metric_scores(
        rehearsal,
        original,
        covariance,
        float(shrinkage),
        float(ridge_gamma),
    )
    original_prediction = original_scores.argmax(1)
    base_prototypes = original[:base_count]
    guarded = original.clone()
    current_scores = original_scores.clone()
    selected_steps: Dict[str, float] = {}
    original_min_separation: Dict[str, float] = {}
    selected_min_separation: Dict[str, float] = {}
    blocked_by_rehearsal = 0
    blocked_by_geometry = 0
    proposal_class_count = 0

    for class_id in range(base_count, class_count):
        if torch.equal(proposed[class_id], original[class_id]):
            selected_steps[str(class_id)] = 0.0
            continue
        proposal_class_count += 1
        original_distance = -2.0 * metric_scores(
            base_prototypes,
            original[class_id : class_id + 1],
            covariance,
            float(shrinkage),
            float(ridge_gamma),
        ).squeeze(1)
        # The quadratic form is nonnegative; cancellation in the distance
        # expansion can produce values a few ulps below zero.
        original_distance = original_distance.clamp_min(0.0)
        original_min = float(original_distance.min().item())
        original_min_separation[str(class_id)] = original_min
        accepted = None
        accepted_min = original_min
        saw_rehearsal_failure = False
        saw_geometry_failure = False
        for step in steps:
            if step == 0.0:
                candidate = original[class_id]
                if not torch.equal(current_scores.argmax(1), original_prediction):
                    raise RuntimeError(
                        "sequential guard invariant failed at no-update step"
                    )
                accepted = candidate
                accepted_min = original_min
                selected_steps[str(class_id)] = 0.0
                break
            else:
                candidate = F.normalize(
                    (
                        (1.0 - step) * original[class_id]
                        + step * proposed[class_id]
                    ).unsqueeze(0),
                    dim=1,
                ).squeeze(0)
            # Rescore the complete candidate head exactly as inference does.
            # Column-wise recomputation is algebraically equivalent but can change
            # an argmax tie by a few floating-point ulps.
            trial_prototypes = guarded.clone()
            trial_prototypes[class_id] = candidate
            trial_scores = metric_scores(
                rehearsal,
                trial_prototypes,
                covariance,
                float(shrinkage),
                float(ridge_gamma),
            )
            rehearsal_safe = torch.equal(
                trial_scores.argmax(1), original_prediction
            )
            candidate_distance = -2.0 * metric_scores(
                base_prototypes,
                candidate.unsqueeze(0),
                covariance,
                float(shrinkage),
                float(ridge_gamma),
            ).squeeze(1)
            candidate_distance = candidate_distance.clamp_min(0.0)
            candidate_min = float(candidate_distance.min().item())
            geometry_safe = candidate_min + 1e-8 >= ratio * original_min
            saw_rehearsal_failure = saw_rehearsal_failure or not rehearsal_safe
            saw_geometry_failure = saw_geometry_failure or not geometry_safe
            if rehearsal_safe and geometry_safe:
                accepted = candidate
                accepted_min = candidate_min
                selected_steps[str(class_id)] = step
                break
        if accepted is None:
            raise RuntimeError("zero-step guard candidate unexpectedly failed")
        guarded[class_id] = accepted
        if selected_steps[str(class_id)] > 0.0:
            current_scores = trial_scores
        selected_min_separation[str(class_id)] = accepted_min
        if selected_steps[str(class_id)] == 0.0:
            blocked_by_rehearsal += int(saw_rehearsal_failure)
            blocked_by_geometry += int(saw_geometry_failure)

    if not torch.equal(guarded[:base_count], original[:base_count]):
        raise RuntimeError("base guard modified a base prototype")
    final_scores = metric_scores(
        rehearsal,
        guarded,
        covariance,
        float(shrinkage),
        float(ridge_gamma),
    )
    prediction_changes = int(
        (final_scores.argmax(1) != original_prediction).sum().item()
    )
    if prediction_changes:
        raise RuntimeError(
            "base guard changed {} rehearsal predictions".format(
                prediction_changes
            )
        )
    positive_steps = [value for value in selected_steps.values() if value > 0.0]
    full_steps = sum(value == 1.0 for value in selected_steps.values())
    return GuardedPrototypeOutput(
        prototypes=guarded,
        metadata={
            "method": "per-novel QPR base-rehearsal/geometry guard",
            "separation_ratio": ratio,
            "step_grid": list(steps),
            "base_rehearsal_count": int(rehearsal.shape[0]),
            "proposal_class_count": int(proposal_class_count),
            "accepted_class_count": len(positive_steps),
            "full_step_class_count": int(full_steps),
            "mean_positive_step": (
                float(sum(positive_steps) / len(positive_steps))
                if positive_steps
                else 0.0
            ),
            "selected_steps": selected_steps,
            "blocked_by_rehearsal_count": int(blocked_by_rehearsal),
            "blocked_by_geometry_count": int(blocked_by_geometry),
            "base_rehearsal_prediction_changes": prediction_changes,
            "original_min_base_separation": original_min_separation,
            "selected_min_base_separation": selected_min_separation,
            "updates_base_prototypes": False,
            "uses_query_labels": False,
            "uses_query_marginal_constraint": False,
            "uses_base_training_features": True,
            "receives_base_labels_at_guard_time": False,
        },
    )
