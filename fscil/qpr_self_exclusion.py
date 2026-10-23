"""Exact query-wise self-exclusion for one-shot QPR.

QPR's admission decision for a query depends only on the fixed pre-update
scores for that query.  Removing query i can consequently alter only the
prototype of the novel class to which i was admitted.  This permits exact
leave-one-query-out scoring with one ordinary cohort update plus one corrected
score entry per contributing query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .geometry import (
    CovarianceEigensystem,
    inverse_eigenvalues,
    metric_scores,
    rotate_to_covariance_basis,
)
from .transductive_methods import query_prototype_rectification


@dataclass(frozen=True)
class QPRSelfExclusionOutput:
    same_cohort_scores: torch.Tensor
    leave_one_out_scores: torch.Tensor
    same_cohort_prototypes: torch.Tensor
    metadata: Dict[str, object]


def _paired_metric_scores(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    covariance: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
) -> torch.Tensor:
    """Score paired query/prototype rows under the fixed Mahalanobis metric."""
    if queries.shape != prototypes.shape:
        raise ValueError("paired queries and prototypes must have identical shape")
    rotated_query = rotate_to_covariance_basis(queries, covariance)
    rotated_prototype = rotate_to_covariance_basis(prototypes, covariance)
    inverse = inverse_eigenvalues(covariance, shrinkage, ridge_gamma)
    return -0.5 * (
        (rotated_query - rotated_prototype).square() * inverse.unsqueeze(0)
    ).sum(1)


def qpr_same_and_leave_one_out_scores(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    baseline_scores: torch.Tensor,
    covariance: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
    num_base_classes: int,
    alpha: float,
    mass_threshold: float,
    min_count: int,
    logit_scale: float,
    query_chunk: int = 2048,
) -> QPRSelfExclusionOutput:
    """Return ordinary QPR and exact leave-one-query-out QPR scores.

    For each query, the leave-one-out row is the score obtained by literally
    removing that query from the adaptation cohort, running the unchanged
    one-shot QPR operator, and then classifying the held-out query.
    """
    output = query_prototype_rectification(
        query,
        prototypes,
        baseline_scores,
        int(num_base_classes),
        alpha=float(alpha),
        mass_threshold=float(mass_threshold),
        min_count=int(min_count),
        logit_scale=float(logit_scale),
    )
    same_scores = metric_scores(
        query,
        output.prototypes,
        covariance,
        float(shrinkage),
        float(ridge_gamma),
        int(query_chunk),
    )
    leave_scores = same_scores.clone()

    probability = torch.softmax(
        baseline_scores.float() * float(logit_scale), dim=1
    )
    prediction = probability.argmax(1)
    base_count = int(num_base_classes)
    novel_probability = probability[:, base_count:]
    novel_mass = novel_probability.sum(1)
    assigned = novel_probability.argmax(1) + base_count
    gate = (prediction >= base_count) & (
        novel_mass >= float(mass_threshold)
    )

    original = prototypes.float()
    normalized_original = F.normalize(original, dim=1)
    original_radius = original.norm(dim=1)
    corrected_entries = 0
    threshold_fallback_entries = 0
    contributing_queries = 0
    affected_classes = 0
    for class_id in range(base_count, int(prototypes.shape[0])):
        selected = gate & (assigned == class_id)
        selected_index = selected.nonzero(as_tuple=False).flatten()
        count = int(selected_index.numel())
        if count < int(min_count) or float(alpha) == 0.0:
            continue
        affected_classes += 1
        contributing_queries += count
        if count - 1 < int(min_count):
            loo_prototypes = original[class_id].unsqueeze(0).expand(count, -1)
            threshold_fallback_entries += count
        else:
            weights = probability[selected_index, class_id]
            selected_query = query[selected_index].float()
            weighted_sum = (weights.unsqueeze(1) * selected_query).sum(0)
            weight_sum = weights.sum()
            remaining_sum = weighted_sum.unsqueeze(0) - (
                weights.unsqueeze(1) * selected_query
            )
            remaining_weight = weight_sum - weights
            query_direction = F.normalize(
                remaining_sum / remaining_weight.clamp_min(1e-12).unsqueeze(1),
                dim=1,
            )
            direction = F.normalize(
                (1.0 - float(alpha)) * normalized_original[class_id].unsqueeze(0)
                + float(alpha) * query_direction,
                dim=1,
            )
            loo_prototypes = direction * original_radius[class_id]
        corrected = _paired_metric_scores(
            query[selected_index],
            loo_prototypes,
            covariance,
            float(shrinkage),
            float(ridge_gamma),
        )
        leave_scores[selected_index, class_id] = corrected
        corrected_entries += count

    metadata: Dict[str, object] = dict(output.metadata)
    metadata.update(
        {
            "self_exclusion": "exact leave-one-query-out adaptation",
            "classified_query_present_in_its_adaptation_set": False,
            "contributing_query_count": int(contributing_queries),
            "affected_novel_class_count": int(affected_classes),
            "corrected_query_class_score_entries": int(corrected_entries),
            "minimum_count_fallback_entries": int(threshold_fallback_entries),
            "other_query_class_scores_reused_from_same_cohort": True,
            "uses_query_labels": False,
        }
    )
    return QPRSelfExclusionOutput(
        same_cohort_scores=same_scores,
        leave_one_out_scores=leave_scores,
        same_cohort_prototypes=output.prototypes,
        metadata=metadata,
    )
