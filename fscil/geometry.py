"""Covariance geometry and memory-bounded scoring for frozen normalized features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch

from .utils import normalized_mean


@dataclass(frozen=True)
class CovarianceEigensystem:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    delta: torch.Tensor
    dof: int


def pooled_covariance(features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]) -> torch.Tensor:
    selected = list(classes)
    if not selected:
        raise ValueError("at least one class is required")
    dimension = int(features_by_class[selected[0]].shape[1])
    device = features_by_class[selected[0]].device
    covariance_sum = torch.zeros((dimension, dimension), dtype=torch.float32, device=device)
    dof = 0
    for class_id in selected:
        values = features_by_class[class_id].float()
        centered = values - values.mean(0, keepdim=True)
        covariance_sum.add_(centered.t().matmul(centered))
        dof += max(0, values.shape[0] - 1)
    if dof <= 0:
        raise ValueError("pooled covariance has zero degrees of freedom")
    return covariance_sum / float(dof)


def covariance_eigensystem(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> CovarianceEigensystem:
    selected = list(classes)
    sigma = pooled_covariance(features_by_class, selected)
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    eigenvalues = eigenvalues.clamp_min(0.0)
    delta = torch.diagonal(sigma).mean().clamp_min(1e-12)
    dof = sum(max(0, features_by_class[class_id].shape[0] - 1) for class_id in selected)
    return CovarianceEigensystem(eigenvalues, eigenvectors, delta, dof)


def inverse_eigenvalues(system: CovarianceEigensystem, shrinkage: float, ridge_gamma: float) -> torch.Tensor:
    value = float(shrinkage)
    if value < 0.0 or value > 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    denominator = (
        (1.0 - value) * system.eigenvalues
        + (value + float(ridge_gamma)) * system.delta
    )
    return denominator.clamp_min(1e-12).reciprocal()


def transform_metric(
    values: torch.Tensor,
    system: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
) -> torch.Tensor:
    weights = inverse_eigenvalues(system, shrinkage, ridge_gamma).sqrt()
    return values.float().matmul(system.eigenvectors) * weights[None, :]


def rotate_to_covariance_basis(
    values: torch.Tensor, system: CovarianceEigensystem
) -> torch.Tensor:
    return values.float().matmul(system.eigenvectors)


def metric_scores_rotated(
    rotated_queries: torch.Tensor,
    rotated_prototypes: torch.Tensor,
    system: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
    query_chunk: int = 2048,
) -> torch.Tensor:
    """Score values already rotated into the covariance eigenbasis."""
    inverse = inverse_eigenvalues(system, shrinkage, ridge_gamma)
    weighted_prototypes = rotated_prototypes * inverse[None, :]
    proto_norm = (rotated_prototypes * weighted_prototypes).sum(1)
    rows = []
    for start in range(0, rotated_queries.shape[0], query_chunk):
        queries = rotated_queries[start:start + query_chunk]
        query_norm = (queries.square() * inverse[None, :]).sum(1, keepdim=True)
        distance = query_norm + proto_norm[None, :] - 2.0 * queries.matmul(
            weighted_prototypes.t()
        )
        rows.append(-0.5 * distance)
    return torch.cat(rows, 0)


def metric_scores(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    system: CovarianceEigensystem,
    shrinkage: float,
    ridge_gamma: float,
    query_chunk: int = 2048,
) -> torch.Tensor:
    """Return negative half squared Mahalanobis distances without a Q x C x D tensor."""
    rotated_queries = rotate_to_covariance_basis(queries, system)
    rotated_prototypes = rotate_to_covariance_basis(prototypes, system)
    return metric_scores_rotated(
        rotated_queries,
        rotated_prototypes,
        system,
        shrinkage,
        ridge_gamma,
        query_chunk,
    )


def cosine_scores(queries: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return queries.float().matmul(prototypes.float().t())


def prototypes_from_classes(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> torch.Tensor:
    return torch.stack([normalized_mean(features_by_class[class_id]) for class_id in classes], 0)


def metric_trace_penalty(
    system: CovarianceEigensystem, shrinkage: float, ridge_gamma: float, shots: int
) -> float:
    """Finite-shot E[d_G(x, mu_hat)-d_G(x, mu)] under shared covariance."""
    inverse = inverse_eigenvalues(system, shrinkage, ridge_gamma)
    value = (inverse * system.eigenvalues).sum() / float(shots)
    return float(value.item())
