"""Matched-feature inductive heads used by the audited fscil.

The functions in this module are deliberately query-local: none of them estimate a
quantity from a query cohort.  This makes batch/order invariance testable and keeps
the strict-inductive table separate from later transductive experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .utils import normalized_mean


HYCAL_COVARIANCE_LAMBDA = 1e-4
HYCAL_COVARIANCE_GAMMA = 1.0


@dataclass(frozen=True)
class ULAParameters:
    """Parameters of the official Case-A uncertainty-guided adjustment."""

    logit_scale: float = 4.0
    margin_threshold: float = 0.04
    adjustment_strength: float = 0.4
    base_entropy_temperature: float = 0.25
    entropy_slope: float = 8.0
    entropy_threshold: float = 0.06


@dataclass(frozen=True)
class HyCalClassStatistics:
    """Exact low-rank form of one HyCal class precision matrix.

    HyCal's published five-shot parameters are ``alpha=1, beta=0``.  Interpreting
    the otherwise singular zero-temperature sigmoid as the intended hard-threshold
    limit, every candidate class in the standard FSCIL protocol receives unit
    Mahalanobis weight, so neither CLIP text features nor the paper's
    query-wise affine distance normalization can change the prediction.  This
    reconstruction uses the standard higher-is-better score ``-distance`` (the
    paper writes its nonnegative distance inside an otherwise ambiguous argmax).
    It evaluates that image-only limiting classifier without
    materialising one dense D-by-D matrix per class.
    """

    mean: torch.Tensor
    basis: torch.Tensor
    row_precision: torch.Tensor
    count: int
    ridge_precision: float


@dataclass(frozen=True)
class RanPACLowRankSystem:
    """Numerically stable spectral form of a low-rank RanPAC Gram matrix.

    The official RanPAC head solves ``(H.T H + ridge I) W = H.T Y``.  When
    the random-feature dimension exceeds the number of base examples, forming
    that singular ``M x M`` system in float32 can erase a small ridge.  This
    object stores the non-zero right singular subspace of ``H`` in float64, so
    the same ridge-regression solution can be evaluated without retaining the
    original projected rows or changing RanPAC's objective.
    """

    squared_singular_values: torch.Tensor
    right_vectors: torch.Tensor
    rotated_targets: torch.Tensor
    num_samples: int
    feature_dimension: int


def hycal_class_statistics(
    class_features: torch.Tensor,
    covariance_lambda: float = HYCAL_COVARIANCE_LAMBDA,
    covariance_gamma: float = HYCAL_COVARIANCE_GAMMA,
) -> HyCalClassStatistics:
    """Build ``((1-lambda) Cov + lambda gamma I)^-1`` exactly."""
    values = class_features.float()
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("HyCal class features must be a non-empty matrix")
    if not 0.0 <= float(covariance_lambda) <= 1.0:
        raise ValueError("HyCal covariance lambda must lie in [0, 1]")
    ridge = float(covariance_lambda) * float(covariance_gamma)
    if ridge <= 0.0:
        raise ValueError("HyCal covariance regularization must be positive")

    center = values.mean(0)
    centered = values - center[None, :]
    if values.shape[0] == 1:
        basis = values.new_empty((0, values.shape[1]))
        row_precision = values.new_empty((0,))
    else:
        _, singular_values, basis = torch.linalg.svd(
            centered, full_matrices=False
        )
        covariance_eigenvalues = (
            (1.0 - float(covariance_lambda))
            * singular_values.square()
            / float(values.shape[0] - 1)
        )
        row_precision = (ridge + covariance_eigenvalues).reciprocal()
    return HyCalClassStatistics(
        mean=center,
        basis=basis,
        row_precision=row_precision,
        count=int(values.shape[0]),
        ridge_precision=1.0 / ridge,
    )


def hycal_state(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> Tuple[HyCalClassStatistics, ...]:
    return tuple(
        hycal_class_statistics(features_by_class[int(class_id)])
        for class_id in classes
    )


def hycal_squared_mahalanobis_chunked(
    queries: torch.Tensor,
    statistics: HyCalClassStatistics,
    query_chunk: int = 4096,
) -> torch.Tensor:
    """Evaluate one class-specific quadratic form with stable low-rank algebra."""
    if queries.ndim != 2 or queries.shape[1] != statistics.mean.numel():
        raise ValueError("HyCal query/statistic shape mismatch")
    if int(query_chunk) < 1:
        raise ValueError("query_chunk must be positive")
    output = []
    center = statistics.mean.float()
    basis = statistics.basis.float()
    row_precision = statistics.row_precision.float()
    center_square = center.square().sum()
    center_projection = (
        center.matmul(basis.t()) if basis.shape[0] else center.new_empty((0,))
    )
    for start in range(0, queries.shape[0], int(query_chunk)):
        values = queries[start : start + int(query_chunk)].float()
        euclidean_square = (
            values.square().sum(1)
            + center_square
            - 2.0 * values.matmul(center)
        ).clamp_min(0.0)
        if basis.shape[0]:
            projection = values.matmul(basis.t()) - center_projection[None, :]
            projection_square = projection.square()
            orthogonal_square = (
                euclidean_square - projection_square.sum(1)
            ).clamp_min(0.0)
            distance = (
                float(statistics.ridge_precision) * orthogonal_square
                + (projection_square * row_precision[None, :]).sum(1)
            )
        else:
            distance = float(statistics.ridge_precision) * euclidean_square
        output.append(distance.clamp_min(0.0))
    result = torch.cat(output, 0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("HyCal produced non-finite distances")
    return result


def hycal_scores_chunked(
    queries: torch.Tensor,
    statistics: Sequence[HyCalClassStatistics],
    query_chunk: int = 4096,
) -> torch.Tensor:
    """Higher-is-better five-shot HyCal scores, independent for every query."""
    if not statistics:
        raise ValueError("HyCal requires at least one candidate class")
    # At alpha=1 and beta->0 all standard FSCIL class counts are >1, hence
    # HyCal's dynamic weight is exactly one.  Negative distance implements the
    # standard nearest-Mahalanobis interpretation of the published argmax rule.
    if any(record.count <= 1 for record in statistics):
        raise ValueError("image-only HyCal limit requires more than one support per class")
    return -torch.stack(
        [
            hycal_squared_mahalanobis_chunked(
                queries, record, query_chunk=query_chunk
            )
            for record in statistics
        ],
        dim=1,
    )


def hycal_state_bytes(statistics: Sequence[HyCalClassStatistics]) -> int:
    """Stored tensor bytes for the exact low-rank HyCal state."""
    return int(
        sum(
            record.mean.numel() * record.mean.element_size()
            + record.basis.numel() * record.basis.element_size()
            + record.row_precision.numel() * record.row_precision.element_size()
            for record in statistics
        )
    )


def normalized_prototypes(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> torch.Tensor:
    return torch.stack(
        [normalized_mean(features_by_class[int(class_id)]) for class_id in classes], 0
    )


def raw_prototypes(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> torch.Tensor:
    return torch.stack(
        [features_by_class[int(class_id)].float().mean(0) for class_id in classes], 0
    )


def cosine_scores_chunked(
    queries: torch.Tensor, prototypes: torch.Tensor, query_chunk: int = 4096
) -> torch.Tensor:
    rows = []
    normalized_queries = F.normalize(queries.float(), dim=-1)
    normalized_centers = F.normalize(prototypes.float(), dim=-1)
    for start in range(0, normalized_queries.shape[0], query_chunk):
        rows.append(
            normalized_queries[start : start + query_chunk].matmul(normalized_centers.t())
        )
    return torch.cat(rows, 0)


def euclidean_scores_chunked(
    queries: torch.Tensor, prototypes: torch.Tensor, query_chunk: int = 4096
) -> torch.Tensor:
    """Negative squared Euclidean distance to unnormalised arithmetic means."""
    centers = prototypes.float()
    center_norm = centers.square().sum(1)
    rows = []
    for start in range(0, queries.shape[0], query_chunk):
        values = queries[start : start + query_chunk].float()
        distance = (
            values.square().sum(1, keepdim=True)
            + center_norm[None, :]
            - 2.0 * values.matmul(centers.t())
        )
        rows.append(-distance)
    return torch.cat(rows, 0)


def diagonal_lda_scores_chunked(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    variance: torch.Tensor,
    ridge: float,
    query_chunk: int = 4096,
) -> torch.Tensor:
    scale = variance.float().mean().clamp_min(1e-12)
    inverse = (variance.float() + float(ridge) * scale).clamp_min(1e-12).reciprocal()
    weighted_centers = prototypes.float() * inverse[None, :]
    center_norm = (prototypes.float() * weighted_centers).sum(1)
    rows = []
    for start in range(0, queries.shape[0], query_chunk):
        values = queries[start : start + query_chunk].float()
        distance = (
            (values.square() * inverse[None, :]).sum(1, keepdim=True)
            + center_norm[None, :]
            - 2.0 * values.matmul(weighted_centers.t())
        )
        rows.append(-0.5 * distance)
    return torch.cat(rows, 0)


def teen_calibrate(
    base_prototypes: torch.Tensor,
    novel_prototypes: torch.Tensor,
    shift_weight: float,
    softmax_scale: float,
) -> torch.Tensor:
    """TEEN prototype calibration following the authors' official implementation."""
    if novel_prototypes.numel() == 0 or float(shift_weight) == 0.0:
        return F.normalize(novel_prototypes.float(), dim=-1)
    base = F.normalize(base_prototypes.float(), dim=-1)
    novel = F.normalize(novel_prototypes.float(), dim=-1)
    weights = torch.softmax(float(softmax_scale) * novel.matmul(base.t()), dim=1)
    borrowed = F.normalize(weights.matmul(base), dim=-1)
    return F.normalize(
        (1.0 - float(shift_weight)) * novel + float(shift_weight) * borrowed,
        dim=-1,
    )


def ula_adjust_logits(
    cosine_scores: torch.Tensor,
    num_base_classes: int,
    parameters: ULAParameters,
) -> torch.Tensor:
    """Apply ULA's official Case-A rule independently to each query.

    The currently released ULA code reserves but does not enable Case B.  We mirror
    that behavior and retain the official default upstream cosine-logit scale of 4.
    """
    logits = cosine_scores.float() * float(parameters.logit_scale)
    if logits.shape[1] <= int(num_base_classes):
        return logits
    top_values, top_indices = logits.topk(2, dim=1)
    first_value, second_value = top_values[:, 0], top_values[:, 1]
    first_index, second_index = top_indices[:, 0], top_indices[:, 1]
    margin = first_value - second_value
    threshold = max(float(parameters.margin_threshold), 1e-6)
    gate = (
        (first_index < int(num_base_classes))
        & (second_index >= int(num_base_classes))
        & (margin < threshold)
    )
    ambiguity = (threshold - margin).clamp_min(0.0) / threshold
    base_logits = logits[:, : int(num_base_classes)] / max(
        float(parameters.base_entropy_temperature), 1e-6
    )
    base_probability = torch.softmax(base_logits, dim=1)
    entropy = -(
        base_probability * base_probability.clamp_min(1e-12).log()
    ).sum(1) / math.log(float(num_base_classes) + 1e-12)
    entropy_gate = torch.sigmoid(
        float(parameters.entropy_slope)
        * (entropy - float(parameters.entropy_threshold))
    )
    delta = (
        float(parameters.adjustment_strength)
        * ambiguity
        * entropy_gate
        * gate.float()
    )
    adjusted = logits.clone()
    rows = torch.arange(logits.shape[0], device=logits.device)
    adjusted[rows, first_index] -= delta
    adjusted[rows, second_index] += delta
    return adjusted


def fecam_inverse_covariance(
    class_features: torch.Tensor,
    alpha_diagonal: float = 1.0,
    alpha_off_diagonal: float = 0.0,
) -> torch.Tensor:
    """Official FeCAM ViT shrink-and-correlation covariance for one class.

    The official ViT path does not apply the Tukey transform.  That is important for
    CLIP embeddings, whose coordinates may be negative and therefore do not admit the
    released implementation's fractional ``pow`` transform.
    """
    values = class_features.float()
    if values.shape[0] < 2:
        raise ValueError("FeCAM covariance requires at least two class samples")
    centered = values - values.mean(0, keepdim=True)
    covariance = centered.t().matmul(centered) / float(values.shape[0] - 1)
    diagonal_mean = torch.diagonal(covariance).mean().clamp_min(1e-12)
    off_diagonal = covariance.clone()
    off_diagonal.fill_diagonal_(0.0)
    mask = off_diagonal != 0.0
    if bool(mask.any()):
        off_diagonal_mean = off_diagonal[mask].mean()
    else:
        off_diagonal_mean = covariance.new_zeros(())
    identity = torch.eye(covariance.shape[0], device=covariance.device)
    shrunk = (
        covariance
        + float(alpha_diagonal) * diagonal_mean * identity
        + float(alpha_off_diagonal) * off_diagonal_mean * (1.0 - identity)
    )
    standard_deviation = torch.diagonal(shrunk).clamp_min(1e-12).sqrt()
    correlation = shrunk / (
        standard_deviation[:, None] * standard_deviation[None, :]
    )
    correlation = 0.5 * (correlation + correlation.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    floor = eigenvalues.max().clamp_min(1.0) * 1e-6
    inverse = eigenvectors.matmul(
        torch.diag_embed(eigenvalues.clamp_min(floor).reciprocal())
    ).matmul(eigenvectors.t())
    return 0.5 * (inverse + inverse.t())


def fecam_state(
    features_by_class: Dict[int, torch.Tensor], classes: Iterable[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = [int(class_id) for class_id in classes]
    means = normalized_prototypes(features_by_class, selected)
    inverse_covariances = torch.stack(
        [fecam_inverse_covariance(features_by_class[class_id]) for class_id in selected],
        0,
    )
    return means, inverse_covariances


def class_specific_mahalanobis_scores_chunked(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    inverse_covariances: torch.Tensor,
    query_chunk: int = 64,
) -> torch.Tensor:
    rows = []
    values = F.normalize(queries.float(), dim=-1)
    centers = F.normalize(prototypes.float(), dim=-1)
    for start in range(0, values.shape[0], query_chunk):
        difference = values[start : start + query_chunk, None, :] - centers[None, :, :]
        distance = torch.einsum(
            "qcd,cde,qce->qc", difference, inverse_covariances.float(), difference
        )
        rows.append(-distance)
    return torch.cat(rows, 0)


def make_ranpac_projection(
    input_dimension: int,
    output_dimension: int,
    seed: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if int(output_dimension) <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(
        int(input_dimension), int(output_dimension), generator=generator, dtype=torch.float32
    ).to(device)


def ranpac_transform(
    features: torch.Tensor, projection: Optional[torch.Tensor]
) -> torch.Tensor:
    values = features.float()
    if projection is None:
        return values
    return F.relu(values.matmul(projection))


def ranpac_normal_equations(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    projection: Optional[torch.Tensor] = None,
    feature_chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dimension = features.shape[1] if projection is None else projection.shape[1]
    gram = torch.zeros((dimension, dimension), dtype=torch.float32, device=features.device)
    cross = torch.zeros(
        (dimension, int(num_classes)), dtype=torch.float32, device=features.device
    )
    for start in range(0, features.shape[0], feature_chunk):
        transformed = ranpac_transform(
            features[start : start + feature_chunk], projection
        )
        chunk_labels = labels[start : start + feature_chunk].long()
        gram.add_(transformed.t().matmul(transformed))
        cross.index_add_(1, chunk_labels, transformed.t())
    return gram, cross


def ranpac_low_rank_system(
    projected_features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> RanPACLowRankSystem:
    """Factor a low-rank RanPAC system through its sample-space Gram matrix.

    This is algebraically equivalent to the official primal normal equations.
    Float64 is intentional: for ``M=10,000`` and fewer than ``M`` base
    examples, the float32 primal Gram matrix has a large numerical null space
    and can overwhelm small, base-selected ridge values.
    """

    values = projected_features.to(dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("projected RanPAC features must be a non-empty matrix")
    if labels.ndim != 1 or labels.shape[0] != values.shape[0]:
        raise ValueError("projected RanPAC feature/label shape mismatch")
    if values.shape[0] >= values.shape[1]:
        raise ValueError("low-rank RanPAC system requires samples < dimensions")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("projected RanPAC features are not finite")

    sample_gram = values.matmul(values.t())
    sample_gram = 0.5 * (sample_gram + sample_gram.t())
    eigenvalues, left_vectors = torch.linalg.eigh(sample_gram)
    maximum = max(float(eigenvalues[-1].item()), 1.0)
    tolerance = (
        max(int(values.shape[0]), int(values.shape[1]))
        * torch.finfo(values.dtype).eps
        * maximum
    )
    retained = eigenvalues > float(tolerance)
    if not bool(retained.any()):
        raise RuntimeError("projected RanPAC system has zero numerical rank")
    squared_singular_values = eigenvalues[retained]
    left_vectors = left_vectors[:, retained]
    singular_values = squared_singular_values.sqrt()
    right_vectors = values.t().matmul(left_vectors / singular_values[None, :])
    targets = F.one_hot(labels.long(), num_classes=int(num_classes)).to(
        dtype=values.dtype
    )
    rotated_targets = left_vectors.t().matmul(targets)
    return RanPACLowRankSystem(
        squared_singular_values=squared_singular_values,
        right_vectors=right_vectors,
        rotated_targets=rotated_targets,
        num_samples=int(values.shape[0]),
        feature_dimension=int(values.shape[1]),
    )


def solve_ranpac_from_low_rank_system(
    system: RanPACLowRankSystem, ridge: float
) -> torch.Tensor:
    """Solve the official RanPAC ridge head from a low-rank spectral system."""

    if float(ridge) <= 0.0:
        raise ValueError("RanPAC ridge must be positive")
    singular_values = system.squared_singular_values.sqrt()
    coefficients = singular_values / (
        system.squared_singular_values + float(ridge)
    )
    return system.right_vectors.matmul(
        coefficients[:, None] * system.rotated_targets
    )


def ranpac_low_rank_inverse_apply(
    system: RanPACLowRankSystem,
    right_hand_side: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Apply ``(H.T H + ridge I)^-1`` in the stable spectral basis."""

    if float(ridge) <= 0.0:
        raise ValueError("RanPAC ridge must be positive")
    rhs = right_hand_side.to(dtype=system.right_vectors.dtype)
    projection = system.right_vectors.t().matmul(rhs)
    attenuation = system.squared_singular_values / (
        float(ridge) * (system.squared_singular_values + float(ridge))
    )
    return rhs / float(ridge) - system.right_vectors.matmul(
        attenuation[:, None] * projection
    )


def ranpac_normal_equation_relative_residual(
    projected_features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    ridge: float,
    num_classes: int,
) -> float:
    """Return a scale-free residual for a fitted RanPAC normal equation."""

    values = projected_features.to(dtype=weights.dtype)
    targets = F.one_hot(labels.long(), num_classes=int(num_classes)).to(
        dtype=weights.dtype
    )
    right_hand_side = values.t().matmul(targets)
    residual = (
        values.t().matmul(values.matmul(weights))
        + float(ridge) * weights
        - right_hand_side
    )
    denominator = right_hand_side.norm().clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    return float((residual.norm() / denominator).item())


def solve_ranpac(
    gram: torch.Tensor, cross: torch.Tensor, ridge: float
) -> torch.Tensor:
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    system = 0.5 * (gram + gram.t()) + float(ridge) * identity
    return torch.linalg.solve(system, cross)


def solve_ranpac_from_eigensystem(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    cross: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    rotated = eigenvectors.t().matmul(cross)
    rotated = rotated / (eigenvalues[:, None] + float(ridge)).clamp_min(1e-12)
    return eigenvectors.matmul(rotated)


def ranpac_inverse_apply(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    right_hand_side: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Apply ``(G + ridge I)^-1`` using a frozen base eigensystem."""
    rotated = eigenvectors.t().matmul(right_hand_side)
    rotated = rotated / (eigenvalues[:, None] + float(ridge)).clamp_min(1e-12)
    return eigenvectors.matmul(rotated)


def solve_ranpac_low_rank_update(
    base_eigenvalues: torch.Tensor,
    base_eigenvectors: torch.Tensor,
    base_cross: torch.Tensor,
    added_features_h: torch.Tensor,
    added_labels: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Exact Woodbury update after adding a small support set to a base head."""
    base_weights = solve_ranpac_from_eigensystem(
        base_eigenvalues, base_eigenvectors, base_cross, ridge
    )
    if added_features_h.shape[0] == 0:
        return base_weights
    transformed = added_features_h.float()
    inverse_times_support = ranpac_inverse_apply(
        base_eigenvalues,
        base_eigenvectors,
        transformed.t(),
        ridge,
    )
    small_system = torch.eye(
        transformed.shape[0], dtype=transformed.dtype, device=transformed.device
    ) + transformed.matmul(inverse_times_support)
    targets = F.one_hot(
        added_labels.long(), num_classes=base_cross.shape[1]
    ).to(dtype=transformed.dtype)
    residual = targets - transformed.matmul(base_weights)
    correction = torch.linalg.solve(small_system, residual)
    return base_weights + inverse_times_support.matmul(correction)


def ranpac_scores_chunked(
    queries: torch.Tensor,
    weights: torch.Tensor,
    projection: Optional[torch.Tensor] = None,
    query_chunk: int = 2048,
) -> torch.Tensor:
    rows = []
    for start in range(0, queries.shape[0], query_chunk):
        transformed = ranpac_transform(
            queries[start : start + query_chunk], projection
        )
        rows.append(transformed.matmul(weights))
    return torch.cat(rows, 0)
