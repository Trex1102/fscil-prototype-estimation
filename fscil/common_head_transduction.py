"""Explicit common-head adaptations for the two-parent comparison.

Every comparator starts at the supplied prototypes in a fixed Euclidean
coordinate system. ShrinkSel is represented by a linear whitening transform,
normalized by a single base-only scalar; prototypes are never row-normalized
on initialization. Method-specific updates and preprocessing are retained.
Full LaplacianShot applies its native CL2N to the supplied coordinates and
parent directions, using only the transformed base-training mean.
The original native-method implementations remain unchanged.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from .geometry import CovarianceEigensystem, inverse_eigenvalues
from .transductive_methods import (
    TransductiveOutput, _distance_logits, _laplacian_bound_update,
    _log_sinkhorn, _one_hot, _protolp_assignment, _protolp_paper_graph_scores,
    chunked_sparse_euclidean_knn, laplacian_shot, squared_euclidean,
)


METHODS = ("tim_gd", "laplacian_shot", "laplacian_shot_core",
           "protolp_paper", "ptmap_uniform")


def head_transform(covariance: CovarianceEigensystem, shrinkage: float,
                   ridge_gamma: float) -> torch.Tensor:
    """Return A with AA^T = G / mean(eigenvalues(G))."""
    inverse = inverse_eigenvalues(covariance, shrinkage, ridge_gamma)
    weights = (inverse / inverse.mean()).sqrt()
    return covariance.eigenvectors * weights.unsqueeze(0)


def initial_scores(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return -0.5 * squared_euclidean(query, prototypes)


def common_head_method(method: str, support: torch.Tensor,
                       support_labels: torch.Tensor, query: torch.Tensor,
                       prototypes: torch.Tensor,
                       parameters: Mapping[str, object],
                       train_mean: Optional[torch.Tensor] = None) -> TransductiveOutput:
    """Run a locked comparator from explicitly supplied parent prototypes.

Only support labels are accepted; no query labels or empirical marginals
are exposed. Inputs are already in the fixed parent coordinate system.
"""
    if method not in METHODS:
        raise ValueError("Unknown common-head method: " + method)
    classes = int(prototypes.shape[0])
    if query.shape[1] != prototypes.shape[1] or support.shape[1] != query.shape[1]:
        raise ValueError("Head, support, and query dimensions differ")
    if not torch.equal(torch.unique(support_labels).sort().values,
                       torch.arange(classes, device=support_labels.device)):
        raise ValueError("Support labels must cover every parent prototype")
    selected = parameters["selected"]
    metadata = {"method": method, "explicit_parent_initialization": True,
                "uses_query_labels": False, "fixed_parent_geometry": True}
    p = prototypes.detach().clone().float()
    targets = _one_hot(support_labels, classes)
    if method == "tim_gd":
        tim = parameters["tim"]
        temperature = float(tim["temperature"])
        with torch.enable_grad():
            p.requires_grad_(True)
            optimizer = torch.optim.Adam([p], lr=float(tim["learning_rate"]))
            for _ in range(int(tim["iterations"])):
                ps = torch.softmax(_distance_logits(support, p, temperature), 1)
                pq = torch.softmax(_distance_logits(query, p, temperature), 1)
                ce = -(targets * ps.clamp_min(1e-12).log()).sum(1).mean()
                conditional = -(pq * pq.clamp_min(1e-12).log()).sum(1).mean()
                marginal = pq.mean(0)
                entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
                a, b, c = map(float, tim["loss_weights"])
                loss = a * ce - b * entropy + c * conditional
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        p = p.detach()
        scores = _distance_logits(query, p, temperature)
    elif method == "protolp_paper":
        regularization, scale = map(float, selected["protolp_paper"]["values"])
        values = torch.cat([support.float(), query.float()], 0)
        for _ in range(50):
            assignment = _protolp_assignment(values, p, targets, support.shape[0], scale)
            propagated = _protolp_paper_graph_scores(
                assignment, targets, support.shape[0], regularization)
            estimate = propagated.t().matmul(values)
            estimate /= propagated.sum(0).clamp_min(1e-8)[:, None]
            p = p + 0.2 * (estimate - p)
        scores = propagated[support.shape[0]:].clamp_min(1e-12).log()
    elif method == "ptmap_uniform":
        scale = float(selected["ptmap"]["values"][0])
        mass = torch.ones(classes, device=query.device)
        for _ in range(20):
            assignment = _log_sinkhorn(
                -scale * squared_euclidean(query, p), mass, 100).exp()
            estimate = (targets.t().matmul(support.float())
                        + assignment.t().matmul(query.float()))
            estimate /= (targets.sum(0) + assignment.sum(0)).clamp_min(1e-8)[:, None]
            estimate = F.normalize(estimate, dim=1)
            p = F.normalize(p + 0.2 * (estimate - p), dim=1)
        scores = _log_sinkhorn(-scale * squared_euclidean(query, p), mass, 100)
        metadata["marginal_assumption"] = "uniform_equipartition"
    elif method == "laplacian_shot_core":
        neighbors, regularization = selected["laplacian_core"]["values"]
        result = laplacian_shot(p, query, neighbors=int(neighbors),
                                regularization=float(regularization))
        return TransductiveOutput(result.scores, result.prototypes,
                                  {**result.metadata, **metadata})
    else:
        knn, regularization, rectify = selected["laplacian"]["values"]
        if train_mean is not None:
            center = train_mean.float().reshape(1, -1).to(query.device)
            support = F.normalize(support.float() - center, dim=1)
            query = F.normalize(query.float() - center, dim=1)
            # Apply native centering to the supplied parent directions while
            # retaining their initial radii. Rectification may then update
            # these prototypes, as in the native full pipeline.
            p = F.normalize(p - center, dim=1) * p.norm(dim=1, keepdim=True)
        shifted = query
        if bool(int(rectify)):
            shifted = query + (support.mean(0) - query.mean(0)).unsqueeze(0)
            augmented = torch.cat([support, shifted], 0)
            cosine = F.normalize(augmented, dim=1).matmul(F.normalize(p, dim=1).t())
            weights = torch.softmax(10.0 * cosine, 1)
            assignment = cosine.argmax(1)
            p = torch.stack([
                (weights[assignment == c, c, None] * augmented[assignment == c]).mean(0)
                if bool((assignment == c).any()) else p[c]
                for c in range(classes)
            ])
        affinity, graph = chunked_sparse_euclidean_knn(
            shifted, neighbors=min(int(knn), shifted.shape[0]) - 1)
        probability, bound = _laplacian_bound_update(
            squared_euclidean(shifted, p), affinity, float(regularization), 20)
        scores = probability.clamp_min(1e-12).log()
        metadata.update(native_cl2n_retained=train_mean is not None,
                        prototype_rectification=bool(int(rectify)), graph=graph,
                        bound_update=bound)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError(method + " returned non-finite scores")
    return TransductiveOutput(scores, p, metadata)
