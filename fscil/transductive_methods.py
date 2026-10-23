"""Query-cohort methods with explicit information and memory boundaries.

No function in this module accepts query labels.  Marginal-constrained routines
receive a caller-supplied protocol vector; they never estimate it from labels.
Graph methods construct sparse kNN matrices from chunked similarity blocks and
never materialize a Q x Q affinity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransductiveOutput:
    scores: torch.Tensor
    prototypes: torch.Tensor
    metadata: Dict[str, object]


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=int(num_classes)).float()


def _check_inputs(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
) -> None:
    if support.ndim != 2 or query.ndim != 2:
        raise ValueError("support and query must be matrices")
    if support.shape[1] != query.shape[1]:
        raise ValueError("support/query feature dimensions differ")
    if support.shape[0] != support_labels.numel():
        raise ValueError("support feature/label lengths differ")
    if query.shape[0] < 1:
        raise ValueError("query cohort must be non-empty")
    present = torch.unique(support_labels).sort().values
    expected = torch.arange(num_classes, device=support_labels.device)
    if not torch.equal(present, expected):
        raise ValueError("support labels must cover contiguous classes [0, C)")


def support_prototypes(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    normalize: bool = True,
) -> torch.Tensor:
    prototypes = []
    for class_id in range(int(num_classes)):
        values = support[support_labels == class_id].float()
        if values.shape[0] == 0:
            raise ValueError("missing support class {}".format(class_id))
        prototypes.append(values.mean(0))
    result = torch.stack(prototypes, 0)
    return F.normalize(result, dim=1) if normalize else result


def query_prototype_rectification(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    baseline_scores: torch.Tensor,
    num_base_classes: int,
    alpha: float = 0.75,
    mass_threshold: float = 0.3,
    min_count: int = 5,
    logit_scale: float = 1.0,
    novel_margin_threshold: float = 0.0,
    posterior_weighting: bool = True,
    preserve_radius: bool = True,
    require_predicted_novel: bool = True,
    allow_all_novel: bool = False,
) -> TransductiveOutput:
    """Apply this work's QPR gate to an existing image-only head.

    The operator consumes scores from a classifier that was fixed before the
    query cohort was read.  It updates only novel prototypes and receives no
    labels or class-marginal vector.  The returned cosine scores are provided
    for completeness; callers using another geometry should rescore
    ``output.prototypes`` with that same fixed geometry.
    """

    if query.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("query and prototypes must be matrices")
    if query.shape[1] != prototypes.shape[1]:
        raise ValueError("query/prototype feature dimensions differ")
    if baseline_scores.shape != (query.shape[0], prototypes.shape[0]):
        raise ValueError("baseline score shape does not match query/classes")
    class_count = int(prototypes.shape[0])
    base_count = int(num_base_classes)
    if bool(allow_all_novel):
        # Diagnostic-only escape hatch for the balanced low-way format, where an
        # episode has no base block at all.  It relaxes nothing except the
        # base-count precondition: the gates, update, and radius handling below
        # are the reported operator's.  With base_count >= 1 the results are
        # bitwise identical to the default path.
        if base_count < 0 or base_count >= class_count:
            raise ValueError("QPR requires at least one novel class")
    elif base_count < 1 or base_count >= class_count:
        raise ValueError("QPR requires at least one base and one novel class")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= float(mass_threshold) <= 1.0:
        raise ValueError("mass_threshold must be in [0, 1]")
    if int(min_count) < 1:
        raise ValueError("min_count must be positive")
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be positive")
    if not 0.0 <= float(novel_margin_threshold) <= 1.0:
        raise ValueError("novel_margin_threshold must be in [0, 1]")
    if not bool(torch.isfinite(baseline_scores).all()):
        raise ValueError("baseline scores must be finite")

    probabilities = torch.softmax(
        baseline_scores.float() * float(logit_scale), dim=1
    )
    prediction = probabilities.argmax(1)
    novel_probability = probabilities[:, base_count:]
    novel_mass = novel_probability.sum(1)
    assigned_novel = novel_probability.argmax(1) + base_count
    conditional_novel = novel_probability / novel_mass.unsqueeze(1).clamp_min(1e-12)
    if conditional_novel.shape[1] > 1:
        top_two = conditional_novel.topk(2, dim=1).values
        novel_margin = top_two[:, 0] - top_two[:, 1]
    else:
        novel_margin = torch.ones_like(novel_mass)
    gate = (novel_mass >= float(mass_threshold)) & (
        novel_margin >= float(novel_margin_threshold)
    )
    if bool(require_predicted_novel):
        gate = gate & (prediction >= base_count)

    original = prototypes.float()
    normalized_original = F.normalize(original, dim=1)
    original_radius = original.norm(dim=1)
    refined = original.clone()
    admitted_counts: Dict[str, int] = {}
    eligible_classes = []
    updated_classes = []
    for class_id in range(base_count, class_count):
        selected = gate & (assigned_novel == class_id)
        count = int(selected.sum().item())
        admitted_counts[str(class_id)] = count
        if count < int(min_count):
            continue
        eligible_classes.append(class_id)
        if float(alpha) == 0.0:
            continue
        if bool(posterior_weighting):
            weights = probabilities[selected, class_id]
            weights = weights / weights.sum().clamp_min(1e-12)
        else:
            weights = torch.full(
                (count,),
                1.0 / float(count),
                dtype=query.dtype,
                device=query.device,
            )
        query_mean = F.normalize(
            weights.unsqueeze(0).matmul(query[selected].float()), dim=1
        ).squeeze(0)
        refined_direction = F.normalize(
            (
                (1.0 - float(alpha)) * normalized_original[class_id]
                + float(alpha) * query_mean
            ).unsqueeze(0),
            dim=1,
        ).squeeze(0)
        radius = (
            original_radius[class_id]
            if bool(preserve_radius)
            else original_radius.new_tensor(1.0)
        )
        refined[class_id] = refined_direction * radius
        updated_classes.append(class_id)

    if not torch.equal(refined[:base_count], original[:base_count]):
        raise RuntimeError("QPR modified a base-class prototype")
    return TransductiveOutput(
        scores=F.normalize(query.float(), dim=1).matmul(refined.t()),
        prototypes=refined,
        metadata={
            "method": "QPR conservative novel-prototype rectification",
            "innovation_scope": "introduced in this work",
            "implementation_origin": (
                "authors' unpublished internal development repository"
            ),
            "alpha": float(alpha),
            "mass_threshold": float(mass_threshold),
            "min_count": int(min_count),
            "logit_scale": float(logit_scale),
            "novel_margin_threshold": float(novel_margin_threshold),
            "posterior_weighting": bool(posterior_weighting),
            "require_predicted_novel": bool(require_predicted_novel),
            "allow_all_novel": bool(allow_all_novel),
            "base_class_count": base_count,
            "gates_active": base_count >= 1,
            "gated_query_count": int(gate.sum().item()),
            "gated_novel_margin_mean": (
                float(novel_margin[gate].mean().item()) if bool(gate.any()) else None
            ),
            "eligible_class_count": len(eligible_classes),
            "updated_class_count": len(updated_classes),
            "eligible_class_ids": eligible_classes,
            "updated_class_ids": updated_classes,
            "admitted_counts": admitted_counts,
            "updates_base_prototypes": False,
            "preserves_input_prototype_radius": bool(preserve_radius),
            "uses_query_labels": False,
            "uses_query_marginal_constraint": False,
        },
    )


def squared_euclidean(
    values: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    value_norm = values.float().square().sum(1, keepdim=True)
    prototype_norm = prototypes.float().square().sum(1)[None, :]
    return (
        value_norm
        + prototype_norm
        - 2.0 * values.float().matmul(prototypes.float().t())
    ).clamp_min(0.0)


def _distance_logits(
    values: torch.Tensor, weights: torch.Tensor, temperature: float
) -> torch.Tensor:
    # Algebraically identical to the official TIM implementation.
    return float(temperature) * (
        values.float().matmul(weights.float().t())
        - 0.5 * weights.float().square().sum(1)[None, :]
        - 0.5 * values.float().square().sum(1, keepdim=True)
    )


def tim_gd(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
    iterations: int = 150,
    temperature: float = 15.0,
    learning_rate: float = 1e-4,
    cross_entropy_weight: float = 0.1,
    marginal_entropy_weight: float = 1.0,
    conditional_entropy_weight: float = 0.1,
) -> TransductiveOutput:
    """TIM-GD using the official loss and normalized frozen features."""

    _check_inputs(support, support_labels, query, num_classes)
    weights = support_prototypes(
        support, support_labels, num_classes, normalize=False
    ).detach()
    weights.requires_grad_(True)
    optimizer = torch.optim.Adam([weights], lr=float(learning_rate))
    targets = _one_hot(support_labels, num_classes)
    final_loss = float("nan")
    for _ in range(int(iterations)):
        logits_support = _distance_logits(support, weights, temperature)
        logits_query = _distance_logits(query, weights, temperature)
        support_probability = torch.softmax(logits_support, dim=1)
        query_probability = torch.softmax(logits_query, dim=1)
        cross_entropy = -(
            targets * torch.log(support_probability.clamp_min(1e-12))
        ).sum(1).mean()
        conditional_entropy = -(
            query_probability
            * torch.log(query_probability.clamp_min(1e-12))
        ).sum(1).mean()
        marginal = query_probability.mean(0)
        marginal_entropy = -(
            marginal * torch.log(marginal.clamp_min(1e-12))
        ).sum()
        loss = (
            float(cross_entropy_weight) * cross_entropy
            - float(marginal_entropy_weight) * marginal_entropy
            + float(conditional_entropy_weight) * conditional_entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    detached = weights.detach()
    return TransductiveOutput(
        scores=_distance_logits(query, detached, temperature),
        prototypes=detached,
        metadata={
            "method": "TIM-GD",
            "iterations": int(iterations),
            "temperature": float(temperature),
            "learning_rate": float(learning_rate),
            "loss_weights": [
                float(cross_entropy_weight),
                float(marginal_entropy_weight),
                float(conditional_entropy_weight),
            ],
            "final_objective": final_loss,
            "uses_query_marginal_constraint": False,
        },
    )


def _log_sinkhorn(
    logits: torch.Tensor,
    column_mass: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Return a positive plan with unit row mass and prescribed column mass."""

    if logits.ndim != 2:
        raise ValueError("Sinkhorn logits must be a matrix")
    rows, columns = logits.shape
    mass = column_mass.float().to(logits.device)
    if mass.shape != (columns,) or bool((mass <= 0).any()):
        raise ValueError("column mass must be a positive length-C vector")
    mass = mass * (float(rows) / mass.sum())
    log_plan = logits.float()
    log_mass = mass.log()
    for _ in range(int(iterations)):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan + (
            log_mass - torch.logsumexp(log_plan, dim=0)
        )[None, :]
    log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
    return log_plan


def marginal_constrained_scores(
    scores: torch.Tensor,
    class_mass: torch.Tensor,
    temperature: float = 0.1,
    iterations: int = 50,
) -> torch.Tensor:
    """I3 decoder; ``class_mass`` must come from a declared protocol constant."""

    scale = scores.float().std().clamp_min(1e-8)
    return _log_sinkhorn(
        scores.float() / (scale * float(temperature)),
        class_mass,
        iterations,
    )


def pt_map(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
    assignment_scale: float = 10.0,
    update_rate: float = 0.2,
    iterations: int = 20,
    class_mass: Optional[torch.Tensor] = None,
    sinkhorn_iterations: int = 100,
) -> TransductiveOutput:
    """MAP-OT core from PT-MAP, with its marginal assumption explicit.

    If ``class_mass`` is absent the original equipartition assumption is used.
    A non-uniform vector is reserved for the separately reported I3 oracle.
    PT-MAP's power transform is intentionally absent: frozen CLIP embeddings
    are signed, whereas the released square-root transform assumes non-negative
    backbone activations.  Benchmark tables therefore use the honest label
    ``MAP-OT (PT-MAP core)`` rather than ``PT-MAP``.
    """

    _check_inputs(support, support_labels, query, num_classes)
    prototypes = support_prototypes(
        support, support_labels, num_classes, normalize=True
    )
    support_assignment = _one_hot(support_labels, num_classes)
    if class_mass is None:
        mass = torch.ones(num_classes, device=query.device)
        prior_name = "uniform_equipartition"
    else:
        mass = class_mass.float().to(query.device)
        prior_name = "caller_supplied_protocol_marginal"
    query_assignment = None
    for _ in range(int(iterations)):
        cost = squared_euclidean(query, prototypes)
        query_assignment = _log_sinkhorn(
            -float(assignment_scale) * cost,
            mass,
            int(sinkhorn_iterations),
        ).exp()
        numerator = (
            support_assignment.t().matmul(support.float())
            + query_assignment.t().matmul(query.float())
        )
        denominator = (
            support_assignment.sum(0) + query_assignment.sum(0)
        ).clamp_min(1e-8)[:, None]
        estimate = F.normalize(numerator / denominator, dim=1)
        prototypes = F.normalize(
            prototypes + float(update_rate) * (estimate - prototypes), dim=1
        )
    assert query_assignment is not None
    # The released method predicts with the final transport plan, not an
    # unconstrained nearest-centroid pass after the last prototype update.
    final_log_plan = _log_sinkhorn(
        -float(assignment_scale) * squared_euclidean(query, prototypes),
        mass,
        int(sinkhorn_iterations),
    )
    return TransductiveOutput(
        scores=final_log_plan,
        prototypes=prototypes,
        metadata={
            "method": "MAP-OT (PT-MAP core; no power transform)",
            "iterations": int(iterations),
            "assignment_scale": float(assignment_scale),
            "update_rate": float(update_rate),
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "marginal_assumption": prior_name,
            "power_transform_applied": False,
            "power_transform_reason": "signed_CLIP_embeddings",
        },
    )


def _protolp_paper_graph_scores(
    assignment: torch.Tensor,
    support_targets: torch.Tensor,
    support_count: int,
    graph_regularization: float,
) -> torch.Tensor:
    """Eq. 14 without materialising its sample-by-sample graph."""
    labelled_assignment = assignment[: int(support_count)]
    delta = assignment.sum(0).clamp_min(1e-8)
    gram = assignment.t().matmul(assignment)
    # Z^T(I - Z Lambda^-1 Z^T)Z
    graph_regularizer = gram - gram.matmul(gram / delta[:, None])
    graph_regularizer = 0.5 * (graph_regularizer + graph_regularizer.t())
    system = (
        labelled_assignment.t().matmul(labelled_assignment)
        + float(graph_regularization) * graph_regularizer
    )
    system = 0.5 * (system + system.t())
    system = system + 1e-5 * torch.eye(
        system.shape[0], device=system.device, dtype=system.dtype
    )
    rhs = labelled_assignment.t().matmul(support_targets)
    coefficients = torch.linalg.solve(system, rhs)
    propagated = assignment.matmul(coefficients).clamp_min(0.0)
    return propagated / propagated.sum(1, keepdim=True).clamp_min(1e-8)


def _protolp_assignment(
    all_values: torch.Tensor,
    prototypes: torch.Tensor,
    support_targets: torch.Tensor,
    support_count: int,
    assignment_scale: float,
) -> torch.Tensor:
    """ProtoLP Eq. 4: labelled rows are fixed to their one-hot labels."""
    assignment = torch.softmax(
        -float(assignment_scale) * squared_euclidean(all_values, prototypes),
        dim=1,
    )
    labelled = int(support_count)
    if support_targets.shape != assignment[:labelled].shape:
        raise ValueError("ProtoLP support target shape does not match support rows")
    assignment[:labelled] = support_targets.to(
        device=assignment.device, dtype=assignment.dtype
    )
    return assignment


def protolp_paper(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
    graph_regularization: float = 1.0,
    update_rate: float = 0.2,
    iterations: int = 50,
    assignment_scale: float = 1.0,
) -> TransductiveOutput:
    """ProtoLP Algorithm 1 / Eqs. 4 and 9--17 without a marginal prior.

    Equation 4 clamps labelled-support assignments to their one-hot labels and
    applies row-normalized soft assignments only to query rows.  The published
    primary method does not require a uniform query-class marginal.  The C-by-C
    system is algebraically identical to Eq. 14, so no sample-by-sample graph is
    formed.  Frozen CLIP features replace ProtoLP's original features; the
    assignment scale may be selected on class-disjoint base episodes to account
    for that representation-scale change without using target data.
    """

    _check_inputs(support, support_labels, query, num_classes)
    prototypes = support_prototypes(
        support, support_labels, num_classes, normalize=True
    )
    support_targets = _one_hot(support_labels, num_classes)
    all_values = torch.cat([support.float(), query.float()], 0)
    query_scores = None
    for _ in range(int(iterations)):
        assignment = _protolp_assignment(
            all_values,
            prototypes,
            support_targets,
            support.shape[0],
            assignment_scale,
        )
        propagated = _protolp_paper_graph_scores(
            assignment,
            support_targets,
            support.shape[0],
            graph_regularization,
        )
        query_scores = propagated[support.shape[0] :]
        # Eq. 16 uses the propagated soft labels for both labelled and query
        # samples; Eq. 17 then applies the stated exponential update.
        estimate = propagated.t().matmul(all_values)
        estimate = estimate / propagated.sum(0).clamp_min(1e-8)[:, None]
        prototypes = prototypes + float(update_rate) * (estimate - prototypes)
    assert query_scores is not None
    return TransductiveOutput(
        scores=torch.log(query_scores.clamp_min(1e-12)),
        prototypes=prototypes,
        metadata={
            "method": "ProtoLP paper-equation reconstruction (no marginal)",
            "iterations": int(iterations),
            "graph_regularization": float(graph_regularization),
            "update_rate": float(update_rate),
            "assignment_scale": float(assignment_scale),
            "support_assignment": "one_hot_labels_per_Eq_4",
            "uses_query_marginal_constraint": False,
            "paper_equations": [4, 9, 11, 12, 13, 14, 16, 17],
            "matched_feature_reconstruction": True,
            "largest_graph_shape": [int(num_classes), int(num_classes)],
            "dense_sample_graph_constructed": False,
        },
    )


def protolp_uniform(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
    graph_regularization: float = 1.0,
    update_rate: float = 0.2,
    iterations: int = 50,
    assignment_scale: float = 1.0,
    sinkhorn_iterations: int = 100,
) -> TransductiveOutput:
    """ProtoLP paper equations plus its optional uniform Sinkhorn projection.

    This is deliberately separate from :func:`protolp_paper`: Section 3.4 and
    Tables 6--8 of the paper present the uniform projection as an optional prior,
    while Algorithm 1 is the no-Sinkhorn method.  Only the query rows are
    projected; the underlying assignment and Eq. 14 graph remain unchanged.
    """
    _check_inputs(support, support_labels, query, num_classes)
    prototypes = support_prototypes(
        support, support_labels, num_classes, normalize=True
    )
    support_targets = _one_hot(support_labels, num_classes)
    all_values = torch.cat([support.float(), query.float()], 0)
    uniform_mass = torch.ones(
        num_classes, device=query.device, dtype=torch.float32
    )
    final_query_plan = None
    for _ in range(int(iterations)):
        assignment = _protolp_assignment(
            all_values,
            prototypes,
            support_targets,
            support.shape[0],
            assignment_scale,
        )
        propagated = _protolp_paper_graph_scores(
            assignment,
            support_targets,
            support.shape[0],
            graph_regularization,
        )
        propagated_query = _log_sinkhorn(
            torch.log(
                propagated[support.shape[0] :].clamp_min(1e-12)
            ),
            uniform_mass,
            int(sinkhorn_iterations),
        ).exp()
        responsibility = torch.cat(
            [propagated[: support.shape[0]], propagated_query], 0
        )
        estimate = responsibility.t().matmul(all_values)
        estimate = estimate / responsibility.sum(0).clamp_min(1e-8)[:, None]
        prototypes = prototypes + float(update_rate) * (estimate - prototypes)
        final_query_plan = propagated_query
    assert final_query_plan is not None
    final_log_plan = torch.log(final_query_plan.clamp_min(1e-12))
    if not bool(torch.isfinite(final_query_plan).all()):
        raise RuntimeError("ProtoLP-uniform produced non-finite assignments")
    return TransductiveOutput(
        scores=final_log_plan,
        prototypes=prototypes,
        metadata={
            "method": "ProtoLP paper equations + optional uniform Sinkhorn",
            "iterations": int(iterations),
            "graph_regularization": float(graph_regularization),
            "update_rate": float(update_rate),
            "assignment_scale": float(assignment_scale),
            "support_assignment": "one_hot_labels_per_Eq_4",
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "marginal_assumption": "uniform_equipartition",
            "uses_query_marginal_constraint": True,
            "paper_equations": [4, 9, 11, 12, 13, 14, 16, 17, 18],
            "matched_feature_reconstruction": True,
            "largest_graph_shape": [int(num_classes), int(num_classes)],
            "dense_sample_graph_constructed": False,
        },
    )


# Backward-compatible names exist only so explicitly superseded v1 artifacts can
# still be inspected.  New code and paper tables must use ``protolp_paper``.
def protolp_unconstrained(*args, **kwargs) -> TransductiveOutput:
    return protolp_paper(*args, **kwargs)


def protolp(*args, **kwargs) -> TransductiveOutput:
    return protolp_paper(*args, **kwargs)


def chunked_sparse_cosine_knn(
    values: torch.Tensor,
    neighbors: int,
    chunk_size: int = 512,
    symmetric: bool = False,
    positive_only: bool = False,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Create sparse cosine kNN edges without allocating an N x N matrix."""

    normalized = F.normalize(values.float(), dim=1)
    count = int(normalized.shape[0])
    if count < 2:
        raise ValueError("kNN needs at least two nodes")
    k = min(int(neighbors), count - 1)
    if k < 1:
        raise ValueError("neighbors must be positive")
    row_parts = []
    column_parts = []
    value_parts = []
    maximum_block_elements = 0
    for start in range(0, count, int(chunk_size)):
        stop = min(start + int(chunk_size), count)
        similarity = normalized[start:stop].matmul(normalized.t())
        maximum_block_elements = max(maximum_block_elements, similarity.numel())
        local = torch.arange(stop - start, device=values.device)
        global_index = torch.arange(start, stop, device=values.device)
        similarity[local, global_index] = -torch.inf
        score, index = torch.topk(similarity, k=k, dim=1)
        if positive_only:
            score = score.clamp_min(0.0)
        rows = global_index[:, None].expand(-1, k).reshape(-1)
        row_parts.append(rows)
        column_parts.append(index.reshape(-1))
        value_parts.append(score.reshape(-1))
    rows = torch.cat(row_parts)
    columns = torch.cat(column_parts)
    edge_values = torch.cat(value_parts)
    if symmetric:
        original_rows = rows
        rows = torch.cat([original_rows, columns])
        columns = torch.cat([columns, original_rows])
        edge_values = torch.cat([edge_values, edge_values])
    matrix = torch.sparse_coo_tensor(
        torch.stack([rows, columns], 0),
        edge_values,
        size=(count, count),
        device=values.device,
    ).coalesce()
    return matrix, {
        "nodes": count,
        "neighbors": k,
        "directed_edges_before_coalesce": int(rows.numel()),
        "stored_edges": int(matrix._nnz()),
        "max_similarity_block_elements": int(maximum_block_elements),
        "max_similarity_block_bytes": int(
            maximum_block_elements * normalized.element_size()
        ),
    }


def chunked_sparse_euclidean_knn(
    values: torch.Tensor,
    neighbors: int,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Create a directed binary Euclidean kNN graph in bounded memory.

    ``neighbors`` counts non-self neighbours.  LaplacianShot's public ``knn``
    argument includes the sample itself, so its caller passes ``knn - 1``.
    """

    values = values.float()
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("kNN needs at least two nodes")
    k = min(int(neighbors), count - 1)
    if k < 1:
        raise ValueError("neighbors must be positive")
    norms = values.square().sum(1)
    row_parts = []
    column_parts = []
    maximum_block_elements = 0
    for start in range(0, count, int(chunk_size)):
        stop = min(start + int(chunk_size), count)
        distances = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * values[start:stop].matmul(values.t())
        ).clamp_min(0.0)
        maximum_block_elements = max(maximum_block_elements, distances.numel())
        local = torch.arange(stop - start, device=values.device)
        global_index = torch.arange(start, stop, device=values.device)
        distances[local, global_index] = torch.inf
        index = torch.topk(distances, k=k, dim=1, largest=False).indices
        row_parts.append(global_index[:, None].expand(-1, k).reshape(-1))
        column_parts.append(index.reshape(-1))
    rows = torch.cat(row_parts)
    columns = torch.cat(column_parts)
    matrix = torch.sparse_coo_tensor(
        torch.stack([rows, columns], 0),
        torch.ones(rows.numel(), dtype=torch.float32, device=values.device),
        size=(count, count),
        device=values.device,
    ).coalesce()
    return matrix, {
        "nodes": count,
        "neighbors_excluding_self": k,
        "directed_edges_before_coalesce": int(rows.numel()),
        "stored_edges": int(matrix._nnz()),
        "max_distance_block_elements": int(maximum_block_elements),
        "max_distance_block_bytes": int(
            maximum_block_elements * values.element_size()
        ),
    }


def _symmetric_normalize_sparse(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.coalesce()
    index = matrix.indices()
    values = matrix.values().clamp_min(0.0)
    degree = torch.zeros(
        matrix.shape[0], device=values.device, dtype=values.dtype
    )
    degree.scatter_add_(0, index[0], values)
    inverse_sqrt = degree.clamp_min(1e-8).rsqrt()
    normalized = values * inverse_sqrt[index[0]] * inverse_sqrt[index[1]]
    return torch.sparse_coo_tensor(
        index, normalized, matrix.shape, device=matrix.device
    ).coalesce()


def sparse_graph_repair(
    prototypes: torch.Tensor,
    query: torch.Tensor,
    neighbors: int = 10,
    propagation_alpha: float = 0.9,
    propagation_iterations: int = 20,
    prototype_anchor: float = 0.3,
    chunk_size: int = 512,
) -> TransductiveOutput:
    """Sparse version of the submitted graph prototype-repair intervention."""

    class_count = int(prototypes.shape[0])
    nodes = torch.cat(
        [F.normalize(prototypes.float(), dim=1), F.normalize(query.float(), dim=1)],
        0,
    )
    affinity, graph_metadata = chunked_sparse_cosine_knn(
        nodes,
        neighbors=neighbors,
        chunk_size=chunk_size,
        symmetric=True,
        positive_only=True,
    )
    transition = _symmetric_normalize_sparse(affinity)
    seeds = torch.zeros(
        nodes.shape[0], class_count, device=query.device, dtype=torch.float32
    )
    seeds[:class_count] = torch.eye(
        class_count, device=query.device, dtype=torch.float32
    )
    propagated = seeds.clone()
    for _ in range(int(propagation_iterations)):
        propagated = (
            float(propagation_alpha)
            * torch.sparse.mm(transition, propagated)
            + (1.0 - float(propagation_alpha)) * seeds
        )
    responsibility = propagated[class_count:].clamp_min(0.0)
    responsibility = responsibility / responsibility.sum(
        0, keepdim=True
    ).clamp_min(1e-8)
    estimate = responsibility.t().matmul(F.normalize(query.float(), dim=1))
    repaired = F.normalize(
        float(prototype_anchor) * F.normalize(prototypes.float(), dim=1)
        + (1.0 - float(prototype_anchor)) * F.normalize(estimate, dim=1),
        dim=1,
    )
    return TransductiveOutput(
        scores=F.normalize(query.float(), dim=1).matmul(repaired.t()),
        prototypes=repaired,
        metadata={
            "method": "sparse_graph_repair",
            "neighbors": int(neighbors),
            "propagation_alpha": float(propagation_alpha),
            "propagation_iterations": int(propagation_iterations),
            "prototype_anchor": float(prototype_anchor),
            "dense_query_affinity_constructed": False,
            "graph": graph_metadata,
        },
    )


def _laplacian_bound_update(
    unary: torch.Tensor,
    affinity: torch.Tensor,
    regularization: float,
    iterations: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Official LaplacianShot bound update, including its stopping rule."""

    probability = torch.softmax(-unary.float(), dim=1)
    previous_energy = torch.tensor(
        float("inf"), device=unary.device, dtype=torch.float32
    )
    converged = False
    energy = torch.tensor(float("nan"), device=unary.device)
    completed = 0
    for iteration in range(int(iterations)):
        neighbor_term = torch.sparse.mm(affinity, probability)
        probability = torch.softmax(
            -unary.float() + float(regularization) * neighbor_term, dim=1
        )
        energy = (
            probability * torch.log(probability.clamp_min(1e-20))
            + unary.float() * probability
            - float(regularization) * neighbor_term * probability
        ).sum()
        completed = iteration + 1
        if iteration > 1 and bool(
            torch.abs(energy - previous_energy)
            <= 1e-6 * torch.abs(previous_energy)
        ):
            converged = True
            break
        previous_energy = energy.detach().clone()
    return probability, {
        "iterations_requested": int(iterations),
        "iterations_completed": completed,
        "converged_by_official_energy_rule": converged,
        "final_energy": float(energy.item()),
    }


def laplacian_shot(
    prototypes: torch.Tensor,
    query: torch.Tensor,
    neighbors: int = 3,
    regularization: float = 1.0,
    iterations: int = 20,
    chunk_size: int = 512,
) -> TransductiveOutput:
    """LaplacianShot core without its CL2N/prototype-rectification front end.

    This compatibility function exists for already generated artifacts.  New
    benchmark tables use :func:`laplacian_shot_full` and label this older
    function as a core-only ablation.
    """

    unary = squared_euclidean(query, prototypes)
    affinity, graph_metadata = chunked_sparse_cosine_knn(
        query,
        neighbors=neighbors,
        chunk_size=chunk_size,
        symmetric=False,
        positive_only=False,
    )
    affinity = torch.sparse_coo_tensor(
        affinity.indices(),
        torch.ones_like(affinity.values()),
        affinity.shape,
        device=query.device,
    ).coalesce()
    probability, update_metadata = _laplacian_bound_update(
        unary, affinity, regularization, iterations
    )
    return TransductiveOutput(
        scores=torch.log(probability.clamp_min(1e-12)),
        prototypes=prototypes,
        metadata={
            "method": "LaplacianShot core-only ablation",
            "neighbors": int(neighbors),
            "regularization": float(regularization),
            "iterations": int(iterations),
            "binary_directed_knn": True,
            "dense_query_affinity_constructed": False,
            "includes_cl2n": False,
            "includes_prototype_rectification": False,
            "bound_update": update_metadata,
            "graph": graph_metadata,
        },
    )


def laplacian_shot_full(
    support: torch.Tensor,
    support_labels: torch.Tensor,
    query: torch.Tensor,
    num_classes: int,
    train_mean: torch.Tensor,
    knn: int = 3,
    regularization: float = 1.0,
    iterations: int = 20,
    prototype_rectification: bool = True,
    rectification_temperature: float = 10.0,
    chunk_size: int = 512,
) -> TransductiveOutput:
    """Full LaplacianShot inference on frozen features.

    This follows the released evaluation pipeline: CL2N centering by a training
    mean, optional query-shift/prototype rectification, squared-Euclidean unary
    costs, a directed binary Euclidean kNN graph, and the official bound update.
    The release assumes every class receives a rectification assignment.  In the
    much wider all-seen FSCIL setting that need not hold, so an unassigned class
    retains its CL2N support mean instead of being silently dropped.
    """

    _check_inputs(support, support_labels, query, num_classes)
    mean = train_mean.float().reshape(1, -1).to(query.device)
    if mean.shape[1] != query.shape[1]:
        raise ValueError("LaplacianShot train mean has the wrong dimension")
    centered_support = F.normalize(support.float() - mean, dim=1)
    centered_query = F.normalize(query.float() - mean, dim=1)
    initial = support_prototypes(
        centered_support, support_labels, num_classes, normalize=False
    )
    repaired = initial
    shifted_query = centered_query
    assigned_counts = [0 for _ in range(int(num_classes))]
    missing_classes = []
    if bool(prototype_rectification):
        shift = centered_support.mean(0) - centered_query.mean(0)
        shifted_query = centered_query + shift.unsqueeze(0)
        augmented = torch.cat([centered_support, shifted_query], 0)
        cosine = F.normalize(augmented, dim=1).matmul(
            F.normalize(initial, dim=1).t()
        )
        weights = torch.softmax(
            float(rectification_temperature) * cosine, dim=1
        )
        assignment = cosine.argmax(1)
        repaired_rows = []
        for class_id in range(int(num_classes)):
            selected = assignment == class_id
            count = int(selected.sum().item())
            assigned_counts[class_id] = count
            if count == 0:
                # Necessary only outside the narrow native few-shot regime.
                repaired_rows.append(initial[class_id])
                missing_classes.append(class_id)
            else:
                repaired_rows.append(
                    (
                        weights[selected, class_id, None]
                        * augmented[selected]
                    ).mean(0)
                )
        repaired = torch.stack(repaired_rows, 0)

    unary = squared_euclidean(shifted_query, repaired)
    official_knn = min(int(knn), int(shifted_query.shape[0]))
    if official_knn < 2:
        raise ValueError("LaplacianShot knn must include self and one neighbour")
    affinity, graph_metadata = chunked_sparse_euclidean_knn(
        shifted_query,
        neighbors=official_knn - 1,
        chunk_size=chunk_size,
    )
    probability, update_metadata = _laplacian_bound_update(
        unary, affinity, regularization, iterations
    )
    return TransductiveOutput(
        scores=torch.log(probability.clamp_min(1e-12)),
        prototypes=repaired,
        metadata={
            "method": "LaplacianShot full released inference pipeline",
            "knn_including_self": official_knn,
            "regularization": float(regularization),
            "includes_cl2n": True,
            "includes_prototype_rectification": bool(prototype_rectification),
            "rectification_temperature": float(rectification_temperature),
            "rectification_assigned_counts": assigned_counts,
            "rectification_missing_class_count": len(missing_classes),
            "rectification_missing_class_ids": missing_classes,
            "missing_class_fallback": "retain_CL2N_support_mean",
            "binary_directed_euclidean_knn": True,
            "dense_query_affinity_constructed": False,
            "bound_update": update_metadata,
            "graph": graph_metadata,
            "uses_query_labels": False,
            "uses_query_marginal_constraint": False,
        },
    )
