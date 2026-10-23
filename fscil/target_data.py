"""Load and validate target-evaluation data after parameters have been frozen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch

from .constants import KNOWN_TEST_COUNTS
from .utils import file_sha256, tensor_sha256


@dataclass(frozen=True)
class TargetEvaluationData:
    dataset: str
    backbone: Mapping[str, object]
    num_base: int
    num_total: int
    increment: int
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    standard_support_abs: torch.Tensor
    nested_support_abs: torch.Tensor
    support_seeds: List[int]
    support_sha256: str
    protocol_sha256: str
    feature_file_sha256: str
    selection_source_sha256: str
    known_test_counts: Optional[torch.Tensor]


def load_target_evaluation_data(
    feature_cache: Path,
    protocol_bundle: Path,
    device: str,
    verify_file: bool = True,
) -> TargetEvaluationData:
    protocol = torch.load(str(protocol_bundle), map_location="cpu")
    cached = torch.load(str(feature_cache), map_location="cpu")
    train_features_cpu = cached["train"].float().contiguous()
    test_features_cpu = cached["test"].float().contiguous()
    cached_test_labels = cached["tl"].long().contiguous()
    cached_metadata = cached.get("metadata", {})
    expected = protocol["feature_cache"]
    if feature_cache.name != expected["basename"]:
        raise RuntimeError("feature-cache basename does not match locked protocol")
    if verify_file and file_sha256(feature_cache) != expected["file_sha256"]:
        raise RuntimeError("feature-cache file hash mismatch")
    if tensor_sha256(train_features_cpu) != expected["train_tensor_sha256"]:
        raise RuntimeError("train-feature tensor hash mismatch")
    if tensor_sha256(test_features_cpu) != expected["test_tensor_sha256"]:
        raise RuntimeError("test-feature tensor hash mismatch")
    if tensor_sha256(cached_test_labels) != expected["test_label_sha256"]:
        raise RuntimeError("cached test-label hash mismatch")
    selection_source = str(
        cached_metadata.get(
            "pretarget_source_feature_identity", expected["file_sha256"]
        )
    )
    if selection_source != str(
        expected.get("selection_source_sha256", expected["file_sha256"])
    ):
        raise RuntimeError("selector-source identity disagrees with protocol")
    if not torch.equal(cached_test_labels, protocol["test_labels"].long()):
        raise RuntimeError("protocol/cache test labels disagree")
    train_labels = protocol["train_labels"].long().contiguous()
    if train_features_cpu.shape[0] != train_labels.shape[0]:
        raise RuntimeError("train feature/label length mismatch")
    if bool((train_features_cpu.norm(dim=1) - 1.0).abs().max() > 5e-4):
        raise RuntimeError("train features are not normalized")
    if bool((test_features_cpu.norm(dim=1) - 1.0).abs().max() > 5e-4):
        raise RuntimeError("test features are not normalized")

    dataset = str(protocol["dataset"])
    locked_counts = KNOWN_TEST_COUNTS[dataset]
    known_counts = None
    if locked_counts is not None:
        if list(locked_counts) != list(protocol["test_class_counts"]):
            raise RuntimeError("committed known marginal disagrees with protocol manifest")
        known_counts = torch.as_tensor(locked_counts, dtype=torch.float32, device=device)

    return TargetEvaluationData(
        dataset=dataset,
        backbone=dict(protocol["backbone"]),
        num_base=int(protocol["num_base"]),
        num_total=int(protocol["num_total"]),
        increment=int(protocol["increment"]),
        train_features=train_features_cpu.to(device),
        train_labels=train_labels.to(device),
        test_features=test_features_cpu.to(device),
        test_labels=cached_test_labels.to(device),
        standard_support_abs=protocol["standard_support_abs"].long().to(device),
        nested_support_abs=protocol["nested_support_abs"].long().to(device),
        support_seeds=[int(value) for value in protocol["support_seeds"]],
        support_sha256=str(protocol["support_sha256"]),
        protocol_sha256=str(protocol["protocol_sha256"]),
        feature_file_sha256=str(expected["file_sha256"]),
        selection_source_sha256=selection_source,
        known_test_counts=known_counts,
    )


def features_by_class(data: TargetEvaluationData) -> Dict[int, torch.Tensor]:
    return {
        class_id: data.train_features[data.train_labels == class_id]
        for class_id in range(data.num_total)
    }
