#!/usr/bin/env python3
"""Build a small audited protocol bundle around an existing CLIP feature cache.

The generated file contains labels, exact standard five-shot support indices, nested
shot-diagnostic indices, fixed base construction/selection/audit classes, dataset
counts, and hashes.  It does not duplicate the large feature tensors.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from .constants import BACKBONES, DATASETS, PROTOCOL_SEED, SHOTS, SUPPORT_SEEDS
from .utils import atomic_torch_save, file_sha256, git_revision, stable_json_sha256, tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backbone-key", default="clip_vitb16", choices=sorted(BACKBONES)
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        help="base-training-only bundle; defaults beside --output",
    )
    return parser.parse_args()


def data_key(item: Any) -> Any:
    if isinstance(item, str):
        return ("path", os.path.normpath(item))
    array = np.asarray(item)
    return ("array", array.shape, str(array.dtype), hashlib.sha256(array.tobytes()).digest())


def import_fscil(code_root: Path):
    root = code_root.resolve()
    sys.path.insert(0, str(root))
    # The historical repository's datasets/ directory has no __init__.py and can be
    # shadowed by Hugging Face's installed ``datasets`` package.  Load it under a
    # private package name so its relative imports remain correct without editing the
    # user's external repository.
    package_name = "_shrinksel_fscil_datasets"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "datasets")]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = package_name + ".data_manager"
    module_spec = importlib.util.spec_from_file_location(
        module_name, str(root / "datasets" / "data_manager.py")
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("unable to load FSCIL DatasetManager")
    data_manager = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = data_manager
    module_spec.loader.exec_module(data_manager)
    DatasetManager = data_manager.DatasetManager
    from main import setup_cfg  # pylint: disable=import-outside-toplevel
    from utils.util import set_seed  # pylint: disable=import-outside-toplevel

    return DatasetManager, setup_cfg, set_seed


def build_key_map(data: Iterable[Any], targets: np.ndarray) -> Dict[int, Dict[Any, List[int]]]:
    result: Dict[int, Dict[Any, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for index, (item, target) in enumerate(zip(data, targets)):
        result[int(target)][data_key(item)].append(index)
    return result


def recover_standard_supports(dm, key_map, train_targets: np.ndarray) -> torch.Tensor:
    """Recover absolute cache rows from DatasetManager's seeded five-shot samples."""
    num_base = int(dm.cfg.DATASET.NUM_INIT_CLS)
    num_total = int(dm.cfg.DATASET.NUM_CLASSES)
    supports = torch.full((num_total - num_base, 5), -1, dtype=torch.long)
    cursor: Dict[Tuple[int, Any], int] = collections.defaultdict(int)
    for task in range(1, dm.num_tasks):
        sampled = dm.get_dataset(task, "train", mode="test")
        for item, target in zip(sampled.images, np.asarray(sampled.labels)):
            cls = int(target)
            key = data_key(item)
            candidates = key_map[cls][key]
            offset = cursor[(cls, key)]
            if offset >= len(candidates):
                raise RuntimeError("ambiguous/over-consumed dataset key for class {}".format(cls))
            absolute_index = candidates[offset]
            cursor[(cls, key)] += 1
            row = cls - num_base
            empty = int((supports[row] >= 0).sum().item())
            if empty >= 5:
                raise RuntimeError("more than five supports recovered for class {}".format(cls))
            supports[row, empty] = absolute_index
    if bool((supports < 0).any()):
        raise RuntimeError("incomplete support recovery")
    if not bool(torch.as_tensor(train_targets[supports.numpy()] >= num_base).all()):
        raise RuntimeError("support labels do not match novel-class range")
    expected = torch.arange(num_base, num_total)[:, None].expand_as(supports)
    actual = torch.as_tensor(train_targets[supports.numpy()])
    if not torch.equal(actual, expected):
        raise RuntimeError("support-class ordering mismatch")
    return supports


def sample_standard_supports(
    train_targets: np.ndarray, num_base: int, num_total: int, seed: int
) -> torch.Tensor:
    """Create a reproducible five-shot draw when no canonical index list exists."""
    rows: List[List[int]] = []
    for class_id in range(num_base, num_total):
        candidates = np.where(train_targets == class_id)[0].copy()
        if len(candidates) < 5:
            raise RuntimeError("class {} has fewer than five supports".format(class_id))
        class_seed = (
            PROTOCOL_SEED * 1000033 + int(seed) * 10007 + class_id * 1000003
        ) % (2 ** 32 - 1)
        rng = np.random.RandomState(class_seed)
        rng.shuffle(candidates)
        rows.append([int(value) for value in candidates[:5].tolist()])
    result = torch.as_tensor(rows, dtype=torch.long)
    expected = torch.arange(num_base, num_total)[:, None].expand_as(result)
    actual = torch.as_tensor(train_targets[result.numpy()])
    if not torch.equal(actual, expected):
        raise RuntimeError("sampled support-class ordering mismatch")
    return result


def nested_diagnostic_supports(
    standard: torch.Tensor, train_targets: np.ndarray, num_base: int, num_total: int, seed: int
) -> torch.Tensor:
    max_shot = max(SHOTS)
    rows = []
    for class_id in range(num_base, num_total):
        fixed = standard[class_id - num_base].tolist()
        all_rows = np.where(train_targets == class_id)[0].tolist()
        remaining = [index for index in all_rows if index not in set(fixed)]
        class_seed = (PROTOCOL_SEED * 1000003 + seed * 1009 + class_id * 9176) % (2 ** 32 - 1)
        rng = np.random.RandomState(class_seed)
        rng.shuffle(remaining)
        sequence = fixed + remaining
        if len(sequence) < max_shot:
            raise RuntimeError("class {} has fewer than {} training images".format(class_id, max_shot))
        rows.append(sequence[:max_shot])
    return torch.as_tensor(rows, dtype=torch.long)


def class_partition(num_base: int) -> Dict[str, List[int]]:
    rng = np.random.RandomState(PROTOCOL_SEED + num_base)
    classes = np.arange(num_base)
    rng.shuffle(classes)
    n_construction = int(round(0.60 * num_base))
    n_selection = int(round(0.20 * num_base))
    return {
        "construction": sorted(int(x) for x in classes[:n_construction]),
        "selection": sorted(int(x) for x in classes[n_construction:n_construction + n_selection]),
        "audit": sorted(int(x) for x in classes[n_construction + n_selection:]),
    }


def manifest_hash(data: Iterable[Any], targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    for item, target in zip(data, targets):
        digest.update(repr(data_key(item)).encode("utf-8"))
        digest.update(int(target).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    backbone = BACKBONES[args.backbone_key]
    code_root = args.code_root.resolve()
    feature_cache = args.feature_cache.resolve()
    cached = torch.load(str(feature_cache), map_location="cpu")
    train_features = cached["train"].float().contiguous()
    test_features = cached["test"].float().contiguous()
    test_labels = cached["tl"].long().contiguous()
    standalone = spec.get("dataset_loader") in (
        "fgvc_aircraft_2013b",
        "dtd_r1.0.1_split1",
    )
    dataset_manifest_payload: Dict[str, object]
    if standalone:
        if "tr" not in cached:
            raise RuntimeError("standalone feature cache does not contain train labels")
        train_label_tensor = cached["tr"].long().contiguous()
        train_targets = train_label_tensor.numpy()
        test_targets_dataset = test_labels.numpy()
        metadata = cached.get("metadata", {})
        standalone_manifest = metadata.get("standalone_dataset_manifest")
        if not isinstance(standalone_manifest, dict):
            raise RuntimeError("standalone feature cache lacks its dataset manifest")
        if metadata.get("train_label_sha256") != tensor_sha256(train_label_tensor):
            raise RuntimeError("cached standalone train-label hash mismatch")
        if metadata.get("test_label_sha256") != tensor_sha256(test_labels):
            raise RuntimeError("cached standalone test-label hash mismatch")
        if standalone_manifest.get("train_class_counts") != np.bincount(
            train_targets, minlength=spec["num_total"]
        ).astype(int).tolist():
            raise RuntimeError("standalone train counts disagree with cached manifest")
        if standalone_manifest.get("test_class_counts") != np.bincount(
            test_targets_dataset, minlength=spec["num_total"]
        ).astype(int).tolist():
            raise RuntimeError("standalone test counts disagree with cached manifest")
        dataset_manifest_payload = {
            "train_sha256": str(standalone_manifest["train_manifest_sha256"]),
            "test_sha256": str(standalone_manifest["test_manifest_sha256"]),
            "source": standalone_manifest,
        }
        dm0 = None
        cfg = None
        DatasetManager = None
        set_seed = None
    else:
        DatasetManager, setup_cfg, set_seed = import_fscil(code_root)
        old_cwd = Path.cwd()
        os.chdir(str(code_root))
        try:
            cfg = setup_cfg(
                str(code_root / spec["data_cfg"]), str(code_root / spec["train_cfg"])
            )
            cfg.defrost()
            if args.dataset_root is not None:
                cfg.DATASET.ROOT = str(args.dataset_root.resolve())
            cfg.SEED = 1
            cfg.freeze()
            set_seed(1)
            dm0 = DatasetManager(cfg)
        finally:
            os.chdir(str(old_cwd))
        train_targets = np.asarray(dm0.train_targets, dtype=np.int64)
        test_targets_dataset = np.asarray(dm0.test_targets, dtype=np.int64)
        dataset_manifest_payload = {
            "train_sha256": manifest_hash(dm0.train_data, train_targets),
            "test_sha256": manifest_hash(dm0.test_data, test_targets_dataset),
        }
    if train_features.shape[0] != len(train_targets):
        raise RuntimeError("feature/train-label length mismatch")
    if not np.array_equal(test_targets_dataset, test_labels.numpy()):
        raise RuntimeError("cached test labels disagree with DatasetManager")
    if train_features.shape[1] != backbone["feature_dim"]:
        raise RuntimeError("unexpected frozen-backbone feature dimension")
    train_norm_error = float((train_features.norm(dim=1) - 1.0).abs().max().item())
    test_norm_error = float((test_features.norm(dim=1) - 1.0).abs().max().item())
    if max(train_norm_error, test_norm_error) > 5e-4:
        raise RuntimeError("feature cache is not row-normalized")

    if train_targets.min() != 0 or train_targets.max() != spec["num_total"] - 1:
        raise RuntimeError("training labels do not cover the locked class range")
    if test_targets_dataset.min() != 0 or test_targets_dataset.max() != spec["num_total"] - 1:
        raise RuntimeError("test labels do not cover the locked class range")

    key_map = None if standalone else build_key_map(dm0.train_data, train_targets)
    standard_by_seed = []
    nested_by_seed = []
    for seed in SUPPORT_SEEDS:
        if standalone:
            standard = sample_standard_supports(
                train_targets, spec["num_base"], spec["num_total"], int(seed)
            )
        else:
            cfg_seed = copy.deepcopy(cfg)
            cfg_seed.defrost()
            cfg_seed.SEED = int(seed)
            cfg_seed.freeze()
            set_seed(int(seed))
            old_cwd = Path.cwd()
            os.chdir(str(code_root))
            try:
                dm_seed = DatasetManager(cfg_seed)
                standard = recover_standard_supports(dm_seed, key_map, train_targets)
            finally:
                os.chdir(str(old_cwd))
        standard_by_seed.append(standard)
        nested_by_seed.append(
            nested_diagnostic_supports(
                standard, train_targets, spec["num_base"], spec["num_total"], int(seed)
            )
        )
    standard_tensor = torch.stack(standard_by_seed, 0)
    nested_tensor = torch.stack(nested_by_seed, 0)

    train_counts = np.bincount(train_targets, minlength=spec["num_total"]).astype(int).tolist()
    test_counts = np.bincount(test_labels.numpy(), minlength=spec["num_total"]).astype(int).tolist()
    partition = class_partition(spec["num_base"])
    partition_flat = partition["construction"] + partition["selection"] + partition["audit"]
    if sorted(partition_flat) != list(range(spec["num_base"])) or len(set(partition_flat)) != spec["num_base"]:
        raise RuntimeError("invalid class-disjoint base partition")

    protocol_payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "display": spec["display"],
        "backbone": backbone,
        "num_base": spec["num_base"],
        "num_total": spec["num_total"],
        "increment": spec["increment"],
        "shots": 5,
        "support_protocol": (
            "20 deterministic matched draws; K=5 is nested inside K={1,2,5,10,20}"
            if standalone
            else "20 canonical DatasetManager seed draws; K=5 nested in shot diagnostics"
        ),
        "support_seeds": list(SUPPORT_SEEDS),
        "train_labels": torch.as_tensor(train_targets, dtype=torch.long),
        "test_labels": test_labels,
        "standard_support_abs": standard_tensor,
        "nested_support_abs": nested_tensor,
        "nested_shot_values": list(SHOTS),
        "base_partition": partition,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "feature_cache": {
            "basename": feature_cache.name,
            "file_sha256": file_sha256(feature_cache),
            "train_tensor_sha256": tensor_sha256(train_features),
            "test_tensor_sha256": tensor_sha256(test_features),
            "test_label_sha256": tensor_sha256(test_labels),
            "train_shape": list(train_features.shape),
            "test_shape": list(test_features.shape),
            "dtype": str(train_features.dtype),
            "max_row_norm_error": max(train_norm_error, test_norm_error),
            "selection_source_sha256": str(
                cached.get("metadata", {}).get(
                    "pretarget_source_feature_identity",
                    file_sha256(feature_cache),
                )
            ),
        },
        "dataset_manifest": dataset_manifest_payload,
        "support_sha256": tensor_sha256(standard_tensor),
        "nested_support_sha256": tensor_sha256(nested_tensor),
        "code_root_git": git_revision(code_root),
        "protocol_seed": PROTOCOL_SEED,
    }
    hash_view = dict(protocol_payload)
    hash_view.pop("train_labels")
    hash_view.pop("test_labels")
    hash_view.pop("standard_support_abs")
    hash_view.pop("nested_support_abs")
    protocol_payload["protocol_sha256"] = stable_json_sha256(hash_view)
    atomic_torch_save(args.output.resolve(), protocol_payload)

    selection_output = args.selection_output
    if selection_output is None:
        selection_output = args.output.with_name(args.output.stem + "_base_selection.pt")
    base_mask = torch.as_tensor(train_targets < spec["num_base"])
    base_features = train_features[base_mask].contiguous()
    base_labels = torch.as_tensor(train_targets, dtype=torch.long)[base_mask].contiguous()
    selection_payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "display": spec["display"],
        "backbone": backbone,
        "num_base": spec["num_base"],
        "base_features": base_features,
        "base_labels": base_labels,
        "base_partition": partition,
        "train_class_counts_base": train_counts[:spec["num_base"]],
        "protocol_seed": PROTOCOL_SEED,
        "source_feature_file_sha256": protocol_payload["feature_cache"]["file_sha256"],
        "base_feature_sha256": tensor_sha256(base_features),
        "base_label_sha256": tensor_sha256(base_labels),
        "dataset_train_manifest_sha256": protocol_payload["dataset_manifest"]["train_sha256"],
    }
    selection_hash_view = dict(selection_payload)
    selection_hash_view.pop("base_features")
    selection_hash_view.pop("base_labels")
    selection_payload["selection_bundle_sha256"] = stable_json_sha256(selection_hash_view)
    atomic_torch_save(selection_output.resolve(), selection_payload)
    print("saved {}".format(args.output.resolve()))
    print("saved {}".format(selection_output.resolve()))
    print("dataset={} feature={} support={} protocol={}".format(
        args.dataset,
        protocol_payload["feature_cache"]["file_sha256"],
        protocol_payload["support_sha256"],
        protocol_payload["protocol_sha256"],
    ))
    print("base_partition_sizes={}".format({key: len(value) for key, value in partition.items()}))
    print("test_counts={}".format(test_counts))


if __name__ == "__main__":
    main()
