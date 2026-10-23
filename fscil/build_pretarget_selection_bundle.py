#!/usr/bin/env python3
"""Build a base-only selector bundle from a staged train-only feature cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .build_protocol_bundle import class_partition
from .constants import BACKBONES, DATASETS, PROTOCOL_SEED
from .utils import atomic_torch_save, stable_json_sha256, tensor_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("dtd",))
    parser.add_argument("--backbone-key", default="clip_vitb16", choices=sorted(BACKBONES))
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    cached = torch.load(str(args.train_cache.resolve()), map_location="cpu")
    forbidden = sorted(key for key in cached if "test" in key.lower())
    if forbidden or set(cached).difference({"train", "tr", "metadata"}):
        raise RuntimeError("train-only cache exposes unexpected fields {}".format(forbidden))
    metadata = cached.get("metadata", {})
    if metadata.get("dataset") != args.dataset:
        raise RuntimeError("staged cache dataset mismatch")
    if metadata.get("extracted_split_scope") != "train":
        raise RuntimeError("staged cache was not produced by train-only extraction")
    train = cached["train"].float().contiguous()
    labels = cached["tr"].long().contiguous()
    if tensor_sha256(train) != metadata.get("train_feature_sha256"):
        raise RuntimeError("staged train feature hash mismatch")
    if tensor_sha256(labels) != metadata.get("train_label_sha256"):
        raise RuntimeError("staged train label hash mismatch")
    counts = np.bincount(labels.numpy(), minlength=spec["num_total"]).astype(int).tolist()
    manifest = metadata.get("standalone_dataset_manifest")
    if not isinstance(manifest, dict) or counts != manifest.get("train_class_counts"):
        raise RuntimeError("staged train cache/manifest count mismatch")
    base_mask = labels < int(spec["num_base"])
    base_features = train[base_mask].contiguous()
    base_labels = labels[base_mask].contiguous()
    source_identity = tensor_sha256(train)
    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "display": spec["display"],
        "backbone": BACKBONES[args.backbone_key],
        "num_base": int(spec["num_base"]),
        "base_features": base_features,
        "base_labels": base_labels,
        "base_partition": class_partition(int(spec["num_base"])),
        "train_class_counts_base": counts[: int(spec["num_base"])],
        "protocol_seed": PROTOCOL_SEED,
        # This is intentionally a train-tensor identity, not the partial-cache
        # file hash; it remains valid after the separately encoded target split
        # is merged into the final cache.
        "source_feature_file_sha256": source_identity,
        "source_identity_kind": "normalized_train_feature_tensor_sha256",
        "base_feature_sha256": tensor_sha256(base_features),
        "base_label_sha256": tensor_sha256(base_labels),
        "dataset_train_manifest_sha256": manifest["train_manifest_sha256"],
        "target_images_encoded": False,
    }
    hash_view = dict(payload)
    hash_view.pop("base_features")
    hash_view.pop("base_labels")
    payload["selection_bundle_sha256"] = stable_json_sha256(hash_view)
    atomic_torch_save(args.output.resolve(), payload)
    print(
        "saved={} base_shape={} source_train={}".format(
            args.output.resolve(), tuple(base_features.shape), source_identity
        )
    )


if __name__ == "__main__":
    main()
