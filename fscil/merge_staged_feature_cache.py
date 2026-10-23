#!/usr/bin/env python3
"""Merge independently encoded train/test tensors after the pretarget gate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from .utils import atomic_torch_save, file_sha256, tensor_sha256


_FUNCTION_ADDRESS_RE = re.compile(r"(?<= at )0x[0-9a-fA-F]+(?=>)")


def canonical_preprocess_repr(value):
    """Remove runtime-only function addresses from transform representations."""
    if not isinstance(value, str):
        return value
    return _FUNCTION_ADDRESS_RE.sub("runtime-address", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--test-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train = torch.load(str(args.train_cache.resolve()), map_location="cpu")
    test = torch.load(str(args.test_cache.resolve()), map_location="cpu")
    if set(train) != {"train", "tr", "metadata"}:
        raise RuntimeError("invalid train-only staged cache fields")
    if set(test) != {"test", "tl", "metadata"}:
        raise RuntimeError("invalid test-only staged cache fields")
    train_meta = train["metadata"]
    test_meta = test["metadata"]
    if train_meta.get("extracted_split_scope") != "train":
        raise RuntimeError("first input is not train-only")
    if test_meta.get("extracted_split_scope") != "test":
        raise RuntimeError("second input is not test-only")
    for key in (
        "dataset",
        "backbone",
        "weight_file_sha256",
        "standalone_dataset_manifest",
    ):
        if train_meta.get(key) != test_meta.get(key):
            raise RuntimeError("staged cache metadata mismatch for {}".format(key))
    train_preprocess = canonical_preprocess_repr(train_meta.get("preprocess"))
    test_preprocess = canonical_preprocess_repr(test_meta.get("preprocess"))
    if train_preprocess != test_preprocess:
        raise RuntimeError("staged cache metadata mismatch for preprocess")
    if tensor_sha256(train["train"]) != train_meta.get("train_feature_sha256"):
        raise RuntimeError("train feature hash mismatch")
    if tensor_sha256(test["test"]) != test_meta.get("test_feature_sha256"):
        raise RuntimeError("test feature hash mismatch")
    metadata = dict(train_meta)
    metadata["preprocess"] = train_preprocess
    metadata.update(
        {
            "test_label_sha256": test_meta["test_label_sha256"],
            "test_feature_sha256": test_meta["test_feature_sha256"],
            "extracted_split_scope": "staged_train_then_test",
            "train_stage_file_sha256": file_sha256(args.train_cache.resolve()),
            "test_stage_file_sha256": file_sha256(args.test_cache.resolve()),
            "pretarget_source_feature_identity": train_meta["train_feature_sha256"],
        }
    )
    payload = {
        "train": train["train"].contiguous(),
        "tr": train["tr"].contiguous(),
        "test": test["test"].contiguous(),
        "tl": test["tl"].contiguous(),
        "metadata": metadata,
    }
    atomic_torch_save(args.output.resolve(), payload)
    print("saved={} file={}".format(args.output.resolve(), file_sha256(args.output.resolve())))


if __name__ == "__main__":
    main()
