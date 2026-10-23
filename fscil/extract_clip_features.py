#!/usr/bin/env python3
"""Extract deterministic image-only OpenAI CLIP feature caches."""

from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .aircraft_data import aircraft_manifest_metadata, load_aircraft_manifest
from .constants import BACKBONES, DATASETS
from .dtd_data import dtd_manifest_metadata, load_dtd_manifest
from .utils import (
    atomic_torch_save,
    file_sha256,
    resolve_dataset_items,
    seed_everything,
    tensor_sha256,
)


_FUNCTION_ADDRESS_RE = re.compile(r"(?<= at )0x[0-9a-fA-F]+(?=>)")


def stable_preprocess_repr(preprocess) -> str:
    """Return a process-independent description of a torchvision transform."""
    return _FUNCTION_ADDRESS_RE.sub("runtime-address", repr(preprocess))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--backbone-key",
        required=True,
        choices=sorted(key for key in BACKBONES if key.startswith("clip_")),
    )
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split",
        choices=("both", "train", "test"),
        default="both",
        help="staged standalone extraction; default preserves the full-cache path",
    )
    parser.add_argument("--download-root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="required for standalone datasets such as FGVC-Aircraft",
    )
    return parser.parse_args()


def import_openai_clip(code_root: Path):
    root = code_root.resolve()
    sys.path.insert(0, str(root))
    import models.clip.clip as clip  # type: ignore

    return clip


def import_dataset_components(code_root: Path):
    root = code_root.resolve()
    sys.path.insert(0, str(root))
    # Force this historical repository's dataset package ahead of an installed
    # Hugging Face package with the same top-level name.
    package = types.ModuleType("datasets")
    package.__path__ = [str(root / "datasets")]
    package.__package__ = "datasets"
    sys.modules["datasets"] = package
    from datasets.data_manager import DatasetManager, TaskDataset  # type: ignore
    from main import setup_cfg  # type: ignore

    return DatasetManager, TaskDataset, setup_cfg


class ImagePathDataset(torch.utils.data.Dataset):
    """Minimal image-path dataset using the exact CLIP evaluation transform."""

    def __init__(self, paths, labels: np.ndarray, transform) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        if len(self.paths) != len(self.labels):
            raise RuntimeError("image path/label length mismatch")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with self.paths[index].open("rb") as handle:
            image = Image.open(handle).convert("RGB")
            tensor = self.transform(image)
        return {"image": tensor, "label": int(self.labels[index])}


@torch.no_grad()
def encode(
    model,
    loader: DataLoader,
    device: str,
    expected_labels: np.ndarray,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        value = F.normalize(model.encode_image(image).float(), dim=-1)
        features.append(value.cpu())
        labels.append(batch["label"].long().cpu())
        if batch_index % 50 == 0:
            print("{} batches={}".format(label, batch_index + 1), flush=True)
    feature_tensor = torch.cat(features, 0).contiguous()
    label_tensor = torch.cat(labels, 0).contiguous()
    if not np.array_equal(label_tensor.numpy(), expected_labels):
        raise RuntimeError("{} loader order/labels differ from DatasetManager".format(label))
    if bool((feature_tensor.norm(dim=1) - 1.0).abs().max() > 5e-4):
        raise RuntimeError("{} CLIP features are not normalized".format(label))
    return feature_tensor, label_tensor


def main() -> None:
    args = parse_args()
    seed_everything(1)
    code_root = args.code_root.resolve()
    spec = DATASETS[args.dataset]
    backbone = BACKBONES[args.backbone_key]
    clip = import_openai_clip(code_root)

    model_name = str(backbone["openai_model_name"])
    download_root = args.download_root.resolve() if args.download_root else None
    if download_root is not None:
        download_root.mkdir(parents=True, exist_ok=True)
    # The vendored OpenAI CLIP copy predates the later ``download_root``
    # argument to ``clip.load``.  Resolve and checksum the checkpoint in the
    # experiment-owned directory explicitly, then give ``load`` that path.
    # This also keeps model files off the system disk on both remote hosts.
    model_source = model_name
    if download_root is not None:
        url = clip._MODELS[model_name]  # pylint: disable=protected-access
        model_source = clip._download(  # pylint: disable=protected-access
            url, root=str(download_root)
        )
    model, preprocess = clip.load(model_source, device=args.device, jit=False)
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    standalone_metadata = None
    if spec.get("dataset_loader") in ("fgvc_aircraft_2013b", "dtd_r1.0.1_split1"):
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for standalone datasets")
        if spec.get("dataset_loader") == "fgvc_aircraft_2013b":
            manifest = load_aircraft_manifest(args.dataset_root)
            standalone_metadata = aircraft_manifest_metadata(manifest)
        else:
            manifest = load_dtd_manifest(args.dataset_root)
            standalone_metadata = dtd_manifest_metadata(manifest)
        train_targets = manifest.train_labels
        test_targets = manifest.test_labels
        train_dataset = ImagePathDataset(
            manifest.train_paths, train_targets, preprocess
        )
        test_dataset = ImagePathDataset(
            manifest.test_paths, test_targets, preprocess
        )
    else:
        DatasetManager, TaskDataset, setup_cfg = import_dataset_components(code_root)
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
            data_manager = DatasetManager(cfg)
        finally:
            os.chdir(str(old_cwd))
        train_targets = np.asarray(data_manager.train_targets, dtype=np.int64)
        test_targets = np.asarray(data_manager.test_targets, dtype=np.int64)
        train_dataset = TaskDataset(
            resolve_dataset_items(data_manager.train_data, code_root),
            train_targets,
            preprocess,
        )
        test_dataset = TaskDataset(
            resolve_dataset_items(data_manager.test_data, code_root),
            test_targets,
            preprocess,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    encoded = {}
    if args.split in ("both", "train"):
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            generator=generator,
        )
        encoded["train"], encoded["tr"] = encode(
            model,
            train_loader,
            args.device,
            train_targets,
            args.dataset + " train",
        )
    if args.split in ("both", "test"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            generator=generator,
        )
        encoded["test"], encoded["tl"] = encode(
            model,
            test_loader,
            args.device,
            test_targets,
            args.dataset + " test",
        )
    for feature_key in ("train", "test"):
        if feature_key in encoded and encoded[feature_key].shape[1] != int(
            backbone["feature_dim"]
        ):
            raise RuntimeError(
                "extracted feature dimension does not match backbone metadata"
            )

    weight_path = None
    weight_hash = None
    try:
        url = clip._MODELS[model_name]  # pylint: disable=protected-access
        candidate_root = download_root or (Path.home() / ".cache" / "clip")
        candidate = candidate_root / Path(url).name
        if candidate.exists():
            weight_path = str(candidate)
            weight_hash = file_sha256(candidate)
    except (AttributeError, KeyError):
        pass
    metadata = {
            "schema_version": 1,
            "dataset": args.dataset,
            "backbone": backbone,
            "image_only_downstream": True,
            "text_encoder_called": False,
            "preprocess": stable_preprocess_repr(preprocess),
            "weight_path": weight_path,
            "weight_file_sha256": weight_hash,
            "standalone_dataset_manifest": standalone_metadata,
    }
    if "train" in encoded:
        metadata["train_label_sha256"] = tensor_sha256(encoded["tr"])
        metadata["train_feature_sha256"] = tensor_sha256(encoded["train"])
    if "test" in encoded:
        metadata["test_label_sha256"] = tensor_sha256(encoded["tl"])
        metadata["test_feature_sha256"] = tensor_sha256(encoded["test"])
    metadata["extracted_split_scope"] = args.split
    payload = dict(encoded)
    payload["metadata"] = metadata
    atomic_torch_save(args.output.resolve(), payload)
    print(
        "saved={} split={} shapes={} feature={} file={}".format(
            args.output.resolve(),
            args.split,
            {
                key: tuple(encoded[key].shape)
                for key in ("train", "test")
                if key in encoded
            },
            {
                key: metadata[key]
                for key in ("train_feature_sha256", "test_feature_sha256")
                if key in metadata
            },
            file_sha256(args.output.resolve()),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
