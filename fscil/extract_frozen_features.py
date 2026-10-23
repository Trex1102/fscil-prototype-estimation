#!/usr/bin/env python3
"""Extract normalized features from non-CLIP released frozen image encoders.

The output schema deliberately matches :mod:`fscil.extract_clip_features` so the
same protocol builder, selectors, diagnostics, and target evaluators operate on every
representation family.  No class names, prompts, or target labels enter the model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .aircraft_data import aircraft_manifest_metadata, load_aircraft_manifest
from .constants import BACKBONES, DATASETS
from .dtd_data import dtd_manifest_metadata, load_dtd_manifest
from .extract_clip_features import import_dataset_components
from .utils import (
    atomic_torch_save,
    file_sha256,
    resolve_dataset_items,
    seed_everything,
    tensor_sha256,
)


SUPPORTED = ("mae_ft_vitb16", "supervised_resnet50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--backbone-key", required=True, choices=SUPPORTED)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


class ImagePathDataset(torch.utils.data.Dataset):
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


def state_dict_sha256(model: nn.Module) -> str:
    """Hash the exact loaded tensors, independent of cache filename/layout."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_encoder(
    backbone_key: str, download_root: Path
) -> Tuple[nn.Module, object, Dict[str, object]]:
    backbone = BACKBONES[backbone_key]
    torch_root = download_root.resolve() / "torch"
    torch_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_root))

    if backbone_key == "mae_ft_vitb16":
        # Disable the optional Hugging Face mirror so the recorded artifact is the
        # official MAE checkpoint URL in timm's released model configuration.
        import timm  # pylint: disable=import-outside-toplevel

        model = timm.create_model(
            str(backbone["timm_model_name"]),
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        source_url = str(backbone["checkpoint_url"])
        checkpoint_dir = torch_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weight_path = checkpoint_dir / Path(source_url).name
        if not weight_path.exists():
            torch.hub.download_url_to_file(source_url, str(weight_path))
        checkpoint = torch.load(str(weight_path), map_location="cpu")
        state = dict(checkpoint["model"])
        state.pop("head.weight", None)
        state.pop("head.bias", None)
        incompatibility = model.load_state_dict(state, strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(
                "official MAE fine-tuned checkpoint mismatch: missing={} unexpected={}".format(
                    incompatibility.missing_keys, incompatibility.unexpected_keys
                )
            )
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        implementation = {
            "library": "timm",
            "library_version": str(timm.__version__),
            "model_name": str(backbone["timm_model_name"]),
            "pretrained_source_url": source_url,
            "checkpoint_top_level_fields": sorted(checkpoint),
            "removed_classifier_fields": ["head.weight", "head.bias"],
            "data_config": dict(data_config),
            "feature_interface": (
                "num_classes=0 forward output; global average of patch tokens "
                "followed by the released fine-tuning fc_norm"
            ),
        }
    elif backbone_key == "supervised_resnet50":
        from torchvision import models  # pylint: disable=import-outside-toplevel
        import torchvision  # pylint: disable=import-outside-toplevel

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        transform = weights.transforms()
        source_url = str(weights.url)
        weight_path = torch_root / "checkpoints" / Path(source_url).name
        implementation = {
            "library": "torchvision",
            "library_version": str(torchvision.__version__),
            "model_name": "resnet50",
            "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
            "pretrained_source_url": source_url,
            "feature_interface": "penultimate global-average-pooled representation",
        }
    else:  # pragma: no cover - argparse constrains this path
        raise ValueError("unsupported backbone: {}".format(backbone_key))

    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    implementation["model_state_sha256"] = state_dict_sha256(model)
    implementation["weight_path"] = str(weight_path) if weight_path.exists() else None
    implementation["weight_file_sha256"] = (
        file_sha256(weight_path) if weight_path.exists() else None
    )
    return model, transform, implementation


@torch.no_grad()
def encode(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    expected_labels: np.ndarray,
    expected_dim: int,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        value = model(image)
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise RuntimeError("encoder did not return a feature matrix")
        if int(value.shape[1]) != int(expected_dim):
            raise RuntimeError("encoder feature dimension mismatch")
        value = F.normalize(value.float(), dim=-1)
        features.append(value.cpu())
        labels.append(batch["label"].long().cpu())
        if batch_index % 50 == 0:
            print("{} batches={}".format(label, batch_index + 1), flush=True)
    feature_tensor = torch.cat(features, 0).contiguous()
    label_tensor = torch.cat(labels, 0).contiguous()
    if not np.array_equal(label_tensor.numpy(), expected_labels):
        raise RuntimeError("{} loader order/labels differ from DatasetManager".format(label))
    if bool((feature_tensor.norm(dim=1) - 1.0).abs().max() > 5e-4):
        raise RuntimeError("{} features are not normalized".format(label))
    return feature_tensor, label_tensor


def main() -> None:
    args = parse_args()
    seed_everything(1)
    code_root = args.code_root.resolve()
    backbone = BACKBONES[args.backbone_key]
    spec = DATASETS[args.dataset]
    model, transform, implementation = build_encoder(
        args.backbone_key, args.download_root
    )
    model.to(args.device)

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
        train_dataset = ImagePathDataset(manifest.train_paths, train_targets, transform)
        test_dataset = ImagePathDataset(manifest.test_paths, test_targets, transform)
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
            transform,
        )
        test_dataset = TaskDataset(
            resolve_dataset_items(data_manager.test_data, code_root),
            test_targets,
            transform,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )
    train_features, train_labels = encode(
        model,
        train_loader,
        args.device,
        train_targets,
        int(backbone["feature_dim"]),
        args.dataset + " train",
    )
    test_features, test_labels = encode(
        model,
        test_loader,
        args.device,
        test_targets,
        int(backbone["feature_dim"]),
        args.dataset + " test",
    )
    metadata = {
        "schema_version": 1,
        "dataset": args.dataset,
        "backbone": backbone,
        "image_only_downstream": True,
        "text_encoder_called": False,
        "preprocess": repr(transform),
        "weight_path": implementation["weight_path"],
        "weight_file_sha256": implementation["weight_file_sha256"],
        "model_state_sha256": implementation["model_state_sha256"],
        "implementation": implementation,
        "standalone_dataset_manifest": standalone_metadata,
        "train_label_sha256": tensor_sha256(train_labels),
        "train_feature_sha256": tensor_sha256(train_features),
        "test_label_sha256": tensor_sha256(test_labels),
        "test_feature_sha256": tensor_sha256(test_features),
        "extracted_split_scope": "both",
    }
    payload = {
        "train": train_features,
        "tr": train_labels,
        "test": test_features,
        "tl": test_labels,
        "metadata": metadata,
    }
    atomic_torch_save(args.output.resolve(), payload)
    print(
        "saved={} shapes={} state={} file={}".format(
            args.output.resolve(),
            {"train": tuple(train_features.shape), "test": tuple(test_features.shape)},
            implementation["model_state_sha256"],
            file_sha256(args.output.resolve()),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
