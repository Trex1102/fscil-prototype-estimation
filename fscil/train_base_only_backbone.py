#!/usr/bin/env python3
"""Train a ResNet-18 on base-session images only, freeze it, and cache features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .constants import BACKBONES, DATASETS
from .utils import (
    atomic_torch_save,
    file_sha256,
    resolve_dataset_items,
    seed_everything,
    tensor_sha256,
)


DEFAULT_EPOCHS = {"cifar100": 200, "miniimagenet": 120, "cub200": 120}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DEFAULT_EPOCHS)
    )
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def import_dataset(code_root: Path):
    root = code_root.resolve()
    sys.path.insert(0, str(root))
    package = types.ModuleType("datasets")
    package.__path__ = [str(root / "datasets")]
    package.__package__ = "datasets"
    sys.modules["datasets"] = package
    from datasets.data_manager import DatasetManager, TaskDataset  # type: ignore
    from main import setup_cfg  # type: ignore

    return DatasetManager, TaskDataset, setup_cfg


def transformations(dataset: str):
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        evaluate = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return train, evaluate
    size = 84 if dataset == "miniimagenet" else 224
    resize = 92 if dataset == "miniimagenet" else 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train, evaluate


def build_model(dataset: str, num_base: int) -> nn.Module:
    model = models.resnet18(weights=None)
    if dataset == "cifar100":
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, int(num_base))
    return model


def forward_features(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    value = model.conv1(image)
    value = model.bn1(value)
    value = model.relu(value)
    value = model.maxpool(value)
    value = model.layer1(value)
    value = model.layer2(value)
    value = model.layer3(value)
    value = model.layer4(value)
    value = model.avgpool(value)
    return torch.flatten(value, 1)


def load_data(args: argparse.Namespace):
    spec = DATASETS[args.dataset]
    DatasetManager, TaskDataset, setup_cfg = import_dataset(args.code_root)
    old_cwd = Path.cwd()
    os.chdir(str(args.code_root.resolve()))
    try:
        cfg = setup_cfg(
            str(args.code_root / spec["data_cfg"]),
            str(args.code_root / spec["train_cfg"]),
        )
        cfg.defrost()
        if args.dataset_root is not None:
            cfg.DATASET.ROOT = str(args.dataset_root.resolve())
        cfg.SEED = 1
        cfg.freeze()
        manager = DatasetManager(cfg)
    finally:
        os.chdir(str(old_cwd))
    return manager, TaskDataset


def make_loaders(args: argparse.Namespace, manager, TaskDataset):
    train_transform, eval_transform = transformations(args.dataset)
    train_targets = np.asarray(manager.train_targets, dtype=np.int64)
    num_base = int(DATASETS[args.dataset]["num_base"])
    base_mask = train_targets < num_base
    train_items = resolve_dataset_items(manager.train_data, args.code_root)
    test_items = resolve_dataset_items(manager.test_data, args.code_root)
    base_dataset = TaskDataset(
        train_items[base_mask],
        train_targets[base_mask],
        train_transform,
    )
    train_eval_dataset = TaskDataset(
        train_items, train_targets, eval_transform
    )
    test_targets = np.asarray(manager.test_targets, dtype=np.int64)
    test_dataset = TaskDataset(
        test_items, test_targets, eval_transform
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    base_loader = DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return base_loader, train_eval_loader, test_loader, train_targets, test_targets


def train_model(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    epochs: int,
) -> Dict[str, object]:
    optimizer = SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch in loader:
            image = batch["image"].to(args.device, non_blocking=True)
            label = batch["label"].long().to(args.device, non_blocking=True)
            logits = model(image)
            loss = F.cross_entropy(logits, label, label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * image.shape[0]
            correct += int((logits.argmax(1) == label).sum().item())
            count += int(image.shape[0])
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / max(count, 1),
            "accuracy": 100.0 * correct / max(count, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                "dataset={} epoch={}/{} loss={:.4f} accuracy={:.2f}".format(
                    args.dataset,
                    epoch + 1,
                    epochs,
                    row["loss"],
                    row["accuracy"],
                ),
                flush=True,
            )
    return {"history": history, "runtime_sec": time.time() - started}


@torch.no_grad()
def extract(
    model: nn.Module,
    loader: DataLoader,
    expected_labels: np.ndarray,
    device: str,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    labels = []
    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        features.append(F.normalize(forward_features(model, image).float(), dim=-1).cpu())
        labels.append(batch["label"].long().cpu())
        if index % 50 == 0:
            print("{} batches={}".format(label, index + 1), flush=True)
    feature_tensor = torch.cat(features, 0).contiguous()
    label_tensor = torch.cat(labels, 0).contiguous()
    if not np.array_equal(label_tensor.numpy(), expected_labels):
        raise RuntimeError("{} extraction labels/order mismatch".format(label))
    return feature_tensor, label_tensor


def main() -> None:
    args = parse_args()
    seed_everything(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manager, TaskDataset = load_data(args)
    (
        base_loader,
        train_eval_loader,
        test_loader,
        train_targets,
        test_targets,
    ) = make_loaders(args, manager, TaskDataset)
    num_base = int(DATASETS[args.dataset]["num_base"])
    epochs = int(args.epochs) if args.epochs > 0 else DEFAULT_EPOCHS[args.dataset]
    model = build_model(args.dataset, num_base).to(args.device)
    if args.checkpoint.exists():
        checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
        if checkpoint["dataset"] != args.dataset or checkpoint["num_base"] != num_base:
            raise RuntimeError("base-only checkpoint metadata mismatch")
        model.load_state_dict(checkpoint["model_state"])
        training = checkpoint["training"]
        print("loaded {}".format(args.checkpoint), flush=True)
    else:
        training = train_model(model, base_loader, args, epochs)
        checkpoint = {
            "schema_version": 1,
            "dataset": args.dataset,
            "backbone": BACKBONES["base_only_resnet18"],
            "num_base": num_base,
            "optimized_class_ids": list(range(num_base)),
            "excluded_novel_class_ids": list(
                range(num_base, int(DATASETS[args.dataset]["num_total"]))
            ),
            "seed": 1,
            "epochs": epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training": training,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        atomic_torch_save(args.checkpoint.resolve(), checkpoint)
        print("saved {}".format(args.checkpoint.resolve()), flush=True)
    model.to(args.device)
    train_features, train_labels = extract(
        model,
        train_eval_loader,
        train_targets,
        args.device,
        args.dataset + " train",
    )
    test_features, test_labels = extract(
        model,
        test_loader,
        test_targets,
        args.device,
        args.dataset + " test",
    )
    payload = {
        "train": train_features,
        "test": test_features,
        "tl": test_labels,
        "metadata": {
            "schema_version": 1,
            "dataset": args.dataset,
            "backbone": BACKBONES["base_only_resnet18"],
            "optimized_class_ids": list(range(num_base)),
            "excluded_novel_class_ids": list(
                range(num_base, int(DATASETS[args.dataset]["num_total"]))
            ),
            "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
            "train_feature_sha256": tensor_sha256(train_features),
            "test_feature_sha256": tensor_sha256(test_features),
            "test_label_sha256": tensor_sha256(test_labels),
            "epochs": epochs,
            "seed": 1,
        },
    }
    atomic_torch_save(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_shape": list(train_features.shape),
                "test_shape": list(test_features.shape),
                "feature_file_sha256": file_sha256(args.output.resolve()),
                "checkpoint_sha256": payload["metadata"]["checkpoint_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
