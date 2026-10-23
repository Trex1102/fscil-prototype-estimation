"""Audited DTD split-1 manifest for an untouched FSCIL confirmation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .utils import file_sha256


DTD_ARCHIVE_NAME = "dtd-r1.0.1.tar.gz"
DTD_ARCHIVE_BYTES = 625_239_812
DTD_ARCHIVE_SHA256 = "e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205"
DTD_CLASS_ORDER_SEED = 1993
DTD_NUM_CLASSES = 47


@dataclass(frozen=True)
class DTDManifest:
    data_dir: Path
    train_paths: Tuple[Path, ...]
    train_image_ids: Tuple[str, ...]
    train_labels: np.ndarray
    test_paths: Tuple[Path, ...]
    test_image_ids: Tuple[str, ...]
    test_labels: np.ndarray
    classes: Tuple[str, ...]
    raw_class_order: Tuple[int, ...]
    annotation_sha256: Dict[str, str]
    train_manifest_sha256: str
    test_manifest_sha256: str


def resolve_dtd_data_dir(dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    candidates = (root / "dtd", root)
    for candidate in candidates:
        labels = candidate / "labels"
        if (
            candidate.is_dir()
            and (candidate / "images").is_dir()
            and all((labels / name).is_file() for name in ("train1.txt", "val1.txt", "test1.txt"))
        ):
            return candidate.resolve()
    raise FileNotFoundError("could not find extracted DTD under {}".format(root))


def locked_class_order(num_classes: int = DTD_NUM_CLASSES) -> Tuple[int, ...]:
    rng = np.random.RandomState(DTD_CLASS_ORDER_SEED)
    return tuple(int(value) for value in rng.permutation(num_classes).tolist())


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _manifest_hash(image_ids: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for image_id, label in zip(image_ids, labels.tolist()):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(label).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def load_dtd_manifest(dataset_root: Path) -> DTDManifest:
    root = dataset_root.resolve()
    archive = root / DTD_ARCHIVE_NAME
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError("missing or linked DTD archive {}".format(archive))
    if int(archive.stat().st_size) != DTD_ARCHIVE_BYTES:
        raise RuntimeError("DTD archive byte-length mismatch")
    if file_sha256(archive) != DTD_ARCHIVE_SHA256:
        raise RuntimeError("DTD archive SHA-256 mismatch")
    data_dir = resolve_dtd_data_dir(root)
    labels_dir = data_dir / "labels"
    split_rows = {
        name: _read_lines(labels_dir / name)
        for name in ("train1.txt", "val1.txt", "test1.txt")
    }
    split_sets = {name: set(rows) for name, rows in split_rows.items()}
    if any(
        split_sets[left].intersection(split_sets[right])
        for left, right in (("train1.txt", "val1.txt"), ("train1.txt", "test1.txt"), ("val1.txt", "test1.txt"))
    ):
        raise RuntimeError("DTD split-1 annotations overlap")
    classes = tuple(sorted({row.split("/", 1)[0] for rows in split_rows.values() for row in rows}))
    if len(classes) != DTD_NUM_CLASSES:
        raise RuntimeError("expected 47 DTD classes, found {}".format(len(classes)))
    raw_by_name = {name: index for index, name in enumerate(classes)}
    order = locked_class_order(len(classes))
    raw_to_incremental = {raw: new for new, raw in enumerate(order)}

    def parse(rows: Sequence[str]) -> Tuple[Tuple[Path, ...], Tuple[str, ...], np.ndarray]:
        paths = []
        ids = []
        labels = []
        for row in rows:
            pieces = row.split("/", 1)
            if len(pieces) != 2 or pieces[0] not in raw_by_name:
                raise RuntimeError("invalid DTD annotation {!r}".format(row))
            path = (data_dir / "images" / row).resolve()
            if not path.is_file():
                raise FileNotFoundError("missing DTD image {}".format(path))
            paths.append(path)
            ids.append(row)
            labels.append(raw_to_incremental[raw_by_name[pieces[0]]])
        return tuple(paths), tuple(ids), np.asarray(labels, dtype=np.int64)

    train_rows = sorted(split_rows["train1.txt"] + split_rows["val1.txt"])
    test_rows = sorted(split_rows["test1.txt"])
    train_paths, train_ids, train_labels = parse(train_rows)
    test_paths, test_ids, test_labels = parse(test_rows)
    for split_name, labels, expected in (("train+val", train_labels, 80), ("test", test_labels, 40)):
        counts = np.bincount(labels, minlength=DTD_NUM_CLASSES)
        if counts.tolist() != [expected] * DTD_NUM_CLASSES:
            raise RuntimeError("DTD {} counts are not {} per class".format(split_name, expected))
    return DTDManifest(
        data_dir=data_dir,
        train_paths=train_paths,
        train_image_ids=train_ids,
        train_labels=train_labels,
        test_paths=test_paths,
        test_image_ids=test_ids,
        test_labels=test_labels,
        classes=classes,
        raw_class_order=order,
        annotation_sha256={
            name: file_sha256(labels_dir / name) for name in split_rows
        },
        train_manifest_sha256=_manifest_hash(train_ids, train_labels),
        test_manifest_sha256=_manifest_hash(test_ids, test_labels),
    )


def dtd_manifest_metadata(manifest: DTDManifest) -> Dict[str, object]:
    return {
        "dataset_release": "Describable Textures Dataset r1.0.1",
        "split": {"train": "train1+val1", "test": "test1"},
        "num_classes": DTD_NUM_CLASSES,
        "class_order_seed": DTD_CLASS_ORDER_SEED,
        "raw_class_order": list(manifest.raw_class_order),
        "classes_alphabetical": list(manifest.classes),
        "train_manifest_sha256": manifest.train_manifest_sha256,
        "test_manifest_sha256": manifest.test_manifest_sha256,
        "annotation_sha256": dict(manifest.annotation_sha256),
        "archive_name": DTD_ARCHIVE_NAME,
        "archive_bytes": DTD_ARCHIVE_BYTES,
        "archive_sha256": DTD_ARCHIVE_SHA256,
        "train_class_counts": [80] * DTD_NUM_CLASSES,
        "test_class_counts": [40] * DTD_NUM_CLASSES,
    }
