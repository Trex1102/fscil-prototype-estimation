"""Audited FGVC-Aircraft manifest for the released FSCIL big-start protocol.

The CVPRW 2024 FSCIL-Calibration paper uses a 50-class base task followed by
ten 5-way tasks on FGVC-Aircraft. Its released loader uses torchvision's
``trainval``/``test`` split, a 100-class label space, and a user-seeded random
class order. We lock seed 1993, the seed in the supplied experiment configs,
without implying that the repository shipped an Aircraft-specific index list.
This module avoids importing torchvision, whose version differs across hosts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .utils import file_sha256


AIRCRAFT_CLASS_ORDER_SEED = 1993
AIRCRAFT_NUM_CLASSES = 100
AIRCRAFT_PROTOCOL_SOURCE = (
    "Goswami et al., Calibrating Higher-Order Statistics for FSCIL with "
    "Pre-trained Vision Transformers, CVPRW 2024"
)
AIRCRAFT_PROTOCOL_REPOSITORY = "https://github.com/dipamgoswami/FSCIL-Calibration"
AIRCRAFT_PROTOCOL_REVISION = "fc4c232b61de9a522e996eb253909babcbe733d1"


@dataclass(frozen=True)
class AircraftManifest:
    data_dir: Path
    train_paths: Tuple[Path, ...]
    train_image_ids: Tuple[str, ...]
    train_labels: np.ndarray
    test_paths: Tuple[Path, ...]
    test_image_ids: Tuple[str, ...]
    test_labels: np.ndarray
    variants: Tuple[str, ...]
    raw_class_order: Tuple[int, ...]
    annotation_sha256: Dict[str, str]
    train_manifest_sha256: str
    test_manifest_sha256: str


def resolve_aircraft_data_dir(dataset_root: Path) -> Path:
    """Resolve the archive's ``data`` directory without following a broad search."""
    root = dataset_root.resolve()
    candidates = (
        root / "fgvc-aircraft-2013b" / "data",
        root / "data",
        root,
    )
    required = (
        "variants.txt",
        "images_variant_trainval.txt",
        "images_variant_test.txt",
    )
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).is_file() for name in required):
            if not (candidate / "images").is_dir():
                raise RuntimeError("FGVC-Aircraft annotations exist but images/ is absent")
            return candidate.resolve()
    raise FileNotFoundError(
        "could not find extracted FGVC-Aircraft 2013b data under {}".format(root)
    )


def locked_class_order(num_classes: int = AIRCRAFT_NUM_CLASSES) -> Tuple[int, ...]:
    rng = np.random.RandomState(AIRCRAFT_CLASS_ORDER_SEED)
    return tuple(int(value) for value in rng.permutation(num_classes).tolist())


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _manifest_hash(image_ids: Sequence[str], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for image_id, label in zip(image_ids, labels.tolist()):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(label).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _read_split(
    path: Path,
    images_dir: Path,
    variant_to_raw: Dict[str, int],
    raw_to_incremental: Dict[int, int],
) -> Tuple[Tuple[Path, ...], Tuple[str, ...], np.ndarray]:
    image_paths: List[Path] = []
    image_ids: List[str] = []
    labels: List[int] = []
    for line_number, line in enumerate(_read_nonempty_lines(path), start=1):
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            raise RuntimeError("malformed annotation {}:{}".format(path, line_number))
        image_id, variant = pieces
        if variant not in variant_to_raw:
            raise RuntimeError("unknown variant {!r} in {}".format(variant, path))
        image_path = (images_dir / (image_id + ".jpg")).resolve()
        if not image_path.is_file():
            raise FileNotFoundError("missing FGVC-Aircraft image {}".format(image_path))
        raw_label = variant_to_raw[variant]
        image_paths.append(image_path)
        image_ids.append(image_id)
        labels.append(raw_to_incremental[raw_label])
    return tuple(image_paths), tuple(image_ids), np.asarray(labels, dtype=np.int64)


def load_aircraft_manifest(dataset_root: Path) -> AircraftManifest:
    data_dir = resolve_aircraft_data_dir(dataset_root)
    variants_path = data_dir / "variants.txt"
    train_path = data_dir / "images_variant_trainval.txt"
    test_path = data_dir / "images_variant_test.txt"
    variants = tuple(_read_nonempty_lines(variants_path))
    if len(variants) != AIRCRAFT_NUM_CLASSES or len(set(variants)) != len(variants):
        raise RuntimeError(
            "expected exactly {} unique FGVC-Aircraft variants, found {}".format(
                AIRCRAFT_NUM_CLASSES, len(variants)
            )
        )
    order = locked_class_order(len(variants))
    raw_to_incremental = {raw: new for new, raw in enumerate(order)}
    variant_to_raw = {variant: index for index, variant in enumerate(variants)}
    images_dir = data_dir / "images"
    train_paths, train_ids, train_labels = _read_split(
        train_path, images_dir, variant_to_raw, raw_to_incremental
    )
    test_paths, test_ids, test_labels = _read_split(
        test_path, images_dir, variant_to_raw, raw_to_incremental
    )
    for split_name, labels, minimum in (
        ("trainval", train_labels, 20),
        ("test", test_labels, 1),
    ):
        counts = np.bincount(labels, minlength=AIRCRAFT_NUM_CLASSES)
        if len(counts) != AIRCRAFT_NUM_CLASSES or int(counts.min()) < minimum:
            raise RuntimeError(
                "FGVC-Aircraft {} split has invalid per-class counts {}".format(
                    split_name, counts.tolist()
                )
            )
    return AircraftManifest(
        data_dir=data_dir,
        train_paths=train_paths,
        train_image_ids=train_ids,
        train_labels=train_labels,
        test_paths=test_paths,
        test_image_ids=test_ids,
        test_labels=test_labels,
        variants=variants,
        raw_class_order=order,
        annotation_sha256={
            "variants": file_sha256(variants_path),
            "trainval": file_sha256(train_path),
            "test": file_sha256(test_path),
        },
        train_manifest_sha256=_manifest_hash(train_ids, train_labels),
        test_manifest_sha256=_manifest_hash(test_ids, test_labels),
    )


def aircraft_manifest_metadata(manifest: AircraftManifest) -> Dict[str, object]:
    return {
        "dataset_release": "FGVC-Aircraft 2013b",
        "split": {"train": "trainval", "test": "test"},
        "num_classes": AIRCRAFT_NUM_CLASSES,
        "class_order_seed": AIRCRAFT_CLASS_ORDER_SEED,
        "raw_class_order": list(manifest.raw_class_order),
        "protocol_source": AIRCRAFT_PROTOCOL_SOURCE,
        "protocol_repository": AIRCRAFT_PROTOCOL_REPOSITORY,
        "protocol_repository_revision": AIRCRAFT_PROTOCOL_REVISION,
        "train_manifest_sha256": manifest.train_manifest_sha256,
        "test_manifest_sha256": manifest.test_manifest_sha256,
        "annotation_sha256": dict(manifest.annotation_sha256),
        "train_class_counts": np.bincount(
            manifest.train_labels, minlength=AIRCRAFT_NUM_CLASSES
        ).astype(int).tolist(),
        "test_class_counts": np.bincount(
            manifest.test_labels, minlength=AIRCRAFT_NUM_CLASSES
        ).astype(int).tolist(),
    }
