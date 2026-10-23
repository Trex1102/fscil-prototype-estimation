#!/usr/bin/env python3
"""Safely download, validate, and extract the FGVC-Aircraft 2013b archive.

Torchvision's helper does not pin a checksum and treats an existing extraction
directory as complete.  A transiently truncated download can therefore leave a
plausible-looking but incomplete dataset.  This preparation step records the
official byte length, resumes only the exact archive path, validates every tar
member before extraction, and quarantines an incomplete extraction instead of
deleting it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable

from .aircraft_data import aircraft_manifest_metadata, load_aircraft_manifest
from .utils import atomic_write_json, file_sha256


AIRCRAFT_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
    "fgvc-aircraft-2013b.tar.gz"
)
# Content-Length and ETag observed from the official VGG endpoint.  The byte
# length is also consistent with the archive published in January 2014.
AIRCRAFT_ARCHIVE_BYTES = 2_753_340_328
AIRCRAFT_ARCHIVE_ETAG = "a41ca3a8-4f02e3e95a900"
ARCHIVE_NAME = "fgvc-aircraft-2013b.tar.gz"
EXTRACTED_NAME = "fgvc-aircraft-2013b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--attempts", type=int, default=8)
    return parser.parse_args()


def _write(status_path: Path, status: Dict[str, object], **updates: object) -> None:
    status.update(updates)
    atomic_write_json(status_path, status)


def _archive_members(archive: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(str(archive), mode="r:gz") as handle:
        for member in handle:
            yield member


def validate_archive(archive: Path) -> Dict[str, int]:
    """Fully decompress the archive and reject unsafe or unexpected members."""
    member_count = 0
    regular_files = 0
    image_files = 0
    for member in _archive_members(archive):
        member_count += 1
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError("unsafe path in Aircraft archive: {}".format(member.name))
        if not name.parts or name.parts[0] != EXTRACTED_NAME:
            raise RuntimeError(
                "Aircraft archive member is outside expected root: {}".format(
                    member.name
                )
            )
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(
                "Aircraft archive contains a disallowed link/device: {}".format(
                    member.name
                )
            )
        if member.isfile():
            regular_files += 1
            if (
                len(name.parts) == 4
                and name.parts[1:3] == ("data", "images")
                and name.suffix.lower() == ".jpg"
            ):
                image_files += 1
    if member_count < 10_000 or regular_files < 10_000 or image_files != 10_000:
        raise RuntimeError(
            "unexpected Aircraft archive inventory: members={}, files={}, images={}".format(
                member_count, regular_files, image_files
            )
        )
    return {
        "member_count": member_count,
        "regular_file_count": regular_files,
        "image_file_count": image_files,
    }


def download_with_resume(
    curl: str,
    archive: Path,
    attempts: int,
    status_path: Path,
    status: Dict[str, object],
) -> None:
    if attempts < 1:
        raise ValueError("--attempts must be positive")
    for attempt in range(1, attempts + 1):
        current = archive.stat().st_size if archive.exists() else 0
        _write(
            status_path,
            status,
            state="downloading",
            download_attempt=attempt,
            archive_bytes_before_attempt=current,
        )
        command = [
            curl,
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--output",
            str(archive),
            AIRCRAFT_URL,
        ]
        completed = subprocess.run(command, check=False)
        observed = archive.stat().st_size if archive.exists() else 0
        _write(
            status_path,
            status,
            curl_returncode=int(completed.returncode),
            archive_bytes_after_attempt=observed,
        )
        if completed.returncode == 0 and observed == AIRCRAFT_ARCHIVE_BYTES:
            return
        if observed > AIRCRAFT_ARCHIVE_BYTES:
            raise RuntimeError(
                "Aircraft archive exceeds pinned byte length: {} > {}".format(
                    observed, AIRCRAFT_ARCHIVE_BYTES
                )
            )
        if attempt < attempts:
            time.sleep(min(30, attempt * 5))
    raise RuntimeError(
        "Aircraft download did not reach pinned byte length after {} attempts".format(
            attempts
        )
    )


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME
    extracted = root / EXTRACTED_NAME
    status_path = root / "PREPARE_STATUS.json"
    status: Dict[str, object] = {
        "schema_version": 1,
        "stage": "prepare_fgvc_aircraft_2013b",
        "state": "starting",
        "started_unix": time.time(),
        "official_url": AIRCRAFT_URL,
        "pinned_archive_bytes": AIRCRAFT_ARCHIVE_BYTES,
        "official_etag": AIRCRAFT_ARCHIVE_ETAG,
        "archive": str(archive),
        "extracted_root": str(extracted),
    }
    atomic_write_json(status_path, status)

    try:
        if archive.is_symlink() or extracted.is_symlink():
            raise RuntimeError("Aircraft archive/extraction target must not be a symlink")

        # A complete extraction is accepted only after the same manifest audit used
        # by the experiment stage.  This makes the preparation idempotent.
        if extracted.is_dir():
            try:
                manifest = load_aircraft_manifest(root)
            except (FileNotFoundError, RuntimeError):
                manifest = None
            if manifest is not None:
                metadata = aircraft_manifest_metadata(manifest)
                if len(manifest.train_paths) != 6_667 or len(manifest.test_paths) != 3_333:
                    raise RuntimeError("complete-looking Aircraft manifest has wrong size")
                _write(
                    status_path,
                    status,
                    state="complete_existing_extraction",
                    manifest=metadata,
                    finished_unix=time.time(),
                )
                print(json.dumps(status, indent=2, sort_keys=True))
                return

        observed = archive.stat().st_size if archive.exists() else 0
        if observed > AIRCRAFT_ARCHIVE_BYTES:
            raise RuntimeError(
                "existing archive exceeds pinned byte length: {}".format(observed)
            )
        if observed != AIRCRAFT_ARCHIVE_BYTES:
            download_with_resume(
                args.curl, archive, int(args.attempts), status_path, status
            )
        observed = archive.stat().st_size
        if observed != AIRCRAFT_ARCHIVE_BYTES:
            raise RuntimeError(
                "archive byte-length mismatch: observed {}, expected {}".format(
                    observed, AIRCRAFT_ARCHIVE_BYTES
                )
            )

        _write(status_path, status, state="validating_archive")
        inventory = validate_archive(archive)
        archive_sha256 = file_sha256(archive)
        _write(
            status_path,
            status,
            archive_inventory=inventory,
            archive_sha256=archive_sha256,
        )

        quarantined = None
        if extracted.exists():
            if not extracted.is_dir():
                raise RuntimeError("Aircraft extraction target exists but is not a directory")
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            quarantined = root / (EXTRACTED_NAME + ".incomplete-" + stamp)
            if quarantined.exists():
                raise RuntimeError("quarantine target already exists: {}".format(quarantined))
            os.replace(str(extracted), str(quarantined))
            _write(
                status_path,
                status,
                state="quarantined_incomplete_extraction",
                quarantined_extraction=str(quarantined),
            )

        _write(status_path, status, state="extracting_validated_archive")
        with tarfile.open(str(archive), mode="r:gz") as handle:
            handle.extractall(str(root))

        manifest = load_aircraft_manifest(root)
        if len(manifest.train_paths) != 6_667 or len(manifest.test_paths) != 3_333:
            raise RuntimeError(
                "Aircraft split-size mismatch after extraction: trainval={}, test={}".format(
                    len(manifest.train_paths), len(manifest.test_paths)
                )
            )
        metadata = aircraft_manifest_metadata(manifest)
        _write(
            status_path,
            status,
            state="complete",
            manifest=metadata,
            finished_unix=time.time(),
            runtime_sec=time.time() - float(status["started_unix"]),
        )
        print(json.dumps(status, indent=2, sort_keys=True))
    except Exception as error:
        _write(
            status_path,
            status,
            state="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_unix=time.time(),
        )
        raise


if __name__ == "__main__":
    main()
