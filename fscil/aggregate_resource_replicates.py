#!/usr/bin/env python3
"""Aggregate validated mode-resolved resource analyses across process repeats.

Accuracy, work, identity, support count, fallback count, and method metadata must
match exactly. Runtime uses the median independent process repeat; peak allocation
uses the maximum so the reported memory number remains conservative.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import median
from typing import Dict, Mapping, Sequence

from .analyze_selector_stability import atomic_write_json, stable_json_sha256


def _load_valid(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = str(payload["result_sha256"])
    unhashed = copy.deepcopy(dict(payload))
    unhashed.pop("result_sha256")
    if stable_json_sha256(unhashed) != expected:
        raise RuntimeError("resource analysis hash mismatch: " + str(path))
    return payload


def _indexed_settings(
    payload: Mapping[str, object],
) -> Dict[str, Mapping[str, object]]:
    settings = {str(row["dataset"]): row for row in payload["settings"]}
    if len(settings) != len(payload["settings"]):
        raise RuntimeError("duplicate resource dataset")
    return settings


def _constant(rows: Sequence[Mapping[str, object]], key: str) -> object:
    values = [row[key] for row in rows]
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError("resource replicate mismatch for " + key)
    return copy.deepcopy(values[0])


def _aggregate_method(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    for key in ("accuracy", "calls", "fallback_calls", "last_metadata"):
        _constant(rows, key)
    runtime_keys = (
        "runtime_sec",
        "microseconds_per_unique_prediction",
        "microseconds_per_context_row",
    )
    peak_keys = ("peak_allocated_bytes", "peak_allocated_mib")
    result = {
        "accuracy": _constant(rows, "accuracy"),
        "calls": _constant(rows, "calls"),
        "fallback_calls": _constant(rows, "fallback_calls"),
        "last_metadata": _constant(rows, "last_metadata"),
    }
    for key in runtime_keys:
        values = [float(row[key]) for row in rows]
        result[key] = float(median(values))
        result[key + "_replicates"] = values
        result[key + "_range"] = [float(min(values)), float(max(values))]
    for key in peak_keys:
        values = [float(row[key]) for row in rows]
        result[key] = (
            int(max(values)) if key.endswith("_bytes") else float(max(values))
        )
        result[key + "_replicates"] = [
            int(value) if key.endswith("_bytes") else float(value)
            for value in values
        ]
    return result


def _aggregate_mode(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    for key in (
        "dataset",
        "backbone",
        "identity",
        "support_draws",
        "mode",
        "classes",
        "feature_dimension",
        "work",
        "persistent_state",
    ):
        _constant(rows, key)
    method_names = [set(row["methods"]) for row in rows]
    if any(names != method_names[0] for names in method_names[1:]):
        raise RuntimeError("resource replicate method schema mismatch")
    result = {
        key: _constant(rows, key)
        for key in (
            "dataset",
            "backbone",
            "identity",
            "support_draws",
            "mode",
            "classes",
            "feature_dimension",
            "work",
            "persistent_state",
        )
    }
    result["paths"] = [str(row["path"]) for row in rows]
    result["result_sha256_replicates"] = [
        str(row["result_sha256"]) for row in rows
    ]
    result["methods"] = {
        method: _aggregate_method([row["methods"][method] for row in rows])
        for method in sorted(method_names[0])
    }
    return result


def aggregate(
    payloads: Sequence[Mapping[str, object]],
    paths: Sequence[Path],
) -> Dict[str, object]:
    if len(payloads) < 3 or len(payloads) != len(paths):
        raise ValueError("resource aggregation requires at least three matched repeats")
    if any(
        payload.get("stage")
        != "validated_mode_resolved_transductive_resource_analysis"
        for payload in payloads
    ):
        raise RuntimeError("invalid resource analysis stage")
    indexed = [_indexed_settings(payload) for payload in payloads]
    datasets = set(indexed[0])
    if any(set(rows) != datasets for rows in indexed[1:]):
        raise RuntimeError("resource replicate dataset matrix mismatch")
    settings = []
    for dataset in sorted(datasets):
        source_rows = [rows[dataset] for rows in indexed]
        for key in ("identity",):
            _constant(source_rows, key)
        qpr_modes = [set(row["qpr_modes"]) for row in source_rows]
        comparator_modes = [set(row["comparator_modes"]) for row in source_rows]
        if any(value != qpr_modes[0] for value in qpr_modes[1:]):
            raise RuntimeError(dataset + " QPR resource modes differ across repeats")
        if any(value != comparator_modes[0] for value in comparator_modes[1:]):
            raise RuntimeError(
                dataset + " comparator resource modes differ across repeats"
            )
        settings.append(
            {
                "dataset": dataset,
                "identity": _constant(source_rows, "identity"),
                "qpr_modes": {
                    mode: _aggregate_mode(
                        [row["qpr_modes"][mode] for row in source_rows]
                    )
                    for mode in sorted(qpr_modes[0])
                },
                "comparator_modes": {
                    mode: _aggregate_mode(
                        [row["comparator_modes"][mode] for row in source_rows]
                    )
                    for mode in sorted(comparator_modes[0])
                },
            }
        )
    qpr_method_rows = [
        method
        for setting in settings
        for mode in setting["qpr_modes"].values()
        for method in mode["methods"].values()
    ]
    scope = copy.deepcopy(dict(payloads[0]["measurement_scope"]))
    scope.update(
        {
            "independent_process_repeats": len(payloads),
            "runtime_aggregation": "median across complete mode-isolated process repeats",
            "runtime_repeat_range_retained": True,
            "peak_memory_aggregation": "maximum across process repeats",
        }
    )
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "validated_mode_resolved_transductive_resource_analysis",
        "measurement_scope": scope,
        "replicate_analysis_paths": [str(path.resolve()) for path in paths],
        "replicate_analysis_sha256": [
            str(payload["result_sha256"]) for payload in payloads
        ],
        "settings": settings,
        "aggregate": {
            "dataset_count": len(settings),
            "qpr_mode_rows": sum(len(row["qpr_modes"]) for row in settings),
            "comparator_mode_rows": sum(
                len(row["comparator_modes"]) for row in settings
            ),
            "minimum_qpr_microseconds_per_prediction": min(
                float(row["microseconds_per_unique_prediction"])
                for setting in settings
                for mode in setting["qpr_modes"].values()
                for method, row in mode["methods"].items()
                if method == "qpr_selected"
            ),
            "maximum_peak_allocated_mib": max(
                float(row["peak_allocated_mib"]) for row in qpr_method_rows
            ),
            "independent_process_repeats": len(payloads),
        },
    }
    result["result_sha256"] = stable_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True, type=Path)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.analysis]
    if len(paths) != int(args.expected_repeats):
        raise RuntimeError(
            "expected {} resource repeats, found {}".format(
                args.expected_repeats, len(paths)
            )
        )
    payloads = [_load_valid(path) for path in paths]
    result = aggregate(payloads, paths)
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
