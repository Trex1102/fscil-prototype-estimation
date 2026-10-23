#!/usr/bin/env python3
"""Run the DTD confirmation with base audits completed before test extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from reproduce import CODE


def build_plan(args):
    run = args.run_root.resolve()
    features = run / "artifacts/features"
    protocol = run / "artifacts/protocol/dtd_clip_vitb16_protocol.pt"
    base = run / "artifacts/protocol/dtd_clip_vitb16_protocol_base_selection.pt"
    parameters = run / "parameters/clip_vitb16/dtd"
    results = run / "results/clip_vitb16/dtd"
    train = features / "dtd_train.pt"
    test = features / "dtd_test.pt"
    full = features / "clip_feats_cls_dtd_vitb16.pt"
    radial = parameters / "shrinksel_selection.json"
    radial_audit = parameters / "shrinksel_audit.json"
    qpr = parameters / "qpr_selection.json"
    qpr_audit = parameters / "qpr_audit.json"
    plan = []

    def add(name, module, *options):
        plan.append({"name": name, "command": [sys.executable, "-m", "fscil." + module] + [str(x) for x in options]})

    device = ["--device", args.device]
    extraction = ["--dataset", "dtd", "--backbone-key", "clip_vitb16", "--code-root", CODE,
                  "--dataset-root", args.data_root.resolve(), "--download-root", run / "weights",
                  "--workers", args.workers] + device
    add("train_features", "extract_clip_features", *extraction, "--split", "train", "--output", train)
    add("base_bundle", "build_pretarget_selection_bundle", "--dataset", "dtd", "--train-cache", train, "--output", base)
    for operation in ("select", "audit"):
        extra = ["--parameters", radial] if operation == "audit" else []
        add("shrinksel_" + operation, "radial_selection", operation, "--bundle", base,
            "--episodes", 200, "--output", radial if operation == "select" else radial_audit, *extra, *device)
    for operation in ("select", "audit"):
        extra = ["--parameters", qpr] if operation == "audit" else []
        add("qpr_" + operation, "qpr_calibration", operation, "--bundle", base,
            "--shrink-parameters", radial, "--episodes", 20, "--regime", "one_increment",
            "--output", qpr if operation == "select" else qpr_audit, *extra, *device)
    # This command is reached only after both audits return successfully.
    add("test_features", "extract_clip_features", *extraction, "--split", "test", "--output", test)
    add("merge_features", "merge_staged_feature_cache", "--train-cache", train, "--test-cache", test, "--output", full)
    add("target_protocol", "build_protocol_bundle", "--dataset", "dtd", "--code-root", CODE,
        "--feature-cache", full, "--output", protocol,
        "--selection-output", run / "artifacts/protocol/dtd_final_base_selection.pt")
    target = ["--feature-cache", full, "--protocol", protocol, "--seed-limit", args.seeds] + device
    add("shrinksel_target", "run_radial_inductive", *target,
        "--radial-parameters", radial, "--radial-audit", radial_audit,
        "--submitted-shrink-parameters", radial, "--output", results / "shrinksel.json")
    add("qpr_target", "run_qpr", *target, "--shrink-parameters", radial,
        "--qpr-parameters", qpr, "--qpr-audit", qpr_audit, "--sessions", "all",
        "--modes", "full", "--methods", "shrinksel,qpr_selected", "--output", results / "qpr.json")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted DTD root accepted by fscil.dtd_data")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    for item in plan:
        print(item["name"] + ": " + " ".join(shlex.quote(x) for x in item["command"]), flush=True)
    if args.dry_run:
        return
    run = args.run_root.resolve()
    for name in ("artifacts/features", "artifacts/protocol", "parameters/clip_vitb16/dtd", "results/clip_vitb16/dtd", "logs/clip_vitb16/dtd"):
        (run / name).mkdir(parents=True, exist_ok=True)
    status = {"dataset": "dtd", "test_extraction_after_base_audits": True, "completed": []}
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(CODE)
    for item in plan:
        log = run / "logs/clip_vitb16/dtd" / (item["name"] + ".log")
        print("Running " + item["name"] + "; log: " + str(log), flush=True)
        with log.open("w") as handle:
            result = subprocess.run(item["command"], cwd=str(CODE), env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        status["completed"].append({"name": item["name"], "returncode": result.returncode})
        (run / "results/clip_vitb16/dtd/reproduction.json").write_text(json.dumps(status, indent=2) + "\n")
        if result.returncode:
            print(log.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)
        if item["name"] == "shrinksel_audit":
            audit_path = run / "parameters/clip_vitb16/dtd/shrinksel_audit.json"
            if not json.loads(audit_path.read_text())["passes_non_regression_gate"]:
                status["state"] = "stopped_at_base_audit"
                (run / "results/clip_vitb16/dtd/reproduction.json").write_text(json.dumps(status, indent=2) + "\n")
                print("DTD base audit failed; test images remain unencoded", file=sys.stderr)
                raise SystemExit(3)


if __name__ == "__main__":
    main()
