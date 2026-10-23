#!/usr/bin/env python3
"""Run frozen-feature diagnosis, ShrinkSel, and corrected CLIP QPR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


CODE = Path(__file__).resolve().parent
DATASETS = ("cub200", "cifar100", "miniimagenet", "fgvc_aircraft")
STAGES = ("prepare", "shrinksel", "qpr", "diagnostics", "baselines", "comparators")
BACKBONES = ("clip_vitb16", "clip_vitb32", "clip_rn50", "mae_ft_vitb16", "supervised_resnet50")


def build_plan(args):
    run = args.run_root.resolve()
    data = args.dataset
    backbone = getattr(args, "backbone", "clip_vitb16")
    if backbone == "clip_vitb16":
        cache_name = "clip_feats_cls_" + data + "_vitb16.pt"
    else:
        cache_name = ("clip_feats_cls_" if backbone.startswith("clip_") else "frozen_feats_cls_") + data + "_" + backbone + ".pt"
    cache = (args.feature_cache or run / "artifacts/features" / cache_name).resolve()
    protocol = (args.protocol or run / "artifacts/protocol" / (data + "_" + backbone + "_protocol.pt")).resolve()
    base = run / "artifacts/protocol" / (data + "_" + backbone + "_protocol_base_selection.pt")
    parameters = run / "parameters" / backbone / data
    results = run / "results" / backbone / data
    scalar = parameters / "scalar_selection.json"
    scalar_audit = parameters / "scalar_audit.json"
    radial = parameters / "shrinksel_selection.json"
    radial_audit = parameters / "shrinksel_audit.json"
    qpr = parameters / "qpr_selection.json"
    qpr_audit = parameters / "qpr_audit.json"
    plan = []

    def add(stage, label, module, *options):
        if stage in args.stages:
            plan.append({"stage": stage, "name": label, "command": [sys.executable, "-m", "fscil." + module] + [str(x) for x in options]})

    device = ["--device", args.device]
    target = ["--feature-cache", cache, "--protocol", protocol, "--seed-limit", args.seeds] + device
    if args.feature_cache is None:
        extractor = "extract_clip_features" if backbone.startswith("clip_") else "extract_frozen_features"
        add("prepare", "features", extractor, "--dataset", data,
            "--backbone-key", backbone, "--code-root", CODE,
            "--dataset-root", args.data_root.resolve(), "--download-root", run / "weights",
            "--output", cache, "--workers", args.workers, *device)
    if args.protocol is None:
        add("prepare", "protocol", "build_protocol_bundle", "--dataset", data,
            "--backbone-key", backbone, "--code-root", CODE,
            "--dataset-root", args.data_root.resolve(), "--feature-cache", cache,
            "--output", protocol, "--selection-output", base)
    for module, name, output, audit_output in (
        ("selection", "scalar", scalar, scalar_audit),
        ("radial_selection", "shrinksel", radial, radial_audit),
    ):
        add("shrinksel", name + "_selection", module, "select", "--bundle", base,
            "--episodes", args.selection_episodes, "--output", output, *device)
        add("shrinksel", name + "_audit", module, "audit", "--bundle", base,
            "--episodes", args.audit_episodes, "--parameters", output,
            "--output", audit_output, *device)
    add("shrinksel", "inductive", "run_radial_inductive", *target,
        "--radial-parameters", radial, "--radial-audit", radial_audit,
        "--submitted-shrink-parameters", scalar, "--output", results / "shrinksel.json")
    add("qpr", "qpr_selection", "qpr_calibration", "select", "--bundle", base,
        "--shrink-parameters", radial, "--episodes", args.qpr_episodes,
        "--regime", "one_increment", "--output", qpr, *device)
    add("qpr", "qpr_audit", "qpr_calibration", "audit", "--bundle", base,
        "--shrink-parameters", radial, "--parameters", qpr,
        "--episodes", args.qpr_episodes, "--regime", "one_increment",
        "--output", qpr_audit, *device)
    add("qpr", "qpr", "run_qpr_self_exclusion", *target,
        "--shrink-parameters", radial, "--qpr-parameters", qpr,
        "--qpr-audit", qpr_audit, "--sessions", "all", "--output", results / "qpr.json")
    add("diagnostics", "diagnostics", "diagnostics", *target,
        "--bootstrap", args.bootstrap, "--output", results / "diagnostics.json")
    baseline = parameters / "baseline_selection.json"
    baseline_audit = parameters / "baseline_audit.json"
    for operation in ("select", "audit"):
        extra = ["--parameters", baseline] if operation == "audit" else []
        add("baselines", "baseline_" + operation, "baseline_selection", operation,
            "--bundle", base, "--episodes", args.selection_episodes if operation == "select" else args.audit_episodes,
            "--output", baseline if operation == "select" else baseline_audit, *extra, *device)
    add("baselines", "baselines", "run_inductive", *target,
        "--shrink-parameters", scalar, "--shrink-audit", scalar_audit,
        "--baseline-parameters", baseline, "--baseline-audit", baseline_audit,
        "--output", results / "baselines.json")
    transductive = parameters / "transductive_selection.json"
    transductive_audit = parameters / "transductive_audit.json"
    for operation in ("select", "audit"):
        extra = ["--parameters", transductive] if operation == "audit" else []
        add("comparators", "transductive_" + operation, "transductive_selection", operation,
            "--bundle", base, "--episodes", args.qpr_episodes,
            "--output", transductive if operation == "select" else transductive_audit, *extra, *device)
    add("comparators", "comparators", "run_transductive", *target,
        "--shrink-parameters", radial, "--radial-audit", radial_audit,
        "--transductive-parameters", transductive, "--transductive-audit", transductive_audit,
        "--methods", "cosine,shrinksel,tim_gd,laplacian_shot,laplacian_shot_core,protolp_paper,ptmap_uniform",
        "--sessions", "all", "--modes", "full", "--output", results / "comparators.json")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, default="clip_vitb16")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Exact dataset root expected by the published loader; see README")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, help="Reuse an existing matching CLIP ViT-B/16 cache")
    parser.add_argument("--protocol", type=Path, help="Reuse an existing protocol; selection stages also require its base bundle in RUN/artifacts/protocol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--selection-episodes", type=int, default=200)
    parser.add_argument("--audit-episodes", type=int, default=200)
    parser.add_argument("--qpr-episodes", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--stages", help="Comma-separated stages: " + ",".join(STAGES) + "; QPR defaults to CLIP representations only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    default_stages = [stage for stage in STAGES
        if (stage != "qpr" or (args.backbone.startswith("clip_") and args.dataset != "fgvc_aircraft"))
        and (stage != "comparators" or (args.backbone == "clip_vitb16" and args.dataset != "fgvc_aircraft"))]
    selected_stages = args.stages or ",".join(default_stages)
    args.stages = [item.strip() for item in selected_stages.split(",") if item.strip()]
    if not args.stages or set(args.stages) - set(STAGES):
        parser.error("Unknown or empty stage list")
    if "qpr" in args.stages and not args.backbone.startswith("clip_"):
        parser.error("The corrected QPR benchmark in this paper covers CLIP representations")
    plan = build_plan(args)
    for item in plan:
        print(item["name"] + ": " + " ".join(shlex.quote(x) for x in item["command"]), flush=True)
    if args.dry_run:
        return
    run = args.run_root.resolve()
    for name in ("logs/" + args.backbone + "/" + args.dataset,
                 "parameters/" + args.backbone + "/" + args.dataset,
                 "results/" + args.backbone + "/" + args.dataset,
                 "artifacts/protocol", "artifacts/features"):
        (run / name).mkdir(parents=True, exist_ok=True)
    status = {"dataset": args.dataset, "backbone": args.backbone, "seeds": args.seeds, "commands": plan, "completed": []}
    status_path = run / "results" / args.backbone / args.dataset / "reproduction.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(CODE)
    for item in plan:
        log_path = run / "logs" / args.backbone / args.dataset / (item["name"] + ".log")
        print("Running " + item["name"] + "; log: " + str(log_path), flush=True)
        with log_path.open("w") as handle:
            result = subprocess.run(item["command"], cwd=str(CODE), env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        status["completed"].append({"name": item["name"], "returncode": result.returncode})
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if result.returncode:
            print(log_path.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
