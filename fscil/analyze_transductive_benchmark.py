#!/usr/bin/env python3
"""Consolidate matched ShrinkSel/QPR transductive benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

DATASETS = ("cub200", "cifar100", "miniimagenet")
METHODS = (
    "shrinksel",
    "qpr",
    "tim_gd",
    "laplacian_shot",
    "laplacian_shot_core",
    "protolp_paper",
    "ptmap_uniform",
)
TABLE_METHODS = ("cosine",) + METHODS
METHOD_LABELS = {
    "cosine": "Cosine-NCM (inductive reference)",
    "shrinksel": "ShrinkSel",
    "qpr": "ShrinkSel + QPR",
    "tim_gd": "TIM-GD",
    "laplacian_shot": "LaplacianShot-full",
    "laplacian_shot_core": "LaplacianShot-core",
    "protolp_paper": "ProtoLP",
    "ptmap_uniform": "MAP-OT (PT-MAP core; uniform prior)",
}
METHOD_LATEX_LABELS = {
    **METHOD_LABELS,
    "laplacian_shot": "LapShot-full",
    "laplacian_shot_core": "LapShot-core",
    "ptmap_uniform": r"MAP-OT$_u$",
    "cosine": "Cosine-NCM",
}
EXPECTED_SESSIONS = {
    "cub200": tuple(range(11)),
    "cifar100": tuple(range(9)),
    "miniimagenet": tuple(range(9)),
}
EXPECTED_REVISED_HEAD = {
    "cub200": ("resultant_debiased", 0.75),
    "cifar100": ("resultant_debiased", 0.0),
    "miniimagenet": ("resultant_debiased", 0.0),
}


def stable_json_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def final_rows(payload: Mapping[str, object], method: str) -> List[Mapping[str, object]]:
    result = []
    for seed in payload["per_seed"]:
        sessions = seed["sessions"]
        if not sessions:
            raise RuntimeError("empty session record")
        result.append(sessions[-1]["modes"]["full"]["methods"][method])
    return result


def session_average(payload: Mapping[str, object], method: str) -> List[float]:
    result = []
    for seed in payload["per_seed"]:
        values = [
            float(session["modes"]["full"]["methods"][method]["overall"])
            for session in seed["sessions"]
        ]
        result.append(float(mean(values)))
    return result


def paired_interval(
    left: Sequence[float], right: Sequence[float], seed: int
) -> Tuple[float, List[float]]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if difference.size < 2:
        return float(difference.mean()), [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    index = rng.randint(0, difference.size, size=(10000, difference.size))
    samples = difference[index].mean(1)
    return float(difference.mean()), [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def summarize_dataset(
    payload: Mapping[str, object], dataset: str, seed: int
) -> Dict[str, object]:
    if payload.get("dataset") != dataset:
        raise RuntimeError("dataset/file mismatch")
    if payload.get("stage") != "target_transductive_evaluation_after_protocol_lock":
        raise RuntimeError("unexpected target transductive stage")
    expected_mode, expected_lambda = EXPECTED_REVISED_HEAD[dataset]
    if payload.get("shrinksel_prototype_mode") != expected_mode:
        raise RuntimeError("benchmark does not use revised ShrinkSel prototypes")
    if float(payload.get("shrinksel_lambda")) != expected_lambda:
        raise RuntimeError("benchmark does not use revised ShrinkSel geometry")
    if tuple(payload["methods"]) != METHODS:
        raise RuntimeError("unexpected method order")
    if payload["modes"] != ["full"] or payload["session_scope"] != "all":
        raise RuntimeError("benchmark must contain all sessions in full-cohort mode")
    information = payload.get("information_levels", {})
    expected_information = {
        "shrinksel": "I0/I1 strict inductive",
        "qpr": "I2 unlabeled query cohort",
        "tim_gd": "I2 unlabeled query cohort",
        "laplacian_shot": "I2 unlabeled query cohort",
        "laplacian_shot_core": "I2 unlabeled query cohort",
        "protolp_paper": "I2 unlabeled query cohort",
        "ptmap_uniform": "I2 plus assumed uniform class marginal",
    }
    for method, required_prefix in expected_information.items():
        observed = str(information.get(method, ""))
        if not observed.startswith(required_prefix):
            raise RuntimeError(
                "{} has an unexpected information budget: {!r}".format(
                    method, observed
                )
            )
    if len(payload["support_seeds"]) != 20:
        raise RuntimeError("benchmark must contain twenty support draws")
    support_seeds = [int(value) for value in payload["support_seeds"]]
    if len(set(support_seeds)) != 20:
        raise RuntimeError("benchmark support draws must be unique")
    per_seed = payload["per_seed"]
    if len(per_seed) != 20:
        raise RuntimeError("benchmark must contain one record per support draw")
    if [int(row["support_seed"]) for row in per_seed] != support_seeds:
        raise RuntimeError("per-seed records do not match the support-seed manifest")
    if support_seeds != list(range(1, 21)):
        raise RuntimeError("benchmark must use the locked support seeds 1--20")
    expected_sessions = EXPECTED_SESSIONS[dataset]
    for seed_row in per_seed:
        actual_sessions = tuple(
            int(session["session_index"]) for session in seed_row["sessions"]
        )
        if actual_sessions != expected_sessions:
            raise RuntimeError(
                "{} has incomplete or reordered sessions".format(dataset)
            )
        for session in seed_row["sessions"]:
            observed = tuple(session["modes"]["full"]["methods"].keys())
            # ``atomic_write_json`` sorts object keys, so nested mapping order is
            # not a protocol signal.  The authoritative row order is the
            # top-level ``methods`` list checked above; nested rows must contain
            # exactly the same unique method set.
            if len(observed) != len(METHODS) or set(observed) != set(METHODS):
                raise RuntimeError("session method rows do not match the protocol")
    provenance = payload.get("selection_provenance", {})
    for item in (
        "shrinksel",
        "transductive",
        "transductive_audit",
        "qpr",
        "qpr_audit",
        "shrinksel_audit",
    ):
        if not provenance.get(item):
            raise RuntimeError("{} provenance is missing".format(item))
    qpr_metadata = payload.get("resources", {}).get("qpr", {}).get(
        "last_metadata", {}
    )
    if qpr_metadata.get("uses_query_labels") is not False:
        raise RuntimeError("QPR query-label boundary is not explicitly false")
    if qpr_metadata.get("uses_query_marginal_constraint") is not False:
        raise RuntimeError("QPR marginal boundary is not explicitly false")
    if qpr_metadata.get("updates_base_prototypes") is not False:
        raise RuntimeError("QPR frozen-base boundary is not explicitly false")
    if qpr_metadata.get("preserves_input_prototype_radius") is not True:
        raise RuntimeError("QPR did not preserve revised prototype radii")
    protolp_metadata = payload.get("resources", {}).get(
        "protolp_paper", {}
    ).get("last_metadata", {})
    if protolp_metadata.get("support_assignment") != "one_hot_labels_per_Eq_4":
        raise RuntimeError("ProtoLP did not clamp labelled supports per Eq. 4")
    if protolp_metadata.get("uses_query_marginal_constraint") is not False:
        raise RuntimeError("the primary ProtoLP row must be marginal-free")
    laplacian_metadata = payload.get("resources", {}).get(
        "laplacian_shot", {}
    ).get("last_metadata", {})
    if laplacian_metadata.get("includes_cl2n") is not True:
        raise RuntimeError("LaplacianShot-full did not apply released CL2N")
    if laplacian_metadata.get("binary_directed_euclidean_knn") is not True:
        raise RuntimeError("LaplacianShot-full graph semantics are not explicit")
    laplacian_core_metadata = payload.get("resources", {}).get(
        "laplacian_shot_core", {}
    ).get("last_metadata", {})
    if laplacian_core_metadata.get("includes_cl2n") is not False:
        raise RuntimeError("LaplacianShot-core silently applied CL2N")
    if laplacian_core_metadata.get("includes_prototype_rectification") is not False:
        raise RuntimeError("LaplacianShot-core silently applied rectification")
    map_metadata = payload.get("resources", {}).get(
        "ptmap_uniform", {}
    ).get("last_metadata", {})
    if map_metadata.get("power_transform_applied") is not False:
        raise RuntimeError("MAP-OT core silently applied a power transform")
    if map_metadata.get("power_transform_reason") != "signed_CLIP_embeddings":
        raise RuntimeError("MAP-OT/PT-MAP naming boundary is missing")
    methods: Dict[str, object] = {}
    for method in METHODS:
        final = final_rows(payload, method)
        session = session_average(payload, method)
        methods[method] = {
            "label": METHOD_LABELS[method],
            "session_average_mean": float(mean(session)),
            "session_average_std": float(np.std(session)),
            "final_overall_mean": float(mean(float(row["overall"]) for row in final)),
            "final_overall_std": float(
                np.std([float(row["overall"]) for row in final])
            ),
            "final_base_mean": float(mean(float(row["base"]) for row in final)),
            "final_base_std": float(np.std([float(row["base"]) for row in final])),
            "final_novel_mean": float(mean(float(row["novel"]) for row in final)),
            "final_novel_std": float(np.std([float(row["novel"]) for row in final])),
        }
    qpr_session = session_average(payload, "qpr")
    comparisons = {}
    for index, comparator in enumerate(METHODS):
        if comparator == "qpr":
            continue
        delta, interval = paired_interval(
            qpr_session, session_average(payload, comparator), seed + index * 1009
        )
        comparisons[comparator] = {
            "qpr_minus_comparator_session_average": delta,
            "bootstrap_95": interval,
        }
    final_qpr = final_rows(payload, "qpr")
    final_shrink = final_rows(payload, "shrinksel")
    for index, metric in enumerate(("overall", "base", "novel")):
        delta, interval = paired_interval(
            [float(row[metric]) for row in final_qpr],
            [float(row[metric]) for row in final_shrink],
            seed + 50021 + index * 2017,
        )
        methods["qpr"]["final_delta_" + metric + "_vs_shrinksel"] = delta
        methods["qpr"][
            "final_delta_" + metric + "_vs_shrinksel_bootstrap_95"
        ] = interval
    marginal_free = (
        "qpr",
        "tim_gd",
        "laplacian_shot",
        "laplacian_shot_core",
        "protolp_paper",
    )
    best = max(marginal_free, key=lambda name: methods[name]["session_average_mean"])
    return {
        "methods": methods,
        "qpr_comparisons": comparisons,
        "best_marginal_free_i2": best,
        "best_marginal_free_i2_label": METHOD_LABELS[best],
        "qpr_is_best_marginal_free_i2": best == "qpr",
        "result_sha256": payload["result_sha256"],
        "protocol_sha256": payload["protocol_sha256"],
        "support_sha256": payload["support_sha256"],
    }


def attach_inductive_cosine(
    row: Dict[str, object],
    inductive: Mapping[str, object],
    transductive: Mapping[str, object],
    seed: int,
) -> None:
    """Attach the matched I0 cosine anchor without changing I2 rankings."""
    if inductive.get("stage") != "strict_inductive_target_evaluation_after_parameter_lock":
        raise RuntimeError("unexpected inductive cosine source stage")
    for key in ("dataset", "protocol_sha256", "support_sha256"):
        if inductive.get(key) != transductive.get(key):
            raise RuntimeError("inductive/transductive mismatch for {}".format(key))
    if inductive.get("feature_file_sha256") != transductive.get("feature_file_sha256"):
        raise RuntimeError("inductive/transductive feature mismatch")
    if list(inductive.get("support_seeds", [])) != list(range(1, 21)):
        raise RuntimeError("inductive cosine anchor lacks the locked 20 draws")
    cosine = inductive.get("summary", {}).get("cosine")
    if not isinstance(cosine, dict):
        raise RuntimeError("inductive source lacks cosine summary")
    per_seed = [float(value) for value in cosine["session_average_per_seed"]]
    if len(per_seed) != 20:
        raise RuntimeError("inductive cosine anchor lacks 20 paired values")
    row["methods"]["cosine"] = {
        "label": METHOD_LABELS["cosine"],
        "session_average_mean": float(cosine["session_average_mean"]),
        "session_average_std": float(cosine["session_average_std"]),
        "final_overall_mean": float(cosine["final_accuracy_mean"]),
        "final_overall_std": float(cosine["final_accuracy_std"]),
        "information_level": "I0/I1 strict inductive",
    }
    qpr_session = session_average(transductive, "qpr")
    delta, interval = paired_interval(qpr_session, per_seed, seed)
    row["qpr_comparisons"]["cosine"] = {
        "qpr_minus_comparator_session_average": delta,
        "bootstrap_95": interval,
    }


def latex(status: Mapping[str, object]) -> str:
    lines = [
        "% Generated from transductive_benchmark/STATUS.json; do not edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched query-assisted benchmark on frozen CLIP ViT-B/16. Values are session-average accuracy over all FSCIL sessions (mean$\pm$std over 20 support draws); $\Delta$ is relative to the support-only ShrinkSel anchor. LaplacianShot-full includes released preprocessing, whereas LaplacianShot-core applies only its graph update to matched normalized features. MAP-OT$_u$ assumes a uniform query marginal and is ranked separately. The best marginal-free query-assisted result is bold.}",
        r"\label{tab:transductive-benchmark}",
        r"\small",
        r"\setlength{\tabcolsep}{4.1pt}",
        r"\begin{tabular}{lccc|ccc}",
        r"\toprule",
        r"Method & CUB & CIFAR & miniIN & $\Delta$CUB & $\Delta$CIFAR & $\Delta$miniIN \\",
        r"\midrule",
    ]
    datasets = status["datasets"]
    for method in TABLE_METHODS:
        values = []
        deltas = []
        for dataset in DATASETS:
            row = datasets[dataset]
            score = float(row["methods"][method]["session_average_mean"])
            deviation = float(row["methods"][method]["session_average_std"])
            formatted = f"{score:.2f}$\\pm${deviation:.2f}"
            if row["best_marginal_free_i2"] == method:
                values.append(r"\textbf{" + formatted + "}")
            else:
                values.append(formatted)
            baseline = float(row["methods"]["shrinksel"]["session_average_mean"])
            deltas.append(f"{score - baseline:+.2f}")
        label = METHOD_LATEX_LABELS[method]
        if method in ("shrinksel", "ptmap_uniform"):
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(values + deltas) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def final_latex(status: Mapping[str, object]) -> str:
    lines = [
        "% Generated from transductive_benchmark/STATUS.json; do not edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final-session accuracy (Full/Base/Novel) for the inductive reference and QPR over twenty matched support draws. The last column gives QPR's paired mean change.}",
        r"\label{tab:supp-qpr-final}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & ShrinkSel & ShrinkSel+QPR & $\Delta$ Full/Base/Novel \\",
        r"\midrule",
    ]
    labels = {"cub200": "CUB-200", "cifar100": "CIFAR-100", "miniimagenet": "miniImageNet"}
    for dataset in DATASETS:
        methods = status["datasets"][dataset]["methods"]
        base = methods["shrinksel"]
        qpr = methods["qpr"]
        base_triplet = "/".join(
            f"{float(base['final_' + key + '_mean']):.2f}"
            for key in ("overall", "base", "novel")
        )
        qpr_triplet = "/".join(
            f"{float(qpr['final_' + key + '_mean']):.2f}"
            for key in ("overall", "base", "novel")
        )
        delta_triplet = "/".join(
            f"{float(qpr['final_delta_' + key + '_vs_shrinksel']):+.2f}"
            for key in ("overall", "base", "novel")
        )
        lines.append(
            labels[dataset]
            + " & "
            + base_triplet
            + " & "
            + qpr_triplet
            + " & "
            + delta_triplet
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def comparison_latex(status: Mapping[str, object]) -> str:
    lines = [
        "% Generated from transductive_benchmark/STATUS.json; do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired difference in session-average accuracy between QPR and each comparator over twenty matched support draws. Brackets are paired-bootstrap 95\% intervals. MAP-OT$_u$ (PT-MAP core) uses an additional uniform-prior assumption.}",
        r"\label{tab:supp-qpr-paired}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Dataset & Comparator & QPR $-$ comparator [95\%] \\",
        r"\midrule",
    ]
    labels = {
        "cub200": "CUB-200",
        "cifar100": "CIFAR-100",
        "miniimagenet": "miniImageNet",
    }
    comparators = (
        "cosine",
        "shrinksel",
        "tim_gd",
        "laplacian_shot",
        "laplacian_shot_core",
        "protolp_paper",
        "ptmap_uniform",
    )
    for dataset in DATASETS:
        row = status["datasets"][dataset]
        for index, comparator in enumerate(comparators):
            comparison = row["qpr_comparisons"][comparator]
            delta = float(comparison["qpr_minus_comparator_session_average"])
            low, high = [float(value) for value in comparison["bootstrap_95"]]
            dataset_label = labels[dataset] if index == 0 else ""
            lines.append(
                dataset_label
                + " & "
                + METHOD_LATEX_LABELS[comparator]
                + " & "
                + f"{delta:+.2f} [{low:+.2f}, {high:+.2f}]"
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--inductive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latex-output", type=Path)
    parser.add_argument("--final-latex-output", type=Path)
    parser.add_argument("--comparison-latex-output", type=Path)
    args = parser.parse_args()
    status: Dict[str, object] = {
        "schema_version": 3,
        "stage": "matched_transductive_benchmark",
        "shrinksel_variant": "revised radial-aware robust selector",
        "information_boundary": {
            "inductive_reference": ["cosine", "shrinksel"],
            "marginal_free_i2": [
                "qpr",
                "tim_gd",
                "laplacian_shot",
                "laplacian_shot_core",
                "protolp_paper",
            ],
            "uniform_prior": ["ptmap_uniform"],
            "query_labels_used": False,
            "target_hyperparameter_selection": False,
        },
        "datasets": {},
    }
    for index, dataset in enumerate(DATASETS):
        payload = load(args.input_root / (dataset + "_all20_full.json"))
        if payload["dataset"] != dataset:
            raise RuntimeError("dataset/file mismatch")
        row = summarize_dataset(
            payload, dataset, 1729 + index * 10007
        )
        inductive = load(args.inductive_root / (dataset + "_inductive.json"))
        attach_inductive_cosine(
            row,
            inductive,
            payload,
            89021 + index * 10007,
        )
        status["datasets"][dataset] = row
    status["qpr_best_marginal_free_count"] = sum(
        int(row["qpr_is_best_marginal_free_i2"])
        for row in status["datasets"].values()
    )
    status["result_sha256"] = stable_json_sha256(status)
    atomic_write_json(args.output, status)
    if args.latex_output is not None:
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(latex(status), encoding="utf-8")
    if args.final_latex_output is not None:
        args.final_latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.final_latex_output.write_text(final_latex(status), encoding="utf-8")
    if args.comparison_latex_output is not None:
        args.comparison_latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_latex_output.write_text(
            comparison_latex(status), encoding="utf-8"
        )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
