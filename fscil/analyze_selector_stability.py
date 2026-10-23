#!/usr/bin/env python3
"""Leave-one-fold-out stability analysis for saved base-only ShrinkSel selectors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Dict, Mapping, Sequence

TIE_RANK = {"raw": 2, "resultant_debiased": 1, "normalized": 0}


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
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_key(row: Mapping[str, object]) -> str:
    return "{}|{}".format(row["prototype_mode"], float(row["lambda"]))


def _rank(row: Mapping[str, object]) -> tuple[float, float, int, float]:
    return (
        float(row["mean_gain_vs_shared_lda"]),
        float(row["worst_fold_gain_vs_shared_lda"]),
        TIE_RANK[str(row["prototype_mode"])],
        -float(row["lambda"]),
    )


def _select(
    candidates: Sequence[Mapping[str, object]],
    retained_folds: Sequence[int],
    reference_key: str | None = None,
) -> Dict[str, object]:
    evaluated = []
    for candidate in candidates:
        source = [float(value) for value in candidate["fold_gains_vs_shared_lda"]]
        gains = [source[index] for index in retained_folds]
        evaluated.append(
            {
                "prototype_mode": str(candidate["prototype_mode"]),
                "lambda": float(candidate["lambda"]),
                "fold_gains_vs_shared_lda": gains,
                "mean_gain_vs_shared_lda": float(mean(gains)),
                "worst_fold_gain_vs_shared_lda": float(min(gains)),
            }
        )
    feasible = [
        row for row in evaluated if row["worst_fold_gain_vs_shared_lda"] >= 0.0
    ]
    if not feasible:
        raise RuntimeError("shared-LDA endpoint is unexpectedly infeasible")
    ranked = sorted(feasible, key=_rank, reverse=True)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    result = {
        "retained_folds": [int(value) for value in retained_folds],
        "feasible_candidate_count": len(feasible),
        "winner": winner,
        "winner_key": _candidate_key(winner),
        "runner_up": runner_up,
        "runner_up_key": _candidate_key(runner_up) if runner_up else None,
        "winner_minus_runner_up_mean_gain": (
            float(
                winner["mean_gain_vs_shared_lda"]
                - runner_up["mean_gain_vs_shared_lda"]
            )
            if runner_up
            else None
        ),
    }
    if reference_key is not None:
        reference = next(
            (row for row in evaluated if _candidate_key(row) == reference_key), None
        )
        if reference is None:
            raise RuntimeError("reference candidate is absent")
        result.update(
            {
                "reference_key": reference_key,
                "reference_feasible": bool(
                    reference["worst_fold_gain_vs_shared_lda"] >= 0.0
                ),
                "winner_minus_reference_mean_gain": float(
                    winner["mean_gain_vs_shared_lda"]
                    - reference["mean_gain_vs_shared_lda"]
                ),
                "winner_matches_reference_prototype_mode": bool(
                    winner["prototype_mode"] == reference["prototype_mode"]
                ),
                "winner_lambda_distance_from_reference": abs(
                    float(winner["lambda"]) - float(reference["lambda"])
                ),
            }
        )
    return result


def analyze_selection(name: str, path: Path) -> Dict[str, object]:
    payload = _load(path)
    if payload.get("stage") != "base_selection_radial_shrinksel":
        raise RuntimeError(name + " is not a radial selection artifact")
    if payload.get("selection_scheme") != "four_fold_class_disjoint_cross_validation":
        raise RuntimeError(name + " lacks the locked four-fold selector")
    embedded = str(payload["selection_result_sha256"])
    unhashed = copy.deepcopy(dict(payload))
    unhashed.pop("selection_result_sha256")
    if stable_json_sha256(unhashed) != embedded:
        raise RuntimeError(name + " selection hash mismatch")
    candidates = payload["robust_selection"]["candidates"]
    if len(candidates) != 21:
        raise RuntimeError(name + " does not contain the complete 3x7 grid")
    fold_count = len(candidates[0]["fold_gains_vs_shared_lda"])
    if fold_count != 4 or any(
        len(row["fold_gains_vs_shared_lda"]) != fold_count for row in candidates
    ):
        raise RuntimeError(name + " has an invalid fold grid")
    full = _select(candidates, list(range(fold_count)))
    locked_key = "{}|{}".format(
        payload["selected_prototype_mode"], float(payload["selected_lambda"])
    )
    if full["winner_key"] != locked_key:
        raise RuntimeError(name + " cannot reproduce the locked winner")
    leave_one_out = []
    for omitted in range(fold_count):
        retained = [value for value in range(fold_count) if value != omitted]
        row = _select(candidates, retained, locked_key)
        row["omitted_fold"] = omitted
        row["matches_locked_winner"] = row["winner_key"] == locked_key
        leave_one_out.append(row)
    defined_margins = [
        float(row["winner_minus_runner_up_mean_gain"])
        for row in leave_one_out
        if row["winner_minus_runner_up_mean_gain"] is not None
    ]
    return {
        "name": name,
        "dataset": payload["dataset"],
        "selection_path": str(path.resolve()),
        "selection_result_sha256": embedded,
        "locked_winner_key": locked_key,
        "full_four_fold": full,
        "leave_one_fold_out": leave_one_out,
        "locked_winner_retained_count": sum(
            int(row["matches_locked_winner"]) for row in leave_one_out
        ),
        "locked_winner_retained_fraction": float(
            mean(float(row["matches_locked_winner"]) for row in leave_one_out)
        ),
        "locked_prototype_mode_retained_count": sum(
            int(row["winner_matches_reference_prototype_mode"])
            for row in leave_one_out
        ),
        "locked_winner_base_score_within_0.10_count": sum(
            int(float(row["winner_minus_reference_mean_gain"]) <= 0.10)
            for row in leave_one_out
        ),
        "locked_winner_base_score_within_0.25_count": sum(
            int(float(row["winner_minus_reference_mean_gain"]) <= 0.25)
            for row in leave_one_out
        ),
        "maximum_leave_one_out_locked_regret": max(
            float(row["winner_minus_reference_mean_gain"])
            for row in leave_one_out
        ),
        "locked_candidate_infeasible_omission_count": sum(
            int(not row["reference_feasible"]) for row in leave_one_out
        ),
        "unique_leave_one_out_winners": sorted(
            {str(row["winner_key"]) for row in leave_one_out}
        ),
        "minimum_leave_one_out_margin": (
            min(defined_margins) if defined_margins else None
        ),
        "single_feasible_candidate_omission_count": sum(
            int(row["winner_minus_runner_up_mean_gain"] is None)
            for row in leave_one_out
        ),
        "minimum_feasible_candidate_count": min(
            int(row["feasible_candidate_count"]) for row in leave_one_out
        ),
        "maximum_feasible_candidate_count": max(
            int(row["feasible_candidate_count"]) for row in leave_one_out
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load(manifest_path)
    settings = manifest.get("settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("selector manifest requires a non-empty settings list")
    base = manifest_path.parent
    rows = []
    for setting in settings:
        raw_path = Path(str(setting["selection_path"]))
        path = raw_path if raw_path.is_absolute() else (base / raw_path).resolve()
        rows.append(analyze_selection(str(setting["name"]), path))
    result: Dict[str, object] = {
        "schema_version": 1,
        "stage": "saved_base_only_selector_leave_one_fold_out_stability",
        "information_boundary": {
            "target_features_used": False,
            "target_labels_used": False,
            "saved_base_only_fold_curves_used": True,
            "parameters_changed": False,
        },
        "manifest": str(manifest_path),
        "manifest_sha256": stable_json_sha256(manifest),
        "setting_count": len(rows),
        "settings": rows,
        "aggregate": {
            "leave_one_out_checks": 4 * len(rows),
            "locked_winner_retained_checks": sum(
                int(row["locked_winner_retained_count"]) for row in rows
            ),
            "locked_winner_retained_fraction": float(
                mean(float(row["locked_winner_retained_fraction"]) for row in rows)
            ),
            "settings_stable_on_all_four_omissions": sum(
                int(row["locked_winner_retained_count"] == 4) for row in rows
            ),
            "locked_prototype_mode_retained_fraction": float(
                sum(int(row["locked_prototype_mode_retained_count"]) for row in rows)
                / (4 * len(rows))
            ),
            "locked_winner_base_score_within_0.10_fraction": float(
                sum(
                    int(row["locked_winner_base_score_within_0.10_count"])
                    for row in rows
                )
                / (4 * len(rows))
            ),
            "locked_winner_base_score_within_0.25_fraction": float(
                sum(
                    int(row["locked_winner_base_score_within_0.25_count"])
                    for row in rows
                )
                / (4 * len(rows))
            ),
            "maximum_leave_one_out_locked_regret": max(
                float(row["maximum_leave_one_out_locked_regret"]) for row in rows
            ),
            "locked_candidate_infeasible_omission_count": sum(
                int(row["locked_candidate_infeasible_omission_count"])
                for row in rows
            ),
            "minimum_leave_one_out_margin": min(
                float(row["minimum_leave_one_out_margin"])
                for row in rows
                if row["minimum_leave_one_out_margin"] is not None
            ),
            "single_feasible_candidate_omission_count": sum(
                int(row["single_feasible_candidate_omission_count"])
                for row in rows
            ),
        },
    }
    result["result_sha256"] = stable_json_sha256(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
