from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from fscil.qpr_calibration import _clean_episode
from fscil.cumulative_radial_selection import cumulative_requested_counts
from fscil.diagnostics import two_way_bootstrap
from fscil.geometry import covariance_eigensystem, metric_scores
from fscil.qpr_selection import deployed_qpr_choice
from fscil.qpr_self_exclusion import qpr_same_and_leave_one_out_scores
from fscil.radial_selection import deployed_radial_choice
from fscil.transductive_methods import query_prototype_rectification
from fscil.transductive_selection import BaseTransductivePools


class MethodResilienceTest(unittest.TestCase):
    def test_two_way_bootstrap_serializes_ratio_denominator(self) -> None:
        k_class = np.asarray([[0.4, 0.6], [0.5, 0.7]], dtype=np.float64)
        full_class = np.asarray([0.7, 0.8], dtype=np.float64)
        lp_class = np.asarray([0.8, 0.9], dtype=np.float64)
        result = two_way_bootstrap(
            k_class, full_class, lp_class, draws=200, seed=17
        )
        self.assertIn("total_LP_minus_K", result)
        self.assertAlmostEqual(
            result["estimation_term"]["mean"]
            + result["form_term"]["mean"],
            result["total_LP_minus_K"]["mean"],
            places=10,
        )
        self.assertGreater(result["total_LP_minus_K"]["ci95"][0], 0.0)

    def test_radial_audit_deploys_exact_lda_fallback(self) -> None:
        parameters = {
            "selection_result_sha256": "selection",
            "selected_prototype_mode": "resultant_debiased",
            "selected_lambda": 0.75,
        }
        passed = {
            "selection_result_sha256": "selection",
            "passes_non_regression_gate": True,
        }
        failed = {
            "selection_result_sha256": "selection",
            "passes_non_regression_gate": False,
        }
        self.assertEqual(
            deployed_radial_choice(parameters, passed),
            ("resultant_debiased", 0.75),
        )
        self.assertEqual(deployed_radial_choice(parameters, failed), ("raw", 0.0))

    def test_qpr_audit_deploys_no_update_for_legacy_failed_audit(self) -> None:
        parameters = {
            "selection_result_sha256": "selection",
            "selected": {
                "alpha": 0.75,
                "mass_threshold": 0.6,
                "min_count": 5,
                "logit_scale": 0.3,
            },
        }
        failed_legacy_audit = {
            "selection_result_sha256": "selection",
            "passes_transfer_safety_gate": False,
        }
        self.assertEqual(
            deployed_qpr_choice(parameters, failed_legacy_audit),
            {
                "alpha": 0.0,
                "mass_threshold": 0.6,
                "min_count": 5,
                "logit_scale": 0.3,
            },
        )

    def test_cumulative_schedule_matches_public_class_schedule(self) -> None:
        self.assertEqual(
            cumulative_requested_counts("cifar100"),
            (5, 10, 15, 20, 25, 30, 35, 40),
        )
        self.assertEqual(
            cumulative_requested_counts("miniimagenet"),
            (5, 10, 15, 20, 25, 30, 35, 40),
        )
        self.assertEqual(
            cumulative_requested_counts("cub200"),
            (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
        )

    def test_clean_qpr_episode_is_identity_disjoint_and_uses_parent_radius(self) -> None:
        torch.manual_seed(7)
        construction = {
            class_id: F.normalize(torch.randn(18, 6) + class_id, dim=1)
            for class_id in (0, 1)
        }
        anchor_cohort = {
            class_id: F.normalize(torch.randn(15, 6) + class_id, dim=1)
            for class_id in (0, 1)
        }
        candidate = {
            class_id: F.normalize(torch.randn(25, 6) + class_id, dim=1)
            for class_id in (2, 3)
        }
        pools = BaseTransductivePools(
            dataset="cifar100",
            construction=construction,
            candidate=candidate,
            bundle_sha256="bundle",
            source_feature_file_sha256="features",
        )
        episode = _clean_episode(
            pools,
            construction,
            anchor_cohort,
            candidate,
            episode_index=0,
            imbalanced=False,
            requested_pseudo_count=2,
            prototype_mode="raw",
        )
        self.assertFalse(
            set(episode.anchor_identity_ids).intersection(episode.pseudo_identity_ids)
        )
        self.assertTrue(
            torch.allclose(episode.prototypes[0], construction[0].mean(0))
        )
        self.assertNotAlmostEqual(float(episode.prototypes[0].norm()), 1.0)

    def test_vectorized_self_exclusion_matches_literal_reruns(self) -> None:
        torch.manual_seed(19)
        dimension = 8
        query = F.normalize(torch.randn(12, dimension), dim=1)
        prototypes = F.normalize(torch.randn(3, dimension), dim=1)
        baseline_scores = torch.full((12, 3), -4.0)
        baseline_scores[:6, 1] = 4.0
        baseline_scores[:6, 2] = 1.0
        baseline_scores[6:, 1] = 1.0
        baseline_scores[6:, 2] = 4.0
        covariance_pool = {
            class_id: F.normalize(torch.randn(20, dimension), dim=1)
            for class_id in range(3)
        }
        covariance = covariance_eigensystem(covariance_pool, range(3))
        vectorized = qpr_same_and_leave_one_out_scores(
            query,
            prototypes,
            baseline_scores,
            covariance,
            shrinkage=0.5,
            ridge_gamma=1.0,
            num_base_classes=1,
            alpha=0.75,
            mass_threshold=0.3,
            min_count=5,
            logit_scale=1.0,
        )
        literal_rows = []
        for heldout in range(query.shape[0]):
            retained = torch.arange(query.shape[0]) != heldout
            adapted = query_prototype_rectification(
                query[retained],
                prototypes,
                baseline_scores[retained],
                1,
                alpha=0.75,
                mass_threshold=0.3,
                min_count=5,
                logit_scale=1.0,
            )
            literal_rows.append(
                metric_scores(
                    query[heldout : heldout + 1],
                    adapted.prototypes,
                    covariance,
                    0.5,
                    1.0,
                ).squeeze(0)
            )
        literal = torch.stack(literal_rows, 0)
        self.assertTrue(
            torch.allclose(
                vectorized.leave_one_out_scores,
                literal,
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertEqual(
            vectorized.metadata["corrected_query_class_score_entries"], 12
        )
        self.assertFalse(
            vectorized.metadata[
                "classified_query_present_in_its_adaptation_set"
            ]
        )


if __name__ == "__main__":
    unittest.main()
