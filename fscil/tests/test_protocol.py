from __future__ import annotations

import dataclasses
import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from fscil.aircraft_data import locked_class_order as aircraft_locked_class_order
from fscil.dtd_data import (
    DTD_ARCHIVE_BYTES,
    DTD_ARCHIVE_NAME,
    DTD_ARCHIVE_SHA256,
    locked_class_order as dtd_locked_class_order,
)
from fscil.prepare_aircraft_dataset import (
    AIRCRAFT_ARCHIVE_BYTES,
    ARCHIVE_NAME,
    EXTRACTED_NAME,
)
from fscil.baseline_methods import (
    ULAParameters,
    cosine_scores_chunked,
    euclidean_scores_chunked,
    fecam_inverse_covariance,
    hycal_class_statistics,
    hycal_scores_chunked,
    hycal_squared_mahalanobis_chunked,
    ranpac_low_rank_inverse_apply,
    ranpac_low_rank_system,
    ranpac_normal_equations,
    ranpac_normal_equation_relative_residual,
    ranpac_scores_chunked,
    ranpac_transform,
    solve_ranpac_low_rank_update,
    solve_ranpac,
    solve_ranpac_from_eigensystem,
    solve_ranpac_from_low_rank_system,
    teen_calibrate,
    ula_adjust_logits,
)
from fscil.constants import RIDGE_GAMMA
from fscil.build_protocol_bundle import (
    nested_diagnostic_supports,
    sample_standard_supports,
)
from fscil.geometry import (
    cosine_scores,
    covariance_eigensystem,
    metric_scores,
    metric_scores_rotated,
    metric_trace_penalty,
    rotate_to_covariance_basis,
)
from fscil.qpr_base_guard import base_guarded_prototypes
from fscil.radial_selection import (
    PROTOTYPE_MODES,
    _best_crossvalidated_candidate,
    prototype_from_samples,
    session_average_group_weights,
)
from fscil.randumb_ncm import make_rbf_projection, rbf_transform
from fscil.selection import AuditData, ConstructionSelectionData
from fscil.transductive_methods import (
    _protolp_assignment,
    _protolp_paper_graph_scores,
    _laplacian_bound_update,
    chunked_sparse_euclidean_knn,
    chunked_sparse_cosine_knn,
    laplacian_shot,
    laplacian_shot_full,
    marginal_constrained_scores,
    protolp_paper,
    protolp_unconstrained,
    protolp_uniform,
    pt_map,
    query_prototype_rectification,
    sparse_graph_repair,
    squared_euclidean,
    support_prototypes,
    tim_gd,
)
from fscil.transductive_selection import BaseTransductivePools, select as select_transductive


class ProtocolIsolationTest(unittest.TestCase):
    def test_radial_selector_uses_protocol_weights_and_debiased_resultant(self) -> None:
        cifar_base, cifar_novel = session_average_group_weights("cifar100")
        cub_base, cub_novel = session_average_group_weights("cub200")
        self.assertAlmostEqual(cifar_base + cifar_novel, 1.0)
        self.assertAlmostEqual(cub_base + cub_novel, 1.0)
        self.assertLess(cifar_novel, 0.5)
        self.assertLess(cub_novel, 0.5)
        identical = torch.tensor([[1.0, 0.0]] * 5)
        corrected = prototype_from_samples(identical, "resultant_debiased")
        self.assertTrue(torch.allclose(corrected, torch.tensor([1.0, 0.0])))
        orthogonal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        corrected = prototype_from_samples(orthogonal, "resultant_debiased")
        self.assertTrue(torch.equal(corrected, torch.zeros(2)))

    def test_radial_selector_rejects_fold_regression(self) -> None:
        from fscil.constants import LAMBDA_GRID

        folds = []
        for fold_index in range(2):
            curve = {}
            for mode in PROTOTYPE_MODES:
                curve[mode] = {}
                for shrinkage in LAMBDA_GRID:
                    score = 0.0 if mode == "raw" and float(shrinkage) == 0.0 else -5.0
                    curve[mode][str(float(shrinkage))] = {
                        "session_weighted": {"mean": score}
                    }
            curve["normalized"]["0.5"]["session_weighted"]["mean"] = (
                10.0 if fold_index == 0 else -1.0
            )
            curve["resultant_debiased"]["0.25"]["session_weighted"]["mean"] = 1.0
            folds.append({"curve": curve})
        mode, shrinkage, details = _best_crossvalidated_candidate(
            {"fold_evaluations": folds}, "session_weighted"
        )
        self.assertEqual((mode, shrinkage), ("resultant_debiased", 0.25))
        self.assertGreaterEqual(
            details["selected"]["worst_fold_gain_vs_shared_lda"], 0.0
        )

    def test_factorial_arms_preserve_declared_endpoints(self) -> None:
        from fscil.run_radial_factorial_ablation import RIDGE_GRID, component_arms

        arms = component_arms("resultant_debiased", 0.75)
        self.assertEqual(arms["cosine"], ("normalized", 1.0))
        self.assertEqual(arms["shared_lda"], ("raw", 0.0))
        self.assertEqual(arms["geometry_only"], ("normalized", 0.75))
        self.assertEqual(arms["radius_only"], ("resultant_debiased", 1.0))
        self.assertEqual(arms["shrinksel"], ("resultant_debiased", 0.75))
        self.assertIn(RIDGE_GAMMA, RIDGE_GRID)

    def test_qpr_ablation_keeps_reported_configuration_and_is_oat(self) -> None:
        from fscil.run_qpr_ablation import (
            component_configurations,
            sensitivity_configurations,
        )

        selected = (0.75, 0.6, 5, 0.03)
        components = component_configurations(selected)
        self.assertEqual(
            components["qpr_selected"],
            {
                "alpha": 0.75,
                "mass_threshold": 0.6,
                "min_count": 5,
                "logit_scale": 0.03,
                "posterior_weighting": True,
                "preserve_radius": True,
                "require_predicted_novel": True,
            },
        )
        self.assertFalse(
            components["qpr_no_predicted_novel_gate"]["require_predicted_novel"]
        )
        self.assertEqual(len(components), 8)
        sensitivity = sensitivity_configurations(selected)
        self.assertEqual(len(sensitivity), 18)
        self.assertEqual(sensitivity["alpha=0.0"]["mass_threshold"], 0.6)
        self.assertEqual(sensitivity["mass=0.0"]["alpha"], 0.75)
        self.assertEqual(sensitivity["min_count=20"]["logit_scale"], 0.03)
        self.assertEqual(sensitivity["scale=1.0"]["min_count"], 5)

    def test_adaptation_audit_evaluates_the_locked_choice(self) -> None:
        from fscil.run_adaptation_baselines import _audit_locked_hyperparameters

        selection = {
            "selected": {
                "clip_adapter_visual": "alpha0.2",
                "adam_concat": "alpha0.2",
                "lp_joint_replay": "steps300",
            }
        }
        audit = {
            "curves": {
                "clip_adapter_visual": {"alpha0.2": 4.0, "alpha0.5": 5.0},
                "adam_concat": {"alpha0.2": 7.0, "alpha0.5": 6.0},
                "lp_joint_replay": {"steps100": 8.0, "steps300": 9.0},
            }
        }
        rows = _audit_locked_hyperparameters(selection, audit)
        self.assertEqual(rows["clip_adapter_visual"]["locked_key"], "alpha0.2")
        self.assertEqual(rows["clip_adapter_visual"]["audit_best_key"], "alpha0.5")
        self.assertAlmostEqual(rows["clip_adapter_visual"]["audit_regret"], 1.0)
        self.assertAlmostEqual(rows["adam_concat"]["audit_regret"], 0.0)
        self.assertAlmostEqual(rows["lp_joint_replay"]["audit_regret"], 0.0)

    def test_selector_api_has_no_target_test_fields(self) -> None:
        for data_class in (ConstructionSelectionData, AuditData):
            names = {field.name.lower() for field in dataclasses.fields(data_class)}
            self.assertFalse(any("test" in name for name in names))
            self.assertFalse(any("query" in name for name in names))
            self.assertFalse(any("target" in name for name in names))

    def test_transductive_benchmark_summary_reports_base_accuracy(self) -> None:
        from fscil.run_transductive import _summary

        row = {
            "overall": 75.0,
            "class_balanced": 74.0,
            "base": 80.0,
            "novel": 65.0,
        }
        summary = _summary({"full": {"qpr": [row, row]}})["full"]["qpr"]
        self.assertEqual(summary["base_mean"], 80.0)
        self.assertEqual(summary["base_std"], 0.0)
        self.assertEqual(summary["novel_mean"], 65.0)

    def test_only_explicit_i3_methods_request_the_realized_marginal(self) -> None:
        from fscil.run_transductive import (
            _requires_inference_marginal,
            _requires_protocol_counts,
        )

        marginal_free = (
            "qpr",
            "tim_gd",
            "laplacian_shot",
            "protolp_paper",
            "ptmap_uniform",
            "protolp_uniform",
        )
        self.assertFalse(_requires_inference_marginal(marginal_free))
        self.assertFalse(_requires_protocol_counts(marginal_free, ("full",)))
        self.assertTrue(
            _requires_protocol_counts(marginal_free, ("imbalanced10",))
        )
        self.assertTrue(
            _requires_inference_marginal(("shrinksel_known_marginal",))
        )

    def test_qpr_queue_uses_declared_benchmark_rows(self) -> None:
        from fscil.run_qpr_benchmark_queue import METHODS

        self.assertEqual(
            METHODS,
            (
                "shrinksel",
                "qpr",
                "tim_gd",
                "laplacian_shot",
                "laplacian_shot_core",
                "protolp_paper",
                "ptmap_uniform",
            ),
        )

    def test_qpr_base_session_noop_is_explicit(self) -> None:
        from fscil.run_qpr import base_only_noop_gate_diagnostics

        row = base_only_noop_gate_diagnostics()
        self.assertTrue(row["base_only_session_noop"])
        self.assertEqual(row["gated_query_count"], 0)
        self.assertEqual(row["eligible_class_count"], 0)
        self.assertEqual(row["updated_class_count"], 0)

    def test_transductive_selection_can_lock_only_reported_families(self) -> None:
        generator = torch.Generator().manual_seed(123)
        construction = {
            class_id: F.normalize(torch.randn(20, 8, generator=generator), dim=1)
            for class_id in range(4)
        }
        candidate = {
            class_id: F.normalize(torch.randn(24, 8, generator=generator), dim=1)
            for class_id in range(4, 9)
        }
        pools = BaseTransductivePools(
            dataset="cifar100",
            construction=construction,
            candidate=candidate,
            bundle_sha256="bundle",
            source_feature_file_sha256="feature",
        )
        result = select_transductive(
            pools, episodes=1, families=("protolp_paper",)
        )
        self.assertEqual(tuple(result["selected"]), ("protolp_paper",))
        self.assertEqual(
            result["evaluation"]["families"], ["protolp_paper"]
        )
        self.assertEqual(len(result["selected"]["protolp_paper"]["values"]), 2)

    def test_qpr_benchmark_analyzer_enforces_pairing_and_information(self) -> None:
        from fscil.analyze_transductive_benchmark import METHODS, summarize_dataset

        per_seed = []
        for support_seed in range(1, 21):
            sessions = []
            for session_index in range(9):
                methods = {}
                for method_index, method in enumerate(METHODS):
                    score = 70.0 + method_index + 0.01 * support_seed
                    methods[method] = {
                        "overall": score,
                        "class_balanced": score,
                        "base": score + 2.0,
                        "novel": None if session_index == 0 else score - 3.0,
                        "per_class": [],
                    }
                # JSON artifacts are written with sorted object keys.  Nested
                # mapping order must therefore not be mistaken for method order;
                # the top-level method manifest remains authoritative.
                methods = dict(sorted(methods.items()))
                sessions.append(
                    {
                        "session_index": session_index,
                        "modes": {"full": {"methods": methods}},
                    }
                )
            per_seed.append({"support_seed": support_seed, "sessions": sessions})
        payload = {
            "dataset": "cifar100",
            "stage": "target_transductive_evaluation_after_protocol_lock",
            "shrinksel_prototype_mode": "resultant_debiased",
            "shrinksel_lambda": 0.0,
            "methods": list(METHODS),
            "modes": ["full"],
            "session_scope": "all",
            "information_levels": {
                "shrinksel": "I0/I1 strict inductive",
                "qpr": (
                    "I2 unlabeled query cohort; QPR refines only novel prototypes"
                ),
                "tim_gd": "I2 unlabeled query cohort",
                "laplacian_shot": "I2 unlabeled query cohort",
                "laplacian_shot_core": (
                    "I2 unlabeled query cohort; graph/bound-update core"
                ),
                "protolp_paper": "I2 unlabeled query cohort",
                "ptmap_uniform": "I2 plus assumed uniform class marginal",
            },
            "support_seeds": list(range(1, 21)),
            "per_seed": per_seed,
            "selection_provenance": {
                "shrinksel": "shrink",
                "shrinksel_audit": "shrink-audit",
                "transductive": "selection",
                "transductive_audit": "selection-audit",
                "qpr": "qpr-selection",
                "qpr_audit": "qpr-audit",
            },
            "resources": {
                "qpr": {
                    "last_metadata": {
                        "uses_query_labels": False,
                        "uses_query_marginal_constraint": False,
                        "updates_base_prototypes": False,
                        "preserves_input_prototype_radius": True,
                    }
                },
                "protolp_paper": {
                    "last_metadata": {
                        "support_assignment": "one_hot_labels_per_Eq_4",
                        "uses_query_marginal_constraint": False,
                    }
                },
                "laplacian_shot": {
                    "last_metadata": {
                        "includes_cl2n": True,
                        "binary_directed_euclidean_knn": True,
                    }
                },
                "laplacian_shot_core": {
                    "last_metadata": {
                        "includes_cl2n": False,
                        "includes_prototype_rectification": False,
                    }
                },
                "ptmap_uniform": {
                    "last_metadata": {
                        "power_transform_applied": False,
                        "power_transform_reason": "signed_CLIP_embeddings",
                    }
                },
            },
            "result_sha256": "result",
            "protocol_sha256": "protocol",
            "support_sha256": "support",
        }
        summary = summarize_dataset(payload, "cifar100", 17)
        self.assertAlmostEqual(
            summary["methods"]["qpr"]["session_average_mean"], 71.105
        )
        self.assertAlmostEqual(
            summary["methods"]["qpr"]["final_delta_overall_vs_shrinksel"],
            1.0,
        )
        payload["resources"]["qpr"]["last_metadata"]["uses_query_labels"] = True
        with self.assertRaisesRegex(RuntimeError, "query-label boundary"):
            summarize_dataset(payload, "cifar100", 17)


class AircraftProtocolTest(unittest.TestCase):
    def test_aircraft_archive_contract_is_locked(self):
        self.assertEqual(AIRCRAFT_ARCHIVE_BYTES, 2_753_340_328)
        self.assertEqual(ARCHIVE_NAME, "fgvc-aircraft-2013b.tar.gz")
        self.assertEqual(EXTRACTED_NAME, "fgvc-aircraft-2013b")

    def test_seed_1993_class_order_is_locked(self) -> None:
        order = aircraft_locked_class_order()
        self.assertEqual(len(order), 100)
        self.assertEqual(set(order), set(range(100)))
        self.assertEqual(order[:10], (68, 56, 78, 8, 23, 84, 90, 65, 74, 76))

    def test_aircraft_support_draws_are_nested_and_reproducible(self) -> None:
        labels = np.repeat(np.arange(100, dtype=np.int64), 25)
        first = sample_standard_supports(labels, 50, 100, seed=1)
        repeated = sample_standard_supports(labels, 50, 100, seed=1)
        second = sample_standard_supports(labels, 50, 100, seed=2)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, second))
        nested = nested_diagnostic_supports(first, labels, 50, 100, seed=1)
        self.assertTrue(torch.equal(first, nested[:, :5]))
        actual = torch.as_tensor(labels[nested.numpy()])
        expected = torch.arange(50, 100)[:, None].expand_as(actual)
        self.assertTrue(torch.equal(actual, expected))


class DTDProtocolTest(unittest.TestCase):
    def test_dtd_release_contract_is_locked(self) -> None:
        self.assertEqual(DTD_ARCHIVE_NAME, "dtd-r1.0.1.tar.gz")
        self.assertEqual(DTD_ARCHIVE_BYTES, 625_239_812)
        self.assertEqual(
            DTD_ARCHIVE_SHA256,
            "e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
        )

    def test_dtd_seed_1993_class_order_is_locked(self) -> None:
        order = dtd_locked_class_order()
        self.assertEqual(len(order), 47)
        self.assertEqual(set(order), set(range(47)))
        self.assertEqual(order[:10], (10, 41, 38, 40, 37, 16, 21, 24, 22, 43))


class InductiveBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(37)
        self.queries = F.normalize(torch.randn(31, 12, generator=generator), dim=-1)
        self.base = F.normalize(torch.randn(7, 12, generator=generator), dim=-1)
        self.novel = F.normalize(torch.randn(3, 12, generator=generator), dim=-1)

    def test_teen_matches_released_formula(self) -> None:
        actual = teen_calibrate(self.base, self.novel, 0.5, 16.0)
        weights = torch.softmax(16.0 * self.novel.matmul(self.base.t()), dim=1)
        borrowed = F.normalize(weights.matmul(self.base), dim=-1)
        expected = F.normalize(0.5 * self.novel + 0.5 * borrowed, dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_ula_is_query_batch_and_order_invariant(self) -> None:
        prototypes = torch.cat([self.base, self.novel], 0)
        cosine = cosine_scores_chunked(self.queries, prototypes)
        parameters = ULAParameters()
        whole = ula_adjust_logits(cosine, self.base.shape[0], parameters)
        partitioned = torch.cat(
            [
                ula_adjust_logits(cosine[:11], self.base.shape[0], parameters),
                ula_adjust_logits(cosine[11:], self.base.shape[0], parameters),
            ],
            0,
        )
        self.assertTrue(torch.equal(whole.argmax(1), partitioned.argmax(1)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2)
        order = torch.randperm(self.queries.shape[0], generator=generator)
        reordered = ula_adjust_logits(cosine[order], self.base.shape[0], parameters)
        self.assertTrue(torch.equal(whole.argmax(1)[order], reordered.argmax(1)))

    def test_euclidean_chunking_is_exact(self) -> None:
        raw_centers = 0.7 * torch.cat([self.base, self.novel], 0)
        whole = euclidean_scores_chunked(self.queries, raw_centers, 1000)
        chunked = euclidean_scores_chunked(self.queries, raw_centers, 5)
        self.assertTrue(torch.allclose(whole, chunked, atol=1e-7, rtol=1e-6))

    def test_ranpac_normal_equations_and_eigensolve_match_direct(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(91)
        features = torch.randn(53, 12, generator=generator)
        labels = torch.arange(53) % 5
        gram, cross = ranpac_normal_equations(features, labels, 5, feature_chunk=9)
        one_hot = F.one_hot(labels, 5).float()
        self.assertTrue(torch.allclose(gram, features.t().matmul(features), atol=2e-5))
        self.assertTrue(torch.allclose(cross, features.t().matmul(one_hot), atol=2e-5))
        ridge = 0.3
        direct = solve_ranpac(gram, cross, ridge)
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.t()))
        via_eigen = solve_ranpac_from_eigensystem(
            eigenvalues.clamp_min(0.0), eigenvectors, cross, ridge
        )
        self.assertTrue(torch.allclose(direct, via_eigen, atol=2e-4, rtol=2e-4))
        direct_scores = ranpac_scores_chunked(self.queries, direct, query_chunk=1000)
        chunk_scores = ranpac_scores_chunked(self.queries, direct, query_chunk=4)
        self.assertTrue(
            torch.allclose(direct_scores, chunk_scores, atol=1e-6, rtol=1e-6)
        )

    def test_ranpac_projection_low_rank_update_matches_direct(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(303)
        base = torch.randn(61, 12, generator=generator)
        base_labels = torch.arange(61) % 4
        projection = torch.randn(12, 23, generator=generator)
        gram, cross = ranpac_normal_equations(
            base, base_labels, 6, projection=projection, feature_chunk=13
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.t()))
        added = torch.randn(10, 12, generator=generator)
        added_labels = torch.tensor([4] * 5 + [5] * 5)
        added_h = ranpac_transform(added, projection)
        ridge = 2.0
        updated = solve_ranpac_low_rank_update(
            eigenvalues.clamp_min(0.0),
            eigenvectors,
            cross,
            added_h,
            added_labels,
            ridge,
        )
        added_one_hot = F.one_hot(added_labels, 6).float()
        direct = solve_ranpac(
            gram + added_h.t().matmul(added_h),
            cross + added_h.t().matmul(added_one_hot),
            ridge,
        )
        self.assertTrue(torch.allclose(updated, direct, atol=5e-4, rtol=5e-4))

    def test_ranpac_float64_low_rank_solver_matches_direct(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1701)
        projected = torch.randn(19, 37, generator=generator)
        labels = torch.arange(19) % 6
        ridge = 0.07
        system = ranpac_low_rank_system(projected, labels, 6)
        stable = solve_ranpac_from_low_rank_system(system, ridge)
        targets = F.one_hot(labels, 6).double()
        values = projected.double()
        direct = torch.linalg.solve(
            values.t().matmul(values) + ridge * torch.eye(37, dtype=torch.float64),
            values.t().matmul(targets),
        )
        self.assertTrue(torch.allclose(stable, direct, atol=1e-9, rtol=1e-9))
        rhs = torch.randn(37, 8, generator=generator, dtype=torch.float64)
        stable_inverse = ranpac_low_rank_inverse_apply(system, rhs, ridge)
        direct_inverse = torch.linalg.solve(
            values.t().matmul(values) + ridge * torch.eye(37, dtype=torch.float64),
            rhs,
        )
        self.assertTrue(
            torch.allclose(stable_inverse, direct_inverse, atol=1e-9, rtol=1e-9)
        )
        residual = ranpac_normal_equation_relative_residual(
            projected, labels, stable, ridge, 6
        )
        self.assertLess(residual, 1e-10)

    def test_fecam_metric_is_symmetric_positive_definite(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        features = F.normalize(torch.randn(8, 12, generator=generator), dim=-1)
        inverse = fecam_inverse_covariance(features)
        self.assertTrue(torch.allclose(inverse, inverse.t(), atol=1e-5))
        self.assertGreater(float(torch.linalg.eigvalsh(inverse).min().item()), 0.0)

    def test_hycal_low_rank_distance_matches_dense_inverse(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(411)
        support = F.normalize(torch.randn(5, 12, generator=generator), dim=-1)
        queries = F.normalize(torch.randn(19, 12, generator=generator), dim=-1)
        statistics = hycal_class_statistics(support)
        actual = hycal_squared_mahalanobis_chunked(queries, statistics, 4)
        centered = support - support.mean(0, keepdim=True)
        covariance = centered.t().matmul(centered) / 4.0
        regularized = 0.9999 * covariance + 1e-4 * torch.eye(12)
        inverse = torch.linalg.inv(regularized)
        difference = queries - support.mean(0, keepdim=True)
        expected = torch.einsum("qd,de,qe->q", difference, inverse, difference)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-2, rtol=2e-4))

    def test_hycal_is_query_batch_and_order_invariant(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(812)
        class_features = [
            F.normalize(torch.randn(5, 12, generator=generator), dim=-1)
            for _ in range(4)
        ]
        statistics = tuple(hycal_class_statistics(x) for x in class_features)
        whole = hycal_scores_chunked(self.queries, statistics, 1000)
        chunked = hycal_scores_chunked(self.queries, statistics, 3)
        self.assertTrue(torch.equal(whole.argmax(1), chunked.argmax(1)))
        order = torch.randperm(self.queries.shape[0], generator=generator)
        reordered = hycal_scores_chunked(self.queries[order], statistics, 5)
        self.assertTrue(torch.equal(whole.argmax(1)[order], reordered.argmax(1)))

    def test_randumb_rbf_formula_matches_sklearn(self) -> None:
        from sklearn.kernel_approximation import RBFSampler

        generator = torch.Generator(device="cpu")
        generator.manual_seed(501)
        features = torch.randn(17, 6, generator=generator)
        gamma = 0.7
        weights, offset = make_rbf_projection(6, 19, gamma, 1993, features.device)
        actual = rbf_transform(features, weights, offset).numpy()
        reference = RBFSampler(gamma=gamma, n_components=19, random_state=1993)
        expected = reference.fit_transform(features.numpy())
        self.assertTrue(
            np.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        )


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        self.by_class = {
            class_id: F.normalize(torch.randn(30, 16, generator=generator), dim=-1)
            for class_id in range(6)
        }
        self.system = covariance_eigensystem(self.by_class, sorted(self.by_class))
        self.queries = F.normalize(torch.randn(53, 16, generator=generator), dim=-1)
        self.prototypes = F.normalize(torch.randn(9, 16, generator=generator), dim=-1)

    def test_isotropic_endpoint_matches_cosine_predictions(self) -> None:
        cosine = cosine_scores(self.queries, self.prototypes).argmax(1)
        isotropic = metric_scores(
            self.queries,
            self.prototypes,
            self.system,
            shrinkage=1.0,
            ridge_gamma=RIDGE_GAMMA,
        ).argmax(1)
        self.assertTrue(torch.equal(cosine, isotropic))

    def test_inductive_predictions_are_batch_invariant(self) -> None:
        full = metric_scores(
            self.queries, self.prototypes, self.system, 0.25, RIDGE_GAMMA, query_chunk=1024
        ).argmax(1)
        partitioned = metric_scores(
            self.queries, self.prototypes, self.system, 0.25, RIDGE_GAMMA, query_chunk=7
        ).argmax(1)
        self.assertTrue(torch.equal(full, partitioned))

    def test_inductive_predictions_are_order_invariant(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8)
        order = torch.randperm(self.queries.shape[0], generator=generator)
        inverse = torch.argsort(order)
        reference = metric_scores(
            self.queries, self.prototypes, self.system, 0.5, RIDGE_GAMMA
        ).argmax(1)
        shuffled = metric_scores(
            self.queries[order], self.prototypes, self.system, 0.5, RIDGE_GAMMA
        ).argmax(1)[inverse]
        self.assertTrue(torch.equal(reference, shuffled))

    def test_cached_covariance_basis_scores_match_direct_scoring(self) -> None:
        direct = metric_scores(
            self.queries,
            self.prototypes,
            self.system,
            0.25,
            RIDGE_GAMMA,
            query_chunk=7,
        )
        cached = metric_scores_rotated(
            rotate_to_covariance_basis(self.queries, self.system),
            rotate_to_covariance_basis(self.prototypes, self.system),
            self.system,
            0.25,
            RIDGE_GAMMA,
            query_chunk=7,
        )
        self.assertTrue(torch.equal(direct, cached))

    def test_analytic_penalty_scales_as_inverse_shot_count(self) -> None:
        penalty_five = metric_trace_penalty(self.system, 0.25, RIDGE_GAMMA, 5)
        penalty_ten = metric_trace_penalty(self.system, 0.25, RIDGE_GAMMA, 10)
        self.assertAlmostEqual(penalty_five, 2.0 * penalty_ten, places=6)


class TransductiveOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(733)
        self.support = F.normalize(
            torch.randn(15, 10, generator=generator), dim=1
        )
        self.support_labels = torch.arange(15) % 3
        self.query = F.normalize(
            torch.randn(23, 10, generator=generator), dim=1
        )
        self.prototypes = torch.stack(
            [
                F.normalize(
                    self.support[self.support_labels == class_id].mean(0),
                    dim=0,
                )
                for class_id in range(3)
            ],
            0,
        )

    def test_chunked_sparse_knn_matches_dense_topk(self) -> None:
        graph, metadata = chunked_sparse_cosine_knn(
            self.query, neighbors=4, chunk_size=5, symmetric=False
        )
        dense = self.query.matmul(self.query.t())
        dense.fill_diagonal_(-torch.inf)
        expected = torch.topk(dense, k=4, dim=1).indices
        index = graph.coalesce().indices()
        actual = {
            row: set(index[1, index[0] == row].tolist())
            for row in range(self.query.shape[0])
        }
        for row in range(self.query.shape[0]):
            self.assertEqual(actual[row], set(expected[row].tolist()))
        self.assertLessEqual(
            metadata["max_similarity_block_elements"],
            5 * self.query.shape[0],
        )

    def test_transductive_apis_do_not_accept_query_or_test_labels(self) -> None:
        for function in (
            tim_gd,
            pt_map,
            protolp_paper,
            protolp_unconstrained,
            protolp_uniform,
            query_prototype_rectification,
            sparse_graph_repair,
            laplacian_shot,
        ):
            names = {name.lower() for name in inspect.signature(function).parameters}
            self.assertNotIn("query_labels", names)
            self.assertFalse(any(name.startswith("test") for name in names))
        guard_names = {
            name.lower()
            for name in inspect.signature(base_guarded_prototypes).parameters
        }
        self.assertFalse(any("label" in name for name in guard_names))
        self.assertFalse(any(name.startswith("test") for name in guard_names))

    def test_qpr_base_guard_rejects_a_base_capturing_proposal(self) -> None:
        by_class = {
            0: F.normalize(
                torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]), dim=1
            ),
            1: F.normalize(
                torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 1.0]]), dim=1
            ),
        }
        covariance = covariance_eigensystem(by_class, (0, 1))
        rehearsal = torch.cat([by_class[0], by_class[1]], dim=0)
        original = F.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=1
        )
        proposed = original.clone()
        proposed[2] = torch.tensor([1.0, 0.0])
        guarded = base_guarded_prototypes(
            original,
            proposed,
            2,
            rehearsal,
            covariance,
            shrinkage=1.0,
            ridge_gamma=RIDGE_GAMMA,
            separation_ratio=1.0,
        )
        before = metric_scores(
            rehearsal, original, covariance, 1.0, RIDGE_GAMMA
        ).argmax(1)
        after = metric_scores(
            rehearsal, guarded.prototypes, covariance, 1.0, RIDGE_GAMMA
        ).argmax(1)
        self.assertTrue(torch.equal(guarded.prototypes[:2], original[:2]))
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(guarded.metadata["base_rehearsal_prediction_changes"], 0)
        self.assertLess(guarded.metadata["selected_steps"]["2"], 1.0)

    def test_qpr_base_guard_accepts_a_safe_full_proposal(self) -> None:
        by_class = {
            0: F.normalize(
                torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]), dim=1
            ),
            1: F.normalize(
                torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 1.0]]), dim=1
            ),
        }
        covariance = covariance_eigensystem(by_class, (0, 1))
        rehearsal = torch.cat([by_class[0], by_class[1]], dim=0)
        original = F.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=1
        )
        proposed = original.clone()
        proposed[2] = F.normalize(torch.tensor([-0.8, -0.6]), dim=0)
        guarded = base_guarded_prototypes(
            original,
            proposed,
            2,
            rehearsal,
            covariance,
            shrinkage=1.0,
            ridge_gamma=RIDGE_GAMMA,
            separation_ratio=1.0,
        )
        self.assertTrue(torch.allclose(guarded.prototypes[2], proposed[2]))
        self.assertEqual(guarded.metadata["selected_steps"]["2"], 1.0)
        self.assertEqual(guarded.metadata["full_step_class_count"], 1)

    def test_qpr_base_guard_zero_step_is_always_valid(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(991)
        by_class = {
            class_id: F.normalize(
                torch.randn(6, 5, generator=generator), dim=1
            )
            for class_id in range(2)
        }
        covariance = covariance_eigensystem(by_class, (0, 1))
        original = F.normalize(
            torch.randn(3, 5, generator=generator), dim=1
        )
        proposed = original.clone()
        proposed[2] = original[0]
        guarded = base_guarded_prototypes(
            original,
            proposed,
            2,
            torch.cat([by_class[0], by_class[1]], dim=0),
            covariance,
            shrinkage=0.0,
            ridge_gamma=RIDGE_GAMMA,
            separation_ratio=1.0,
            step_grid=(0.0,),
        )
        self.assertTrue(torch.equal(guarded.prototypes, original))
        self.assertEqual(guarded.metadata["selected_steps"]["2"], 0.0)

    def test_qpr_matches_conservative_reference_and_freezes_base(self) -> None:
        prototypes = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            ),
            dim=1,
        )
        query = F.normalize(
            torch.tensor(
                [
                    [-1.0, 0.1],
                    [-0.9, -0.1],
                    [-1.0, 0.0],
                    [0.1, -1.0],
                    [-0.1, -0.9],
                    [0.0, -1.0],
                ]
            ),
            dim=1,
        )
        scores = 20.0 * query.matmul(prototypes.t())
        output = query_prototype_rectification(
            query,
            prototypes,
            scores,
            num_base_classes=2,
            alpha=0.75,
            mass_threshold=0.3,
            min_count=3,
        )
        probabilities = torch.softmax(scores, dim=1)
        expected = prototypes.clone()
        for class_id in (2, 3):
            selected = probabilities.argmax(1) == class_id
            weights = probabilities[selected, class_id]
            weights = weights / weights.sum()
            query_mean = F.normalize(
                weights.unsqueeze(0).matmul(query[selected]), dim=1
            ).squeeze(0)
            expected[class_id] = F.normalize(
                (0.25 * prototypes[class_id] + 0.75 * query_mean).unsqueeze(0),
                dim=1,
            ).squeeze(0)
        self.assertTrue(torch.equal(output.prototypes[:2], prototypes[:2]))
        self.assertTrue(
            torch.allclose(output.prototypes, expected, atol=1e-7, rtol=1e-6)
        )
        self.assertFalse(output.metadata["updates_base_prototypes"])
        self.assertEqual(output.metadata["updated_class_count"], 2)

    def test_qpr_minimum_count_can_make_the_operator_a_noop(self) -> None:
        scores = 10.0 * self.query.matmul(self.prototypes.t())
        output = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=self.query.shape[0] + 1,
        )
        self.assertTrue(torch.equal(output.prototypes, self.prototypes))
        self.assertEqual(output.metadata["updated_class_count"], 0)

    def test_qpr_base_defections_account_for_every_captured_base_example(
        self,
    ) -> None:
        from fscil.run_qpr_ablation import _diagnostics

        num_base = 2
        class_total = 4
        labels = torch.tensor([0, 0, 1, 1, 2, 3])
        baseline_prediction = torch.tensor([0, 0, 1, 1, 2, 3])
        # Two correct base examples defect to novel prototypes 2 and 3.
        prediction = torch.tensor([0, 2, 1, 3, 2, 3])
        scores = torch.zeros(6, class_total)
        scores[torch.arange(6), prediction] = 5.0
        prototypes = F.normalize(torch.randn(class_total, 4), dim=1)
        diagnostics = _diagnostics(
            baseline_prediction=baseline_prediction,
            prediction=prediction,
            labels=labels,
            scores=scores,
            original=prototypes,
            refined=prototypes,
            full_reference=prototypes,
            num_base=num_base,
            config={
                "logit_scale": 1.0,
                "mass_threshold": 0.3,
                "require_predicted_novel": True,
            },
            metadata={
                "updated_class_ids": [],
                "eligible_class_count": 0,
                "updated_class_count": 0,
            },
        )
        captured = int(diagnostics["base_correct_captured_by_novel"])
        self.assertEqual(captured, 2)
        # The pairwise ledger must be a complete decomposition of that count.
        self.assertEqual(
            sum(int(row["count"]) for row in diagnostics["base_defections"]),
            captured,
        )
        self.assertEqual(
            sum(int(row["captured"]) for row in diagnostics["base_loss_per_class"]),
            captured,
        )
        self.assertEqual(
            sum(
                int(row["captured"])
                for row in diagnostics["novel_capture_per_class"]
            ),
            captured,
        )
        pairs = {
            (int(row["base_class"]), int(row["novel_class"]))
            for row in diagnostics["base_defections"]
        }
        self.assertEqual(pairs, {(0, 2), (1, 3)})
        # Every recorded defection must run from a base class to a novel one.
        for row in diagnostics["base_defections"]:
            self.assertLess(int(row["base_class"]), num_base)
            self.assertGreaterEqual(int(row["novel_class"]), num_base)

    def test_qpr_all_novel_escape_hatch_cannot_change_the_reported_path(
        self,
    ) -> None:
        # The diagnostic flag must relax only the base-count precondition, so
        # any configuration with a base block is bitwise unchanged by it.
        scores = 10.0 * self.query.matmul(self.prototypes.t())
        for base_count in (1, 2):
            reported = query_prototype_rectification(
                self.query, self.prototypes, scores, num_base_classes=base_count
            )
            relaxed = query_prototype_rectification(
                self.query,
                self.prototypes,
                scores,
                num_base_classes=base_count,
                allow_all_novel=True,
            )
            self.assertTrue(
                torch.equal(reported.prototypes, relaxed.prototypes),
                "allow_all_novel changed prototypes at base_count={}".format(
                    base_count
                ),
            )
            self.assertTrue(torch.equal(reported.scores, relaxed.scores))
            self.assertEqual(
                reported.metadata["updated_class_ids"],
                relaxed.metadata["updated_class_ids"],
            )
        # The default path still refuses an all-novel cohort outright.
        with self.assertRaises(ValueError):
            query_prototype_rectification(
                self.query, self.prototypes, scores, num_base_classes=0
            )

    def test_qpr_all_novel_gates_are_vacuous_without_a_base_block(self) -> None:
        # Documents why the native 5-way row is a degenerate variant: with no
        # base classes the mass and predicted-novel gates admit every query.
        scores = 10.0 * self.query.matmul(self.prototypes.t())
        output = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=0,
            # Just below one, because the float32 softmax row sum can land a
            # few ulps under unity; an exact 1.0 threshold would filter on
            # rounding rather than on evidence.
            mass_threshold=0.99,
            min_count=1,
            allow_all_novel=True,
        )
        self.assertFalse(output.metadata["gates_active"])
        self.assertTrue(output.metadata["allow_all_novel"])
        # A 0.99 mass threshold is near-prohibitive when a base block exists;
        # with none, novel mass is ~1 for every row and nothing is filtered.
        self.assertEqual(
            output.metadata["gated_query_count"], int(self.query.shape[0])
        )
        self.assertFalse(output.metadata["uses_query_labels"])

    def test_qpr_preserves_nonunit_prototype_radius(self) -> None:
        prototypes = torch.tensor(
            [[0.8, 0.0], [0.0, 0.7], [-0.4, 0.0], [0.0, -0.5]]
        )
        query = F.normalize(
            torch.tensor(
                [
                    [-1.0, 0.1],
                    [-1.0, -0.1],
                    [-1.0, 0.0],
                    [0.1, -1.0],
                    [-0.1, -1.0],
                    [0.0, -1.0],
                ]
            ),
            dim=1,
        )
        scores = 20.0 * query.matmul(F.normalize(prototypes, dim=1).t())
        output = query_prototype_rectification(
            query,
            prototypes,
            scores,
            num_base_classes=2,
            alpha=0.75,
            mass_threshold=0.3,
            min_count=3,
        )
        self.assertTrue(torch.equal(output.prototypes[:2], prototypes[:2]))
        self.assertTrue(
            torch.allclose(
                output.prototypes.norm(dim=1),
                prototypes.norm(dim=1),
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(output.metadata["preserves_input_prototype_radius"])

    def test_qpr_within_novel_margin_rejects_ambiguous_assignments(self) -> None:
        prototypes = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
            dim=1,
        )
        query = F.normalize(
            torch.tensor(
                [
                    [-1.0, 0.0, -1.0],
                    [-1.0, 0.01, -1.0],
                    [-1.0, -0.01, -1.0],
                    [-0.1, 0.0, -1.0],
                    [0.0, -0.1, -1.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
            dim=1,
        )
        scores = 8.0 * query.matmul(prototypes.t())
        ungated = query_prototype_rectification(
            query,
            prototypes,
            scores,
            num_base_classes=2,
            alpha=0.75,
            min_count=3,
            novel_margin_threshold=0.0,
        )
        gated = query_prototype_rectification(
            query,
            prototypes,
            scores,
            num_base_classes=2,
            alpha=0.75,
            min_count=3,
            novel_margin_threshold=0.5,
        )
        self.assertEqual(ungated.metadata["updated_class_count"], 2)
        self.assertEqual(gated.metadata["updated_class_count"], 1)
        self.assertEqual(gated.metadata["novel_margin_threshold"], 0.5)
        self.assertTrue(torch.equal(gated.prototypes[:2], prototypes[:2]))
        self.assertTrue(torch.equal(gated.prototypes[2], prototypes[2]))

    def test_zero_qpr_margin_is_backward_compatible(self) -> None:
        scores = 13.0 * self.query.matmul(self.prototypes.t())
        default = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=2,
        )
        explicit = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=2,
            novel_margin_threshold=0.0,
        )
        self.assertTrue(torch.equal(default.prototypes, explicit.prototypes))
        self.assertTrue(torch.equal(default.scores, explicit.scores))

    def test_qpr_component_defaults_are_exactly_backward_compatible(self) -> None:
        scores = 13.0 * self.query.matmul(self.prototypes.t())
        default = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=2,
        )
        explicit = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=2,
            posterior_weighting=True,
            preserve_radius=True,
            require_predicted_novel=True,
        )
        self.assertTrue(torch.equal(default.prototypes, explicit.prototypes))
        self.assertTrue(torch.equal(default.scores, explicit.scores))

    def test_qpr_component_controls_preserve_base_and_report_scope(self) -> None:
        prototypes = self.prototypes.clone()
        prototypes[2:] *= 0.4
        scores = 13.0 * self.query.matmul(F.normalize(prototypes, dim=1).t())
        output = query_prototype_rectification(
            self.query,
            prototypes,
            scores,
            num_base_classes=2,
            min_count=1,
            posterior_weighting=False,
            preserve_radius=False,
            require_predicted_novel=False,
        )
        self.assertTrue(torch.equal(output.prototypes[:2], prototypes[:2]))
        updated = output.metadata["updated_class_ids"]
        self.assertGreater(len(updated), 0)
        self.assertTrue(
            torch.allclose(
                output.prototypes[updated].norm(dim=1),
                torch.ones(len(updated)),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertFalse(output.metadata["posterior_weighting"])
        self.assertFalse(output.metadata["preserves_input_prototype_radius"])
        self.assertFalse(output.metadata["require_predicted_novel"])

    def test_qpr_is_query_order_equivariant_for_a_fixed_cohort(self) -> None:
        scores = 13.0 * self.query.matmul(self.prototypes.t())
        reference = query_prototype_rectification(
            self.query,
            self.prototypes,
            scores,
            num_base_classes=2,
            min_count=2,
        )
        order = torch.randperm(
            self.query.shape[0], generator=torch.Generator().manual_seed(814)
        )
        inverse = torch.argsort(order)
        shuffled = query_prototype_rectification(
            self.query[order],
            self.prototypes,
            scores[order],
            num_base_classes=2,
            min_count=2,
        )
        self.assertTrue(
            torch.allclose(
                reference.prototypes,
                shuffled.prototypes,
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                reference.scores,
                shuffled.scores[inverse],
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertEqual(
            reference.metadata["admitted_counts"],
            shuffled.metadata["admitted_counts"],
        )

    def test_cohort_modes_are_deterministic_and_complete(self) -> None:
        from fscil.run_transductive import (
            _fixed_balanced_subset,
            _fixed_imbalanced_subset,
            _mode_batches,
        )

        query = torch.arange(80, dtype=torch.float32).reshape(40, 2)
        labels = torch.arange(4).repeat_interleave(10)
        first = _fixed_balanced_subset(query, labels, 4, 3, namespace_seed=7)
        second = _fixed_balanced_subset(query, labels, 4, 3, namespace_seed=7)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[0].shape[0], 12)
        self.assertEqual(first[2].tolist(), [3.0] * 4)
        imbalanced = _fixed_imbalanced_subset(
            query,
            labels,
            4,
            torch.full((4,), 10.0),
            imbalance_ratio=10.0,
            namespace_seed=3,
        )
        self.assertEqual(imbalanced[3]["imbalance_ratio"], 10.0)
        self.assertGreater(max(imbalanced[3]["counts"]), min(imbalanced[3]["counts"]))
        batches, metadata = _mode_batches("causal5x2", 11, 123, query.device)
        covered = torch.cat([context[retained] for context, retained in batches])
        self.assertTrue(torch.equal(covered.sort().values, torch.arange(11)))
        self.assertEqual(torch.unique(covered).numel(), 11)
        self.assertEqual(metadata["window"], 5)
        self.assertEqual(metadata["stride"], 2)

    def test_shot_count_uses_standard_five_shot_and_nested_other_shots(self) -> None:
        from fscil.run_transductive import _support_summary

        features = F.normalize(torch.arange(50, dtype=torch.float32).reshape(25, 2) + 1.0, dim=1)
        labels = torch.tensor([0, 0] + [1] * 23)
        standard = torch.tensor([[[10, 11, 12, 13, 14]]])
        nested = torch.tensor([[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]])
        data = SimpleNamespace(
            num_base=1,
            train_features=features,
            train_labels=labels,
            standard_support_abs=standard,
            nested_support_abs=nested,
        )
        by_class = {0: features[:2], 1: features[2:]}
        five, five_labels, _, _ = _support_summary(
            data, by_class, 0, 2, "normalized", shot_count=5
        )
        two, two_labels, _, _ = _support_summary(
            data, by_class, 0, 2, "normalized", shot_count=2
        )
        self.assertTrue(torch.equal(five[-5:], features[standard.reshape(-1)]))
        self.assertTrue(torch.equal(two[-2:], features[nested[0, 0, :2]]))
        self.assertEqual(int((five_labels == 0).sum()), 5)
        self.assertEqual(int((two_labels == 0).sum()), 2)

    def test_ptmap_transport_respects_declared_mass(self) -> None:
        mass = torch.tensor([2.0, 5.0, 3.0])
        output = pt_map(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            iterations=2,
            class_mass=mass,
            sinkhorn_iterations=200,
        )
        plan = output.scores.exp()
        expected = mass * (self.query.shape[0] / mass.sum())
        self.assertTrue(
            torch.allclose(plan.sum(0), expected, atol=3e-3, rtol=3e-3)
        )
        self.assertFalse(output.metadata["power_transform_applied"])
        self.assertEqual(
            output.metadata["power_transform_reason"], "signed_CLIP_embeddings"
        )

    def test_known_marginal_sinkhorn_respects_protocol_mass(self) -> None:
        scores = self.query.matmul(self.prototypes.t())
        mass = torch.tensor([2.0, 5.0, 3.0])
        log_plan = marginal_constrained_scores(
            scores, mass, temperature=0.2, iterations=200
        )
        plan = log_plan.exp()
        expected = mass * (scores.shape[0] / mass.sum())
        self.assertTrue(
            torch.allclose(plan.sum(0), expected, atol=3e-3, rtol=3e-3)
        )
        scaled = marginal_constrained_scores(
            scores, 7.0 * mass, temperature=0.2, iterations=200
        )
        self.assertTrue(torch.allclose(log_plan, scaled, atol=2e-5, rtol=2e-5))

    def test_protolp_paper_is_finite_and_order_equivariant(self) -> None:
        reference = protolp_paper(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            iterations=3,
        )
        order = torch.randperm(self.query.shape[0], generator=torch.Generator().manual_seed(9))
        inverse = torch.argsort(order)
        shuffled = protolp_paper(
            self.support,
            self.support_labels,
            self.query[order],
            num_classes=3,
            iterations=3,
        )
        self.assertTrue(torch.isfinite(reference.scores).all())
        self.assertTrue(
            torch.allclose(
                reference.scores,
                shuffled.scores[inverse],
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_protolp_low_rank_graph_matches_dense_equation(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(938)
        assignment = torch.softmax(
            torch.randn(38, 3, generator=generator), dim=1
        )
        support_targets = F.one_hot(torch.arange(15) % 3, 3).float()
        assignment[:15] = support_targets
        actual = _protolp_paper_graph_scores(
            assignment, support_targets, 15, 0.7
        )
        delta = assignment.sum(0)
        sample_graph = assignment.matmul(
            torch.diag(delta.reciprocal())
        ).matmul(assignment.t())
        identity = torch.eye(assignment.shape[0])
        dense_regularizer = assignment.t().matmul(
            identity - sample_graph
        ).matmul(assignment)
        labelled = assignment[:15]
        system = labelled.t().matmul(labelled) + 0.7 * dense_regularizer
        system = 0.5 * (system + system.t()) + 1e-5 * torch.eye(3)
        coefficients = torch.linalg.solve(
            system, labelled.t().matmul(support_targets)
        )
        expected = assignment.matmul(coefficients).clamp_min(0.0)
        expected = expected / expected.sum(1, keepdim=True).clamp_min(1e-8)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_protolp_paper_one_step_matches_equations(self) -> None:
        prototypes = support_prototypes(
            self.support, self.support_labels, 3, normalize=True
        )
        all_values = torch.cat([self.support, self.query], 0)
        assignment = torch.softmax(
            -squared_euclidean(all_values, prototypes), dim=1
        )
        targets = F.one_hot(self.support_labels, 3).float()
        assignment[: self.support.shape[0]] = targets
        propagated = _protolp_paper_graph_scores(
            assignment, targets, self.support.shape[0], 0.7
        )
        estimate = propagated.t().matmul(all_values)
        estimate = estimate / propagated.sum(0)[:, None]
        expected_prototypes = prototypes + 0.2 * (estimate - prototypes)
        actual = protolp_paper(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            graph_regularization=0.7,
            iterations=1,
        )
        self.assertTrue(
            torch.allclose(
                actual.scores.exp(),
                propagated[self.support.shape[0] :],
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                actual.prototypes,
                expected_prototypes,
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertEqual(
            actual.metadata["support_assignment"],
            "one_hot_labels_per_Eq_4",
        )

    def test_protolp_assignment_clamps_only_support_rows(self) -> None:
        prototypes = support_prototypes(
            self.support, self.support_labels, 3, normalize=True
        )
        all_values = torch.cat([self.support, self.query], 0)
        targets = F.one_hot(self.support_labels, 3).float()
        assignment = _protolp_assignment(
            all_values,
            prototypes,
            targets,
            self.support.shape[0],
            assignment_scale=5.0,
        )
        self.assertTrue(
            torch.equal(assignment[: self.support.shape[0]], targets)
        )
        query_assignment = assignment[self.support.shape[0] :]
        self.assertTrue(
            torch.allclose(
                query_assignment.sum(1),
                torch.ones(query_assignment.shape[0]),
                atol=1e-6,
            )
        )
        self.assertTrue(bool(((query_assignment > 0) & (query_assignment < 1)).all()))

    def test_protolp_uniform_projection_respects_uniform_mass(self) -> None:
        output = protolp_uniform(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            iterations=2,
            sinkhorn_iterations=200,
        )
        plan = output.scores.exp()
        expected = torch.full((3,), self.query.shape[0] / 3.0)
        self.assertTrue(torch.allclose(plan.sum(0), expected, atol=3e-3, rtol=3e-3))
        self.assertEqual(
            output.metadata["marginal_assumption"], "uniform_equipartition"
        )

    def test_sparse_graph_and_laplacian_are_chunk_invariant(self) -> None:
        graph_small = sparse_graph_repair(
            self.prototypes, self.query, neighbors=4, chunk_size=5
        )
        graph_large = sparse_graph_repair(
            self.prototypes, self.query, neighbors=4, chunk_size=100
        )
        self.assertTrue(
            torch.allclose(
                graph_small.scores, graph_large.scores, atol=2e-6, rtol=2e-6
            )
        )
        self.assertFalse(graph_small.metadata["dense_query_affinity_constructed"])
        lap_small = laplacian_shot(
            self.prototypes, self.query, neighbors=3, chunk_size=5
        )
        lap_large = laplacian_shot(
            self.prototypes, self.query, neighbors=3, chunk_size=100
        )
        self.assertTrue(
            torch.allclose(
                lap_small.scores, lap_large.scores, atol=2e-6, rtol=2e-6
            )
        )
        self.assertFalse(lap_small.metadata["dense_query_affinity_constructed"])

    def test_full_laplacian_shot_is_chunk_invariant_and_complete(self) -> None:
        train_mean = torch.linspace(-0.03, 0.04, self.support.shape[1])
        small = laplacian_shot_full(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            train_mean=train_mean,
            knn=4,
            chunk_size=5,
        )
        large = laplacian_shot_full(
            self.support,
            self.support_labels,
            self.query,
            num_classes=3,
            train_mean=train_mean,
            knn=4,
            chunk_size=100,
        )
        self.assertTrue(
            torch.allclose(small.scores, large.scores, atol=2e-6, rtol=2e-6)
        )
        self.assertTrue(small.metadata["includes_cl2n"])
        self.assertTrue(small.metadata["includes_prototype_rectification"])
        self.assertEqual(small.metadata["knn_including_self"], 4)
        self.assertFalse(small.metadata["dense_query_affinity_constructed"])

    def test_euclidean_knn_matches_literal_dense_neighbors(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        values = torch.randn(17, 6, generator=generator)
        sparse, metadata = chunked_sparse_euclidean_knn(
            values, neighbors=3, chunk_size=4
        )
        distance = torch.cdist(values, values).square()
        distance.fill_diagonal_(torch.inf)
        expected_columns = torch.topk(
            distance, k=3, dim=1, largest=False
        ).indices
        actual = sparse.to_dense().bool()
        expected = torch.zeros_like(actual)
        expected.scatter_(1, expected_columns, True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(metadata["neighbors_excluding_self"], 3)

    def test_laplacian_sparse_bound_matches_literal_dense_update(self) -> None:
        generator = torch.Generator().manual_seed(717)
        unary = torch.rand(13, 4, generator=generator)
        rows = torch.arange(13).repeat_interleave(2)
        columns = torch.stack(
            [(torch.arange(13) + 1) % 13, (torch.arange(13) + 3) % 13], 1
        ).reshape(-1)
        affinity = torch.sparse_coo_tensor(
            torch.stack([rows, columns]), torch.ones(rows.numel()), (13, 13)
        ).coalesce()
        actual, metadata = _laplacian_bound_update(
            unary, affinity, regularization=0.7, iterations=20
        )
        dense = affinity.to_dense()
        expected = torch.softmax(-unary, dim=1)
        old_energy = torch.tensor(float("inf"))
        completed = 0
        for iteration in range(20):
            neighbor = dense.matmul(expected)
            expected = torch.softmax(-unary + 0.7 * neighbor, dim=1)
            energy = (
                expected * torch.log(expected.clamp_min(1e-20))
                + unary * expected
                - 0.7 * neighbor * expected
            ).sum()
            completed = iteration + 1
            if iteration > 1 and bool(
                torch.abs(energy - old_energy) <= 1e-6 * torch.abs(old_energy)
            ):
                break
            old_energy = energy.clone()
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))
        self.assertEqual(metadata["iterations_completed"], completed)


if __name__ == "__main__":
    unittest.main()
