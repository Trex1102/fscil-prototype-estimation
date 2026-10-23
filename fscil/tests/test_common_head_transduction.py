"""Independent checks for the common-head comparator adaptations."""

import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from fscil.common_head_transduction import common_head_method, head_transform, initial_scores
from fscil.run_common_head_benchmark import ALL_METHODS, HEADS
from fscil.geometry import CovarianceEigensystem, covariance_eigensystem, metric_scores
from fscil.qpr_self_exclusion import qpr_same_and_leave_one_out_scores
from fscil.transductive_methods import (
    laplacian_shot, laplacian_shot_full, protolp_paper, pt_map,
    query_prototype_rectification, support_prototypes, tim_gd,
)


class CommonHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)
        self.support = F.normalize(torch.randn(15, 8), dim=1)
        self.labels = torch.arange(3).repeat_interleave(5)
        self.query = F.normalize(torch.randn(18, 8), dim=1)
        self.p = support_prototypes(self.support, self.labels, 3)
        self.parameters = {
            "selected": {"protolp_paper": {"values": [0.1, 1.0]},
                         "ptmap": {"values": [5.0]},
                         "laplacian": {"values": [3, 0.5, 1]},
                         "laplacian_core": {"values": [3, 0.5]}},
            "tim": {"temperature": 15., "iterations": 3,
                    "learning_rate": 1e-4, "loss_weights": [.1, 1., .1]},
        }

    def run_method(self, name, p=None):
        return common_head_method(name, self.support, self.labels, self.query,
                                  self.p if p is None else p, self.parameters)

    def test_whitening_preserves_parent_distance_ratios_and_predictions(self):
        covariance = covariance_eigensystem(
            {c: self.support[self.labels == c] for c in range(3)}, range(3))
        for lam in (0., .25, .75, 1.):
            p = self.p * torch.tensor([.7, .8, .9])[:, None]
            a = head_transform(covariance, lam, 1.)
            original = metric_scores(self.query, p, covariance, lam, 1.)
            adapted = initial_scores(self.query @ a, p @ a)
            self.assertTrue(torch.equal(original.argmax(1), adapted.argmax(1)))
            ratio = original[0, 0] / adapted[0, 0]
            self.assertTrue(torch.allclose(original, adapted * ratio, atol=2e-4, rtol=1e-5))

    def test_adaptations_match_native_equations_at_same_initialization(self):
        raw = support_prototypes(self.support, self.labels, 3, normalize=False)
        tim = tim_gd(self.support, self.labels, self.query, 3, iterations=3)
        lp = protolp_paper(self.support, self.labels, self.query, 3,
                           graph_regularization=.1, assignment_scale=1.)
        ot = pt_map(self.support, self.labels, self.query, 3, assignment_scale=5.)
        core = laplacian_shot(self.p, self.query, neighbors=3, regularization=.5)
        rect = laplacian_shot_full(
            self.support, self.labels, self.query, 3, train_mean=torch.zeros(8),
            knn=3, regularization=.5)
        for name, initial, expected in (
            ("tim_gd", raw, tim), ("protolp_paper", self.p, lp),
            ("ptmap_uniform", self.p, ot), ("laplacian_shot_core", self.p, core),
            ("laplacian_shot", raw, rect),
        ):
            actual = self.run_method(name, initial)
            self.assertTrue(torch.allclose(actual.scores, expected.scores, atol=2e-5, rtol=2e-5), name)

    def test_zero_step_tim_keeps_supplied_head_including_radius(self):
        self.parameters["tim"]["iterations"] = 0
        p = self.p * torch.tensor([.2, .5, .9])[:, None]
        actual = self.run_method("tim_gd", p)
        self.assertTrue(torch.equal(actual.prototypes, p))
        self.assertTrue(torch.allclose(actual.scores, 15. * initial_scores(self.query, p), atol=2e-6))

    def test_full_laplacian_retains_native_nonzero_mean_preprocessing(self):
        centers = F.normalize(torch.randn(3, 8), dim=1)
        support = centers.repeat_interleave(5, dim=0)
        train_mean = torch.linspace(.05, .3, 8)
        expected = laplacian_shot_full(
            support, self.labels, self.query, 3, train_mean=train_mean,
            knn=3, regularization=.5)
        actual = common_head_method(
            "laplacian_shot", support, self.labels, self.query, centers,
            self.parameters, train_mean=train_mean)
        self.assertTrue(torch.allclose(actual.scores, expected.scores, atol=2e-5, rtol=2e-5))
        self.assertTrue(actual.metadata["native_cl2n_retained"])

    def test_cosine_self_exclusion_matches_literal_leave_one_out(self):
        eye = CovarianceEigensystem(torch.ones(8), torch.eye(8), torch.tensor(.5), 1)
        baseline = self.query @ self.p.t()
        params = dict(alpha=.75, mass_threshold=0., min_count=1, logit_scale=1.)
        output = qpr_same_and_leave_one_out_scores(
            self.query, self.p, baseline, eye, 1., 1., 1, **params)
        for i in range(self.query.shape[0]):
            keep = torch.arange(self.query.shape[0]) != i
            proposal = query_prototype_rectification(
                self.query[keep], self.p, baseline[keep], 1, **params)
            expected = initial_scores(self.query[i:i+1], proposal.prototypes)[0]
            self.assertTrue(torch.allclose(output.leave_one_out_scores[i], expected, atol=2e-6))



if __name__ == "__main__":
    unittest.main()
