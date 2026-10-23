from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import reproduce
import reproduce_dtd
from main import setup_cfg
from fscil.extract_clip_features import import_openai_clip


class ReproductionPackagingTest(unittest.TestCase):


    def test_included_configs_preserve_the_paper_schedules(self):
        for name, base, increment in (("cifar100", 60, 5), ("miniimagenet", 60, 5), ("cub200_bimc_dino_fusion", 100, 10)):
            cfg = setup_cfg(str(reproduce.CODE / "configs/datasets" / (name + ".yaml")),
                            str(reproduce.CODE / "configs/trainers/bimc_dino_fusion.yaml"))
            self.assertEqual(cfg.DATASET.NUM_INIT_CLS, base)
            self.assertEqual(cfg.DATASET.NUM_INC_CLS, increment)
            self.assertEqual(cfg.DATASET.NUM_INC_SHOT, 5)

    def test_clip_import_and_vocabulary_are_self_contained(self):
        clip = import_openai_clip(reproduce.CODE)
        self.assertIn("ViT-B/16", clip.available_models())
        self.assertTrue((reproduce.CODE / "models/clip/bpe_simple_vocab_16e6.txt.gz").is_file())

    def test_primary_plan_uses_corrected_qpr_and_twenty_supports(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(run_root=Path(temp), dataset="cub200", feature_cache=None,
                protocol=None, stages=list(reproduce.STAGES), data_root=Path(temp), device="cpu",
                workers=0, seeds=20, selection_episodes=200, audit_episodes=200,
                qpr_episodes=20, bootstrap=5000)
            plan = reproduce.build_plan(args)
        commands = [item["command"] for item in plan]
        self.assertTrue(any("fscil.qpr_calibration" in command for command in commands))
        self.assertFalse(any("fscil.qpr_selection" in command for command in commands))
        evaluation = next(command for command in commands if "fscil.run_qpr_self_exclusion" in command)
        self.assertEqual(evaluation[evaluation.index("--seed-limit") + 1], "20")
        self.assertEqual(evaluation[evaluation.index("--sessions") + 1], "all")


    def test_dtd_test_extraction_occurs_after_both_base_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(run_root=Path(temp), data_root=Path(temp), device="cpu", workers=0, seeds=20)
            plan = reproduce_dtd.build_plan(args)
        names = [item["name"] for item in plan]
        self.assertLess(names.index("shrinksel_audit"), names.index("test_features"))
        self.assertLess(names.index("qpr_audit"), names.index("test_features"))
        self.assertLess(names.index("base_bundle"), names.index("qpr_select"))


if __name__ == "__main__":
    unittest.main()
