# Prototype Estimation in FSCIL

Official implementation of **Diagnosing Prototype Estimation in Few-Shot Class-Incremental Learning: Training-Free Geometry Selection and Unlabeled-Query Prototype Rectification** (WACV 2027).

The implementation provides frozen-feature prototype diagnostics, ShrinkSel, QPR, comparator methods, and the experiment runners used in the paper. The Python package is `fscil/`.

## Setup

Tested with Python 3.8.20, PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117, NumPy 1.23.5, and scikit-learn 1.3.2. Dependencies are pinned in `requirements.txt`.

```bash
git clone https://github.com/Trex1102/fscil-prototype-estimation.git
cd fscil-prototype-estimation
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
python verify.py
```

Run commands from the repository root. The CLIP implementation and tokenizer vocabulary are included. Image datasets and pretrained checkpoints are external inputs; obtain them under their original terms. Feature extraction downloads the encoder checkpoints from their released URLs. Use `--device cpu` for CPU evaluation; full experiments normally use a GPU.

## Data

Supply `--data-root` with the following layout:

| Dataset | Files below the dataset root |
| --- | --- |
| CUB-200-2011 | `CUB_200_2011/images.txt`, `train_test_split.txt`, `image_class_labels.txt`, and `images/` |
| CIFAR-100 | `cifar-100-python/` (downloaded by torchvision if absent) |
| miniImageNet | `miniimagenet/images/`, `miniimagenet/split/train.csv`, and `test.csv` |
| FGVC-Aircraft | The official `fgvc-aircraft-2013b` extraction |
| DTD | The official r1.0.1 extraction with split-1 annotations |

Preserve the loader ordering and original miniImageNet CSVs. Replacing the splits or class order changes the evaluation protocol.

## Reproduction

```bash
python reproduce.py --dataset cub200 --data-root /datasets --run-root runs/paper
python reproduce.py --dataset cifar100 --data-root /datasets --run-root runs/paper
python reproduce.py --dataset miniimagenet --data-root /datasets --run-root runs/paper
```

Each run extracts features, constructs the protocol and support draws, selects and audits the support-only and query-assisted parameters on base classes, and evaluates the diagnostics and comparison methods. Defaults use CLIP ViT-B/16 and 20 matched support draws. QPR evaluates both the same-cohort and exact self-excluded updates. Selection and audit complete before target evaluation.

Use `--dry-run` to inspect commands, `--feature-cache` to reuse matching features, and `--stages prepare,shrinksel,qpr` to run selected stages. Later stages require the earlier outputs. A one-seed run is a smoke test, not the reported experiment.

For other frozen encoders, use `--backbone clip_vitb32`, `clip_rn50`, `mae_ft_vitb16`, or `supervised_resnet50`. QPR is evaluated by default for the CLIP encoders; non-CLIP runs retain the support-only scope.

```bash
python reproduce.py --dataset fgvc_aircraft --data-root /datasets/fgvc-aircraft-2013b \
  --run-root runs/paper --stages prepare,shrinksel,diagnostics,baselines
python reproduce_dtd.py --data-root /datasets/dtd --run-root runs/paper
```

The DTD runner extracts test features after both base-only audits and stops if the support-only audit fails.

The method-comparison runner takes the parameter directory produced above:

```bash
python -m fscil.run_common_head_benchmark --dataset cub200 \
  --feature-cache runs/paper/artifacts/features/clip_feats_cls_cub200_vitb16.pt \
  --protocol runs/paper/artifacts/protocol/cub200_clip_vitb16_protocol.pt \
  --parameter-root runs/paper/parameters/clip_vitb16/cub200 \
  --output runs/comparison/cub200.json
```

This evaluates the comparator implementations with Cosine-NCM and ShrinkSel, using the same base-selected settings and matched supports. The primary runner also retains each comparator's native inference path. The two paths differ in initialization and preprocessing; keep their outputs distinct when analyzing results.

## Additional experiments

Every module below supports `python -m fscil.MODULE --help`. Use the generated feature, protocol, and parameter files, and write results to a separate output path.

| Experiment | Module |
| --- | --- |
| Radius/covariance factorial | `run_radial_factorial_ablation` |
| Cumulative base calibration | `cumulative_radial_selection` |
| Novel-class arrival order | `run_novel_order_robustness` |
| QPR components, support counts, and cohort size | `run_qpr_ablation`, `run_qpr` |
| QPR selector partitions | `run_corrected_qpr_selector_stability` |
| Adaptation and linear-probe controls | `run_adaptation_baselines`, `run_balanced_linear_probe` |
| RanPAC and RanDumb | `run_ranpac_projection_target`, `run_randumb_stage` |
| Low-way transductive evaluation | `run_native_transductive_sanity` |
| Query-scoring resources | `run_resource_scaling_stage` |
| Base-session-only representation | `train_base_only_backbone`, `run_base_only_control_stage` |

`python verify.py` checks source integrity and runs the tests for the estimators, selection rules, information boundaries, comparator geometry, self-exclusion, data protocols, and reproduction commands. Tests use synthetic inputs and do not require image datasets.

The legacy trainer config filename `bimc_dino_fusion.yaml` supplies dataset/loader settings only. These runners use frozen visual encoders and do not execute a DINOv2 or text-calibration model. CLIP's license is retained in `third_party/CLIP-LICENSE`.
