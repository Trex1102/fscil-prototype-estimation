"""Protocol constants locked before target evaluation.

Nothing in this module is inferred from target predictions.  In particular, methods
must not construct class marginals from evaluation labels at runtime.
"""

from __future__ import annotations


PROTOCOL_SEED = 20260812
SUPPORT_SEEDS = tuple(range(1, 21))
SHOTS = (1, 2, 5, 10, 20)

DATASETS = {
    "cifar100": {
        "display": "CIFAR-100",
        "data_cfg": "configs/datasets/cifar100.yaml",
        "train_cfg": "configs/trainers/bimc_dino_fusion.yaml",
        "num_base": 60,
        "num_total": 100,
        "increment": 5,
        "feature_cache": "clip_feats_cls_cifar100_vitb16.pt",
    },
    "miniimagenet": {
        "display": "miniImageNet",
        "data_cfg": "configs/datasets/miniimagenet.yaml",
        "train_cfg": "configs/trainers/bimc_dino_fusion.yaml",
        "num_base": 60,
        "num_total": 100,
        "increment": 5,
        "feature_cache": "clip_feats_cls_miniimagenet_vitb16.pt",
    },
    "cub200": {
        "display": "CUB-200-2011",
        "data_cfg": "configs/datasets/cub200_bimc_dino_fusion.yaml",
        "train_cfg": "configs/trainers/bimc_dino_fusion.yaml",
        "num_base": 100,
        "num_total": 200,
        "increment": 10,
        "feature_cache": "clip_feats_cls_cub200_vitb16.pt",
    },
    "fgvc_aircraft": {
        "display": "FGVC-Aircraft",
        "dataset_loader": "fgvc_aircraft_2013b",
        "num_base": 50,
        "num_total": 100,
        "increment": 5,
        "feature_cache": "clip_feats_cls_fgvc_aircraft_vitb16.pt",
        "protocol_variant": (
            "published big-start structure: locked seed-1993 class order, 50 base "
            "classes, ten 5-way 5-shot sessions"
        ),
    },
    "dtd": {
        "display": "DTD",
        "dataset_loader": "dtd_r1.0.1_split1",
        "num_base": 27,
        "num_total": 47,
        "increment": 5,
        "feature_cache": "clip_feats_cls_dtd_vitb16.pt",
        "protocol_variant": (
            "locked seed-1993 class order; official split-1 train+validation "
            "pool; 27 base classes and four 5-way 5-shot sessions"
        ),
    },
}

BACKBONES = {
    "clip_vitb16": {
        "key": "clip_vitb16",
        "family": "OpenAI CLIP",
        "architecture": "ViT-B/16",
        "openai_model_name": "ViT-B/16",
        "feature_dim": 512,
        "input_resolution": 224,
        "pretraining_modalities": ["image", "text"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": "WIT-400M (full membership not publicly auditable)",
        "novelty_scope": "head-novel; not claimed pretraining-novel",
    },
    "clip_vitb32": {
        "key": "clip_vitb32",
        "family": "OpenAI CLIP",
        "architecture": "ViT-B/32",
        "openai_model_name": "ViT-B/32",
        "feature_dim": 512,
        "input_resolution": 224,
        "pretraining_modalities": ["image", "text"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": "WIT-400M (full membership not publicly auditable)",
        "novelty_scope": "head-novel; not claimed pretraining-novel",
    },
    "clip_rn50": {
        "key": "clip_rn50",
        "family": "OpenAI CLIP",
        "architecture": "RN50",
        "openai_model_name": "RN50",
        "feature_dim": 1024,
        "input_resolution": 224,
        "pretraining_modalities": ["image", "text"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": "WIT-400M (full membership not publicly auditable)",
        "novelty_scope": "head-novel; not claimed pretraining-novel",
    },
    "mae_ft_vitb16": {
        "key": "mae_ft_vitb16",
        "family": "MAE",
        "architecture": "ViT-B/16",
        "timm_model_name": "vit_base_patch16_224.mae",
        "checkpoint_url": (
            "https://dl.fbaipublicfiles.com/mae/finetune/"
            "mae_finetuned_vit_base.pth"
        ),
        "feature_dim": 768,
        "input_resolution": 224,
        "pretraining_modalities": ["image"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": (
            "ImageNet-1K training images (masked-autoencoder pretraining followed "
            "by supervised ImageNet-1K fine-tuning)"
        ),
        "pretraining_weights": (
            "official MAE ViT-B/16 ImageNet-1K fine-tuned checkpoint"
        ),
        "novelty_scope": "head-novel; not claimed pretraining-novel",
    },
    "supervised_resnet50": {
        "key": "supervised_resnet50",
        "family": "supervised ImageNet",
        "architecture": "ResNet-50",
        "torchvision_weights": "ResNet50_Weights.IMAGENET1K_V2",
        "feature_dim": 2048,
        "input_resolution": 224,
        "pretraining_modalities": ["image"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": "ImageNet-1K training images with labels",
        "pretraining_weights": "torchvision ResNet-50 IMAGENET1K_V2",
        "novelty_scope": "head-novel; not claimed pretraining-novel",
    },
    "base_only_resnet18": {
        "key": "base_only_resnet18",
        "family": "supervised base-session control",
        "architecture": "ResNet-18",
        "feature_dim": 512,
        "input_resolution": "dataset-specific",
        "pretraining_modalities": ["image"],
        "downstream_modalities": ["image"],
        "text_encoder_or_prompt_access": False,
        "pretraining_data": "authorized base-session training images only",
        "novelty_scope": "head-novel and benchmark-class-disjoint pretraining-novel",
    },
}

# Backward-compatible default for the already locked primary protocol.
BACKBONE = BACKBONES["clip_vitb16"]

# The uniform entries are protocol constants for the standard balanced test splits.
# CUB counts were committed after a one-time dataset-manifest audit and must match
# the generated protocol before the known-marginal diagnostic can be enabled.
KNOWN_TEST_COUNTS = {
    "cifar100": tuple([100] * 100),
    "miniimagenet": tuple([100] * 100),
    "cub200": (
        30, 30, 28, 30, 14, 11, 23, 18, 29, 30, 30, 26, 30, 30, 28, 28, 27, 15, 29, 29,
        30, 26, 29, 22, 30, 30, 30, 29, 30, 30, 30, 23, 29, 29, 30, 30, 29, 30, 29, 30,
        30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 28, 30, 29,
        30, 30, 30, 30, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 27, 30, 30, 29, 30, 30,
        30, 30, 30, 23, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 29, 30, 30, 30,
        20, 30, 30, 30, 19, 30, 30, 30, 30, 30, 30, 30, 20, 30, 29, 30, 29, 30, 29, 30,
        30, 30, 30, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        29, 30, 30, 30, 30, 30, 30, 30, 29, 30, 21, 30, 29, 30, 30, 30, 29, 30, 30, 29,
        30, 30, 30, 30, 30, 29, 30, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 26, 29, 30,
        29, 30, 30, 30, 30, 30, 20, 30, 30, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    ),
    # Committed from the 3,333-row 2013b variant test annotation under the locked
    # seed-1993 class permutation. The Aircraft stage verifies this complete tuple
    # against the extracted official archive before computing any target metric.
    "fgvc_aircraft": (
        34, 34, 33, 34, 34, 33, 33, 34, 34, 33, 33, 34, 33, 34, 33, 33, 34,
        34, 33, 34, 33, 33, 34, 33, 33, 33, 34, 34, 33, 34, 33, 33, 33, 33,
        33, 34, 34, 33, 33, 34, 33, 34, 33, 33, 33, 33, 34, 33, 33, 33, 33,
        34, 34, 34, 33, 34, 33, 34, 33, 33, 33, 33, 34, 34, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 34, 33, 34, 33, 33, 33, 34, 33, 33, 33, 33,
        33, 33, 34, 34, 33, 33, 33, 33, 33, 33, 33, 34, 33, 33, 33,
    ),
    "dtd": tuple([40] * 47),
}

LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
RIDGE_GAMMA = 1.0
PSEUDO_EPISODES = 200
PSEUDO_NOVEL_PER_EPISODE = {
    "cifar100": 5,
    "miniimagenet": 5,
    "cub200": 10,
    "fgvc_aircraft": 5,
    "dtd": 5,
}
PSEUDO_QUERY_PER_CLASS = 15
