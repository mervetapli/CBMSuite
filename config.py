"""
Configuration and constants for CBM Suite.

This module centralizes configuration values, magic strings, and common constants
to improve maintainability and reduce duplication across the codebase.
"""

# Checkpoint and pretrained model URLs
SAIL_CHECKPOINT_PATH = "../SAIL/checkpoint/sail_dinov2l_nv2.pt"
SAIL_TEXT_MODEL = "nvidia/NV-Embed-v2"
SAIL_VISION_MODEL = "facebook/dinov2-large"
SAIL_TARGET_DIMENSION = 1024

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"

FLAIR_MODEL_REPO = "xiaorui638/flair"
FLAIR_MODEL_FILE = "flair-cc3m-recap.pt"
FLAIR_MODEL_ARCHITECTURE = "ViT-B-16-FLAIR"

# Dataset and path constants
# Fill these in with your own local paths before training on these datasets.
DATASET_ROOTS = {
    "imagenet_train": "/path/to/ImageNet/train/",
    "imagenet_val": "/path/to/ImageNet/val/",
    "cub_train": "/path/to/CUB_200_2011/train",
    "cub_val": "/path/to/CUB_200_2011/test",
    "imagenet_100_train": "/path/to/imagenet-100/train/",
    "imagenet_100_val": "/path/to/imagenet-100/val/"
}

LABEL_FILES = {
    "places365": "data/categories_places365_clean.txt",
    "imagenet": "data/imagenet_classes.txt",
    "cifar10": "data/cifar10_classes.txt",
    "cifar100": "data/cifar100_classes.txt",
    "cub": "data/cub_classes.txt",
    "imagenet_100": "data/imagenet100_classes.txt"
}

# Pool mode suffixes for save file naming
POOL_MODE_SUFFIX = {"max": "_max", "avg": ""}

# Normalization constants
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

# Franca model configuration
FRANCA_ARCH = "vit_large"
FRANCA_IMG_SIZE = 224
FRANCA_NUM_INTERMEDIATE_LAYERS = 4

# Special model identifiers
FRANCA_MODEL_ID = "franca"
PERCEPTION_ENCODER_ID = "perception_encoder"
