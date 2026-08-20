"""
Data loading utilities for various datasets.

Provides functions for loading images and labels from different datasets
(CIFAR, ImageNet, CUB, etc.) with appropriate preprocessing.
"""

import os
import torch
from torchvision import datasets, transforms
import torchvision
from functools import partial

import config
from model_loaders import TargetModelLoader

# Import optional dependencies
try:
    from franca.hub.backbones import _make_franca_model
    from franca.eval.utils import ModelWithIntermediateLayers
    FRANCA_AVAILABLE = True
except ImportError:
    FRANCA_AVAILABLE = False

try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transforms
    PE_AVAILABLE = True
except ImportError:
    PE_AVAILABLE = False

try:
    from pytorchcv.model_provider import get_model as ptcv_get_model
    PYTORCHCV_AVAILABLE = True
except ImportError:
    PYTORCHCV_AVAILABLE = False

# Use configuration constants
DATASET_ROOTS = config.DATASET_ROOTS
LABEL_FILES = config.LABEL_FILES



def get_imagenet_folder_dataset(path: str, transform=None) -> datasets.ImageFolder:
    """Load ImageNet-style folder dataset."""
    return torchvision.datasets.ImageFolder(path, transform=transform)


def get_data(dataset_name: str, preprocess=None) -> torch.utils.data.Dataset:
    """
    Load dataset by name with optional preprocessing.

    Args:
        dataset_name: Name of the dataset to load
        preprocess: Transform to apply to images

    Returns:
        PyTorch dataset
    """
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                    transform=preprocess)
    


    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]) ) #preprocess)
               
    return data

def get_targets_only(dataset_name: str) -> list:
    """
    Extract target labels from dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of target labels
    """
    pil_data = get_data(dataset_name)
    return pil_data.targets


def get_target_model(target_name: str, device: str):
    """
    Load target model by name.

    This function is kept for backwards compatibility.
    Use TargetModelLoader.load() for new code.

    Args:
        target_name: Name of the target model
        device: Device to load on

    Returns:
        Tuple of (model, preprocessing_function)
    """
    return TargetModelLoader.load(target_name, device)

