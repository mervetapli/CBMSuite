"""
Model loading factories for different model types.

This module provides factory classes for loading various model architectures
(CLIP, SAIL, SigLIP, FLAIR) in a clean, extensible way.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Callable
from functools import partial

import torch
import clip
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from transformers import AutoTokenizer, AutoModel, AutoProcessor

# Import conditional dependencies
try:
    import flair
    FLAIR_AVAILABLE = True
except ImportError:
    FLAIR_AVAILABLE = False

try:
    from SAIL.model import create_model
    SAIL_AVAILABLE = True
except ImportError:
    SAIL_AVAILABLE = False

try:
    from pytorchcv.model_provider import get_model as ptcv_get_model
    PYTORCHCV_AVAILABLE = True
except ImportError:
    PYTORCHCV_AVAILABLE = False

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

import config


class VLMModelLoader(ABC):
    """Base class for vision-language model (VLM) loaders (CLIP, SAIL, SigLIP, FLAIR)."""

    @abstractmethod
    def load_image_text_models(self, device: str) -> Tuple[Any, Any]:
        """Load both image and text encoding models."""
        pass

    @abstractmethod
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for the model."""
        pass

    @abstractmethod
    def prepare_text(self, words: list) -> Any:
        """Prepare text for the model."""
        pass


class StandardCLIPLoader(VLMModelLoader):
    """Loader for standard OpenAI CLIP models."""

    def __init__(self, clip_name: str):
        self.clip_name = clip_name

    def load_image_text_models(self, device: str) -> Tuple[Any, Any]:
        """Load standard CLIP model."""
        model, preprocess = clip.load(self.clip_name, device=device)
        return model, preprocess

    def preprocess_images(self, images):
        """Images are preprocessed during data loading."""
        return images

    def prepare_text(self, words: list):
        """Tokenize text using CLIP tokenizer."""
        text_prompts = [f"a photo of a {word}" for word in words]
        return clip.tokenize(text_prompts)


class SAILModelLoader(VLMModelLoader):
    """Loader for SAIL models."""

    def load_image_text_models(self, device: str) -> Tuple[Any, Any]:
        """Load SAIL model."""
        if not SAIL_AVAILABLE:
            raise ImportError("SAIL package not available. Install SAIL to use SAILModelLoader.")
        sail_model = create_model(
            text_model_name=config.SAIL_TEXT_MODEL,
            vision_model_name=config.SAIL_VISION_MODEL,
            head_weights_path=config.SAIL_CHECKPOINT_PATH,
            target_dimension=config.SAIL_TARGET_DIMENSION,
        )
        sail_model.to(device)
        sail_model.eval()
        return sail_model, None

    def preprocess_images(self, images):
        """Preprocess images for SAIL."""
        return torchvision.transforms.ToTensor()(images)

    def prepare_text(self, words: list):
        """Prepare text for SAIL."""
        return [f"{word}" for word in words]


class SigLIPModelLoader(VLMModelLoader):
    """Loader for SigLIP models."""

    def load_image_text_models(self, device: str) -> Tuple[Any, Any]:
        """Load SigLIP model and processor."""
        model = AutoModel.from_pretrained(config.SIGLIP_MODEL_ID).to(device)
        processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL_ID)
        return model, processor

    def preprocess_images(self, images):
        """Preprocess images for SigLIP."""
        return torchvision.transforms.ToTensor()(images)

    def prepare_text(self, words: list):
        """Prepare text for SigLIP."""
        tokenizer = AutoTokenizer.from_pretrained(config.SIGLIP_MODEL_ID)
        text_prompts = [f"a photo of a {word}" for word in words]
        return tokenizer(text_prompts, padding="max_length", return_tensors="pt")


class FLAIRModelLoader(VLMModelLoader):
    """Loader for FLAIR models."""

    def load_image_text_models(self, device: str) -> Tuple[Any, Any]:
        """Load FLAIR model."""
        if not FLAIR_AVAILABLE:
            raise ImportError("FLAIR package not available. Install FLAIR to use FLAIRModelLoader.")
        pretrained = flair.download_weights_from_hf(
            model_repo=config.FLAIR_MODEL_REPO,
            filename=config.FLAIR_MODEL_FILE
        )
        flair_model, _, processor = flair.create_model_and_transforms(
            config.FLAIR_MODEL_ARCHITECTURE,
            pretrained=pretrained
        )
        flair_model.to(device)
        flair_model.eval()
        return flair_model, processor

    def preprocess_images(self, images):
        """Preprocess images for FLAIR."""
        return images

    def prepare_text(self, words: list):
        """Prepare text for FLAIR."""
        tokenizer = flair.get_tokenizer(config.FLAIR_MODEL_ARCHITECTURE)
        text_prompts = [f"a photo of a {word}" for word in words]
        return tokenizer(text_prompts)


def get_vlm_loader(vlm_name: str) -> VLMModelLoader:
    """Factory function to get appropriate vision-language model loader."""
    if vlm_name.startswith("SAIL"):
        return SAILModelLoader()
    elif vlm_name.startswith("SigLIP"):
        return SigLIPModelLoader()
    elif vlm_name.startswith("FLAIR"):
        return FLAIRModelLoader()
    else:
        return StandardCLIPLoader(vlm_name)


class TargetModelLoader:
    """Loader for target (backbone) models."""

    @staticmethod
    def get_resnet_preprocess() -> transforms.Compose:
        """Get standard ResNet preprocessing."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.RESNET_MEAN, std=config.RESNET_STD)
        ])

    @staticmethod
    def load(target_name: str, device: str) -> Tuple[torch.nn.Module, Callable]:
        """
        Load target model by name.

        Args:
            target_name: Name of the target model to load
            device: Device to load model on ('cuda' or 'cpu')

        Returns:
            Tuple of (model, preprocessing_function)
        """
        if target_name.startswith("clip_"):
            return TargetModelLoader._load_clip(target_name[5:], device)
        elif target_name == 'resnet18_places':
            return TargetModelLoader._load_resnet18_places(device)
        elif target_name == 'resnet18_cub':
            return TargetModelLoader._load_resnet18_cub(device)
        elif target_name == 'dinov2':
            return TargetModelLoader._load_dinov2(device)
        elif target_name == config.PERCEPTION_ENCODER_ID:
            return TargetModelLoader._load_perception_encoder(device)
        elif target_name == config.FRANCA_MODEL_ID:
            return TargetModelLoader._load_franca(device)
        elif target_name.endswith("_v2"):
            return TargetModelLoader._load_torchvision_model(target_name[:-3], device, v2=True)
        else:
            return TargetModelLoader._load_torchvision_model(target_name, device, v2=False)

    @staticmethod
    def _load_clip(clip_model_name: str, device: str) -> Tuple[Callable, Callable]:
        """Load CLIP as target model."""
        model, preprocess = clip.load(clip_model_name, device=device)
        model.eval()
        target_model = lambda x: model.encode_image(x).float()
        return target_model, preprocess

    @staticmethod
    def _load_resnet18_places(device: str) -> Tuple[torch.nn.Module, Callable]:
        """Load ResNet18 trained on Places365."""
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        return target_model, TargetModelLoader.get_resnet_preprocess()

    @staticmethod
    def _load_resnet18_cub(device: str) -> Tuple[torch.nn.Module, Callable]:
        """Load ResNet18 trained on CUB dataset."""
        if not PYTORCHCV_AVAILABLE:
            raise ImportError("pytorchcv not available. Install pytorchcv to load resnet18_cub.")
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        return target_model, TargetModelLoader.get_resnet_preprocess()


    @staticmethod
    def _load_dinov2(device: str) -> Tuple[torch.nn.Module, Callable]:
        """Load DINOv2 model."""
        target_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        target_model.cuda()
        target_model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(244),
            transforms.CenterCrop(224),
            transforms.Normalize([0.5], [0.5])
        ])
        return target_model, preprocess

    @staticmethod
    def _load_perception_encoder(device: str) -> Tuple[torch.nn.Module, Callable]:
        """Load Perception Encoder model."""
        if not PE_AVAILABLE:
            raise ImportError("core.vision_encoder not available.")
        target_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        target_model.cuda()
        target_model.eval()
        preprocess = pe_transforms.get_image_transform(target_model.image_size)
        return target_model, preprocess

    @staticmethod
    def _load_franca(device: str) -> Tuple[torch.nn.Module, Callable]:
        """Load Franca model."""
        if not FRANCA_AVAILABLE:
            raise ImportError("franca not available.")
        model = _make_franca_model(
            arch_name=config.FRANCA_ARCH,
            img_size=config.FRANCA_IMG_SIZE,
            pretrained=True,
            use_rasa_head=True
        )
        autocast_ctx = partial(
            torch.cuda.amp.autocast,
            enabled=True,
            dtype=torch.float32
        )
        target_model = ModelWithIntermediateLayers(
            model,
            config.FRANCA_NUM_INTERMEDIATE_LAYERS,
            autocast_ctx
        )
        target_model.cuda()
        target_model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        return target_model, preprocess

    @staticmethod
    def _load_torchvision_model(
        model_name: str,
        device: str,
        v2: bool = False
    ) -> Tuple[torch.nn.Module, Callable]:
        """Load standard torchvision model."""
        model_name_cap = model_name.replace("resnet", "ResNet")
        weights_type = "IMAGENET1K_V2" if v2 else "IMAGENET1K_V1"
        weights = eval(f"models.{model_name_cap}_Weights.{weights_type}")
        target_model = eval(f"models.{model_name}(weights=weights).to(device)")
        target_model.eval()
        return target_model, weights.transforms()
