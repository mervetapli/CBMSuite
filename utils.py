import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable, List, Any

import data_utils
import config
from model_loaders import TargetModelLoader, get_vlm_loader


def create_save_directory(save_path: str) -> None:
    """Create parent directory for save path if it doesn't exist. No-op for an empty path (used as a "not applicable" sentinel, e.g. no text save path when saving backbone-only features)."""
    if not save_path:
        return
    save_dir = save_path[:save_path.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def all_files_exist(file_dict: Dict[str, str]) -> bool:
    """Check if all files in dictionary exist."""
    return all(os.path.exists(path) for path in file_dict.values())


def get_activation_hook(outputs: List, mode: str = "avg") -> Callable:
    """
    Create a forward hook for capturing activations.

    Args:
        outputs: List to append activations to
        mode: Pooling mode - 'avg' for average pooling, 'max' for max pooling

    Returns:
        Hook function
    """
    def hook(model, input, output):
        if mode == 'avg':
            if len(output.shape) == 4:  # 2D feature maps
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:  # 1D features (already pooled)
                outputs.append(output.detach().cpu())
        elif mode == 'max':
            if len(output.shape) == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())

    return hook


def _register_hooks(model: torch.nn.Module, layers: List[str], outputs: Dict) -> Dict:
    """Register forward hooks on specified layers."""
    hooks = {}
    for layer_name in layers:
        hook_func = get_activation_hook(outputs[layer_name], "avg")
        command = f"model.{layer_name}.register_forward_hook(hook_func)"
        hooks[layer_name] = eval(command)
    return hooks


def save_target_activations(
    target_model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    save_path_template: str,
    target_layers: List[str],
    batch_size: int = 512,
    device: str = "cuda",
    pool_mode: str = "avg",
    num_workers: int = 8,
) -> None:
    """
    Save activations from target model to disk.

    Args:
        target_model: Target model to extract activations from
        dataset: Dataset to process
        save_path_template: Path template with {} placeholder for layer names
        target_layers: List of layer names to extract
        batch_size: Batch size for processing
        device: Device to compute on
        pool_mode: How to pool spatial dimensions ('avg' or 'max')
        num_workers: Number of data loader workers
    """
    create_save_directory(save_path_template.format(target_layers[0]))

    save_paths = {layer: save_path_template.format(layer) for layer in target_layers}

    if all_files_exist(save_paths):
        return

    # Special handling for specific model types
    if "franca" in save_path_template:
        all_features = []
        with torch.no_grad():
            for images, labels in tqdm(
                DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
            ):
                features = target_model(images.to(device))
                intermediate_output = features[-4:]
                output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
                all_features.append(output.to('cpu'))
        torch.save(torch.cat(all_features), save_paths[target_layers[0]])

    elif "perception_encoder" in save_path_template:
        all_features = []
        with torch.no_grad():
            for images, labels in tqdm(
                DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
            ):
                features = target_model.encode_image(images.to(device))
                all_features.append(features.to('cpu'))
        torch.save(torch.cat(all_features), save_paths[target_layers[0]])

    else:
        # Standard layer-wise extraction with hooks
        all_features = {layer: [] for layer in target_layers}
        hooks = _register_hooks(target_model, target_layers, all_features)

        with torch.no_grad():
            for images, labels in tqdm(
                DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
            ):
                _ = target_model(images.to(device))

        for layer in target_layers:
            torch.save(torch.cat(all_features[layer]), save_paths[layer])
            hooks[layer].remove()

    torch.cuda.empty_cache()


def save_vlm_features(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    text_inputs: Any,
    image_save_path: str,
    text_save_path: str,
    model_type: str = "standard",
    batch_size: int = 512,
    device: str = "cuda",
    num_workers: int = 8,
) -> None:
    """
    Save VLM (Vision-Language Model) image and text features.

    Args:
        model: Vision-Language model (CLIP, SAIL, SigLIP, or FLAIR)
        dataset: Image dataset
        text_inputs: Tokenized text inputs
        image_save_path: Path to save image features
        text_save_path: Path to save text features
        model_type: Type of model ('standard', 'SAIL', 'SigLIP', 'FLAIR')
        batch_size: Batch size for processing
        device: Device to compute on
        num_workers: Number of data loader workers
    """
    create_save_directory(image_save_path)
    create_save_directory(text_save_path)

    # Skip if already exist. An empty text_save_path means "no text to save"
    # (e.g. CLIP used as a backbone rather than a VLM), so it's vacuously "done".
    if os.path.exists(image_save_path) and (not text_save_path or os.path.exists(text_save_path)):
        return

    # Save image features
    if not os.path.exists(image_save_path):
        _save_vlm_image_features(
            model, dataset, image_save_path, model_type, batch_size, device, num_workers
        )

    # Save text features
    if text_save_path and not os.path.exists(text_save_path):
        _save_vlm_text_features(model, text_inputs, text_save_path, model_type, batch_size, device)

    torch.cuda.empty_cache()


def _save_vlm_image_features(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    save_path: str,
    model_type: str,
    batch_size: int,
    device: str,
    num_workers: int,
) -> None:
    """Save image features from a vision-language model."""
    all_features = []

    # Loaded once up front rather than per-batch -- AutoProcessor.from_pretrained
    # does non-trivial disk I/O even when cached locally.
    siglip_image_processor = None
    if model_type == "SigLIP":
        from transformers import AutoProcessor
        siglip_image_processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL_ID).image_processor

    with torch.no_grad():
        for images, labels in tqdm(
            DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
        ):
            if model_type == "SAIL":
                image_processor = model.image_processor
                images = image_processor(images=images, return_tensors="pt")
                features = model.encode_image(images.to(device), normalize=True)
                all_features.append(features.cpu())

            elif model_type == "SigLIP":
                # images are already float in [0, 1] (torchvision ToTensor()
                # upstream) -- do_rescale=False so the processor doesn't
                # divide by 255 a second time and collapse pixel values.
                # (Calling .image_processor directly since the composed
                # SiglipProcessor.__call__ doesn't forward do_rescale.)
                images = siglip_image_processor(images=images, return_tensors="pt", do_rescale=False)
                features = model.get_image_features(**images.to(device))
                all_features.append(features.cpu())

            elif model_type == "FLAIR":
                global_image_token, _ = model.encode_image(torch.Tensor(images).to(device))
                global_image_token = model.image_post(global_image_token)
                features = torch.nn.functional.normalize(global_image_token, dim=-1)
                all_features.append(features.cpu())

            elif model_type == "clip_feature":
                # CLIP used as a backbone: `model` here is already the
                # TargetModelLoader-provided callable (e.g. a lambda wrapping
                # .encode_image(...).float()), not the raw CLIP module.
                features = model(torch.Tensor(images).to(device))
                all_features.append(features.cpu())

            else:  # standard CLIP (as a VLM)
                features = model.encode_image(torch.Tensor(images).to(device))
                all_features.append(features.cpu())

            torch.cuda.empty_cache()

    torch.save(torch.cat(all_features), save_path)


def _move_to_device(x: Any, device: str) -> Any:
    """Move a tensor, or each tensor value in a dict/BatchEncoding, to device. Anything else (e.g. a plain list of strings) is passed through unchanged."""
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in x.items()}
    return x


def _save_vlm_text_features(
    model: torch.nn.Module,
    text_inputs: Any,
    save_path: str,
    model_type: str,
    batch_size: int,
    device: str,
) -> None:
    """Save text features from a vision-language model."""
    batch_size = 100
    text_features = []

    with torch.no_grad():
        if model_type == "SigLIP":
            text_features.append(model.get_text_features(**_move_to_device(text_inputs, device)))

        elif model_type == "FLAIR":
            global_text_token, _ = model.encode_text(_move_to_device(text_inputs, device))
            global_text_token = model.text_post(global_text_token)
            features = torch.nn.functional.normalize(global_text_token, dim=-1)
            text_features.append(features)

        else:  # standard CLIP or SAIL
            for i in tqdm(range(math.ceil(len(text_inputs) / batch_size))):
                batch = text_inputs[batch_size * i : batch_size * (i + 1)]

                if model_type == "SAIL":
                    features = model.encode_text(batch, text_list=batch, normalize=True).detach().cpu()
                else:
                    features = model.encode_text(_move_to_device(batch, device))

                text_features.append(features)
                torch.cuda.empty_cache()

    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_path)


def get_save_paths(
    vlm_name: str,
    target_name: str,
    target_layer: str,
    d_probe: str,
    concept_set: str,
    pool_mode: str,
    save_dir: str,
) -> Tuple[str, str, str]:
    """
    Generate save paths for activations.

    Returns:
        Tuple of (target_save_path, vlm_save_path, text_save_path)
    """
    # Target model save path
    if target_name.startswith("clip_"):
        target_save_path = f"{save_dir}/{d_probe}_{target_name.replace('/', '')}.pt"
    else:
        pool_suffix = config.POOL_MODE_SUFFIX.get(pool_mode, "")
        target_save_path = f"{save_dir}/{d_probe}_{target_name}_{target_layer}{pool_suffix}.pt"

    # Extract concept set name from path
    concept_set_name = concept_set.split("/")[-1].split(".")[0]

    # VLM save paths
    if vlm_name.startswith("SAIL"):
        vlm_save_path = f"{save_dir}/{d_probe}_SAIL.pt"
        text_save_path = f"{save_dir}/{concept_set_name}_SAIL.pt"
    elif vlm_name.startswith("SigLIP"):
        vlm_save_path = f"{save_dir}/{d_probe}_SigLIP.pt"
        text_save_path = f"{save_dir}/{concept_set_name}_SigLIP.pt"
    elif vlm_name.startswith("FLAIR"):
        vlm_save_path = f"{save_dir}/{d_probe}_FLAIR.pt"
        text_save_path = f"{save_dir}/{concept_set_name}_FLAIR.pt"
    else:
        vlm_save_path = f"{save_dir}/{d_probe}_clip_{vlm_name.replace('/', '')}.pt"
        text_save_path = f"{save_dir}/{concept_set_name}_{vlm_name.replace('/', '')}.pt"

    return target_save_path, vlm_save_path, text_save_path


def save_activations(
    vlm_name: str,
    target_name: str,
    d_probe: str,
    concept_set: str,
    target_layers: List[str],
    batch_size: int = 512,
    device: str = "cuda",
    pool_mode: str = "avg",
    save_dir: str = "saved_activations",
    num_workers: int = 8,
) -> None:
    """
    Save both target model and vision-language model activations.

    Args:
        vlm_name: Name of vision-language model (CLIP, SAIL, SigLIP, FLAIR)
        target_name: Name of target (backbone) model
        d_probe: Dataset name (e.g., 'cifar10_train')
        concept_set: Path to concept set file
        target_layers: Layers to extract from target model
        batch_size: Batch size for processing
        device: Device to compute on
        pool_mode: How to pool spatial dimensions ('avg' or 'max')
        save_dir: Directory to save activations
        num_workers: Number of data loader workers
    """
    # Get save paths
    target_save_path, vlm_save_path, text_save_path = get_save_paths(
        vlm_name, target_name, "{}", d_probe, concept_set, pool_mode, save_dir
    )

    save_files = {
        "vlm": vlm_save_path,
        "text": text_save_path,
    }
    for layer in target_layers:
        save_files[layer] = target_save_path.format(layer)

    # Skip if all files exist
    if all_files_exist(save_files):
        return

    # Load concept set
    with open(concept_set, 'r') as f:
        words = f.read().split('\n')

    # Load target model
    if target_name.startswith("clip_"):
        target_model, target_preprocess = TargetModelLoader.load(target_name, device)
    else:
        target_model, target_preprocess = TargetModelLoader.load(target_name, device)

    # Load target dataset
    target_dataset = data_utils.get_data(d_probe, target_preprocess)

    # Load and save vision-language model features
    vlm_loader = get_vlm_loader(vlm_name)
    vlm_model, vlm_preprocess = vlm_loader.load_image_text_models(device)

    # Prepare text
    text_inputs = vlm_loader.prepare_text(words)

    # Prepare image data
    if vlm_name.startswith(("SAIL", "SigLIP", "FLAIR")):
        import torchvision.transforms
        vlm_dataset = data_utils.get_data(d_probe, torchvision.transforms.ToTensor())
    else:
        vlm_dataset = data_utils.get_data(d_probe, vlm_preprocess)

    # Determine model type for feature saving
    model_type = "SAIL" if vlm_name.startswith("SAIL") else \
                 "SigLIP" if vlm_name.startswith("SigLIP") else \
                 "FLAIR" if vlm_name.startswith("FLAIR") else "standard"

    # Save VLM features
    save_vlm_features(
        vlm_model,
        vlm_dataset,
        text_inputs,
        vlm_save_path,
        text_save_path,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    # Save target model features
    if target_name.startswith("clip_"):
        save_vlm_features(
            target_model,
            target_dataset,
            None,
            target_save_path,
            "",
            model_type="clip_feature",
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
    else:
        save_target_activations(
            target_model,
            target_dataset,
            target_save_path,
            target_layers,
            batch_size=batch_size,
            device=device,
            pool_mode=pool_mode,
            num_workers=num_workers,
        )



# ============================================================================
# Evaluation functions
# ============================================================================

def get_accuracy_cbm(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    batch_size: int = 250,
    num_workers: int = 2,
) -> float:
    """
    Compute accuracy of CBM model on dataset.

    Args:
        model: CBM model
        dataset: Test dataset
        device: Device to compute on
        batch_size: Batch size
        num_workers: Number of data loader workers

    Returns:
        Accuracy as a fraction
    """
    correct = 0
    total = 0

    for images, labels in tqdm(
        DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    ):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu() == labels)
            total += len(labels)

    return correct / total


def get_preds_cbm(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    batch_size: int = 250,
    num_workers: int = 2,
) -> torch.Tensor:
    """
    Get predictions of CBM model on dataset.

    Args:
        model: CBM model
        dataset: Test dataset
        device: Device to compute on
        batch_size: Batch size
        num_workers: Number of data loader workers

    Returns:
        Tensor of predictions
    """
    preds = []

    for images, labels in tqdm(
        DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    ):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())

    return torch.cat(preds, dim=0)


def get_concept_activations_by_prediction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    batch_size: int = 500,
    num_workers: int = 8,
) -> torch.Tensor:
    """
    Get concept activations grouped by model prediction.

    Args:
        model: CBM model
        dataset: Test dataset
        device: Device to compute on
        batch_size: Batch size
        num_workers: Number of data loader workers

    Returns:
        Tensor of concept activations averaged per class
    """
    preds = []
    concept_acts = []

    for images, _labels in tqdm(
        DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
    ):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())

    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)

    concept_acts_by_pred = []
    for i in range(torch.max(preds) + 1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds == i], dim=0))

    return torch.stack(concept_acts_by_pred, dim=0)
