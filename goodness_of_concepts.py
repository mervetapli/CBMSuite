import torch
import numpy as np
import argparse

import utils
import data_utils


# Constants
DEFAULT_TOP_CONCEPTS = 100  # Number of top concepts to consider for entropy calculation
EPSILON = 1e-10  # For numerical stability in softmax computation
NORM_EPSILON = 1e-10  # For numerical stability in normalization


parser = argparse.ArgumentParser(description='Settings for calculating task specific Goodness of Concepts')

parser.add_argument("--dataset", type=str, default="imagenet_100")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--vlm_name", type=str, default="ViT-B/16", help="Which vision-language model to use")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--top_concepts", type=int, default=DEFAULT_TOP_CONCEPTS, 
                    help="Number of top concepts to consider for entropy calculation")
parser.add_argument("--top_k", type=int, default=250,
                    help="Top-k logits to keep for task-agnostic mode (per image)")
parser.add_argument("--mode", type=str, choices=["specific", "agnostic"], default="specific",
                    help="Mode: 'specific' uses class averages, 'agnostic' computes per-image entropy")



def compute_class_averages(vlm_features, class_indices):
    start_indices, counts = class_indices
    vlm_features_np = vlm_features.numpy()
    
    class_averages = []
    for start_idx, count in zip(start_indices, counts):
        subset = vlm_features_np[start_idx:start_idx + count, :]
        class_averages.append(np.mean(subset, axis=0))
    
    return np.array(class_averages)


def select_top_concepts(class_averages, num_concepts):
    indices = np.argsort(np.abs(class_averages), axis=1)[:, -num_concepts:]
    return np.take_along_axis(class_averages, indices, axis=1)


def compute_entropy(activations, epsilon=EPSILON):
    # Convert activations to probabilities using softmax
    exp_activations = np.exp(activations.real)  # Use real part for safety
    probabilities = exp_activations / (exp_activations.sum(axis=1, keepdims=True) + epsilon)
    
    # Compute entropy
    entropy = (-probabilities * np.log(probabilities + epsilon)).sum(axis=1)
    
    return entropy


def calculate_entropy(args):

    # Load concepts
    with open(args.concept_set, "r") as f:
        concepts = [c for c in f.read().split("\n") if c.strip()]
    
    # Prepare dataset names
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    # Save activations
    for d_probe in [d_train, d_val]:
        utils.save_activations(vlm_name = args.vlm_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)

    # Get save paths
    target_save_name, vlm_save_name, text_save_name = utils.get_save_paths(
        args.vlm_name, args.backbone, args.feature_layer, d_train, 
        args.concept_set, "avg", args.activation_dir
    )
    
    
    # Load and normalize features
    print("Loading and normalizing features...")
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        image_features = torch.load(vlm_save_name, map_location="cpu").float()
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features = text_features / torch.norm(text_features, dim=1, keepdim=True)

        vlm_features = image_features @ text_features.T
        del image_features, text_features

    # Normalize VLM features
    mean = torch.mean(vlm_features)
    std = torch.std(vlm_features)
    vlm_features = (vlm_features - mean) / (std + NORM_EPSILON)
    
    # Branch by mode
    if args.mode == "specific":
        # Task-specific: compute class averages, select top concepts per class
        # Load dataset to get class information
        data = data_utils.get_data(d_train)
        _, start_indices, counts = np.unique(data.targets, return_index=True, return_counts=True)
        
        print("Computing class-wise concept activations...")
        class_averages = compute_class_averages(vlm_features, (start_indices, counts))
        top_concepts = select_top_concepts(class_averages, args.top_concepts)
        print(f"Computing entropy over top {args.top_concepts} concepts per class...")
        entropy_vals = compute_entropy(top_concepts)
        


    else:
        # Task-agnostic: per-image entropy over top-k logits
        vlm_np = vlm_features.numpy()
        k = args.top_k if args.top_k > 0 else vlm_np.shape[1]
        top_concepts = select_top_concepts(vlm_np, k)
        entropy_vals = compute_entropy(top_concepts)


    print("\n" + "="*50)
    print(f"Concept set: {args.concept_set}")
    print("="*50)
    print(f"Entropy statistics (over {args.top_concepts} top concepts per class):")
    print(f"  Min entropy:  {np.min(entropy_vals):.4f}")
    print(f"  Max entropy:  {np.max(entropy_vals):.4f}")
    print(f"  Mean entropy: {np.mean(entropy_vals):.4f}")
    print("="*50)


if __name__=='__main__':
    args = parser.parse_args()
    calculate_entropy(args)
