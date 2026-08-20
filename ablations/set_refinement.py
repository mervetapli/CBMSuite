"""
Ablation: validates the concept-set entropy metric (see goodness_of_concepts.py)
by injecting random/nonsense concepts into a real concept set and checking
whether entropy-guided pruning (prune_n_concepts) preferentially removes the
injected concepts rather than the original ones -- as opposed to a random-
removal baseline, which should remove real and injected concepts at chance
rate.
"""
import random
import argparse

import torch
import numpy as np

import utils
import data_utils
from goodness_of_concepts import compute_entropy

# Constants
CHUNK_SIZE = 10          # concepts removed per pruning iteration
RANDOM_FRACTION = 0.5    # fraction of the merged concept set that is random/nonsense


parser = argparse.ArgumentParser(description='Ablation: validate concept-set entropy via random-concept injection + pruning')

parser.add_argument("--dataset", type=str, default="imagenet_100")
parser.add_argument("--concept_set", type=str, default=None,
                    help="path to concept set name")
parser.add_argument("--random_concept_set", type=str, default="data/concept_sets/random_strings.txt",
                    help="path to the nonsense/random concept set injected for validation")
parser.add_argument("--vlm_name", type=str, default="ViT-B/16", help="Which vision-language model to use")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--feature_layer", type=str, default='layer4',
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                    help="how many concepts to remove per pruning iteration")
parser.add_argument("--random_fraction", type=float, default=RANDOM_FRACTION,
                    help="fraction of the merged concept set that is random/nonsense (injected for validation)")


def prune_n_concepts(act_matrix, n, keep, tol=0.0):
    """
    Greedily remove up to n concepts (columns) from `keep`, each time
    dropping whichever remaining concept's removal decreases (or least
    increases) the mean per-sample entropy of act_matrix. Stops early if no
    remaining concept's removal decreases entropy below `tol`.
    """
    i = 0
    while i < n:
        act_matrix_sub = act_matrix[:, keep]
        base_H = np.mean(compute_entropy(act_matrix_sub))

        deltas = []
        for idx in range(len(keep)):
            act_matrix_removed = np.delete(act_matrix_sub, idx, axis=1)
            new_H = np.mean(compute_entropy(act_matrix_removed))
            deltas.append(new_H - base_H)

        min_delta = min(deltas)
        if min_delta < tol:
            i += 1
            j_remove = keep[int(np.argmin(deltas))]
            keep.remove(j_remove)
        else:
            break

    return keep


def merge_with_random_concepts(text_features, random_text_features, random_fraction):
    """
    Merge a fraction of `random_text_features` into `text_features` to build
    a concept set that is part real, part nonsense.

    Real concepts occupy indices [0, k_real) and injected random concepts
    occupy [k_real, k_real + k_random) in the returned tensor.

    Returns:
        merged_text_features, k_real, k_random
    """
    n_total = text_features.size(0)
    k_random = int(n_total * random_fraction)
    k_real = n_total - k_random

    idx_real = torch.randperm(text_features.size(0))[:k_real]
    idx_random = torch.randperm(random_text_features.size(0))[:k_random]

    merged = torch.cat([text_features[idx_real], random_text_features[idx_random]], dim=0)
    merged = merged / torch.norm(merged, dim=1, keepdim=True)

    return merged, k_real, k_random


def fraction_random_removed(keep, k_real, k_random):
    """Fraction of the originally-injected random concepts that have been removed from `keep`."""
    if k_random == 0:
        return 0.0
    random_still_kept = sum(1 for j in keep if j >= k_real)
    return (k_random - random_still_kept) / k_random


def chunked(total, chunk_size):
    """Batch sizes summing to `total` (last batch may be smaller than chunk_size)."""
    n_full, remainder = divmod(total, chunk_size)
    sizes = [chunk_size] * n_full
    if remainder:
        sizes.append(remainder)
    return sizes


def run_set_refinement(args):
    with open(args.concept_set, "r") as f:
        concepts = [c for c in f.read().split("\n") if c.strip()]

    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    # Save activations (also saves backbone activations as a side effect; not used below)
    for d_probe in [d_train, d_val]:
        utils.save_activations(vlm_name=args.vlm_name, target_name=args.backbone,
                               target_layers=[args.feature_layer], d_probe=d_probe,
                               concept_set=args.concept_set, batch_size=args.batch_size,
                               device=args.device, pool_mode="avg", save_dir=args.activation_dir)

    _, clip_save_name, text_save_name = utils.get_save_paths(args.vlm_name, args.backbone,
                                            args.feature_layer, d_train, args.concept_set, "avg", args.activation_dir)
    _, _, random_text_save_name = utils.get_save_paths(args.vlm_name, args.backbone,
                                            args.feature_layer, d_train, args.random_concept_set, "avg", args.activation_dir)

    data = data_utils.get_data(d_train)
    _, start_index, count = np.unique(data.targets, return_index=True, return_counts=True)

    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        random_text_features = torch.load(random_text_save_name, map_location="cpu").float()

        merged_text_features, k_real, k_random = merge_with_random_concepts(
            text_features, random_text_features, args.random_fraction
        )

        clip_features = image_features @ merged_text_features.T
        del image_features, merged_text_features

    print("Min: {:.4f}  Max: {:.4f}  Mean: {:.4f}".format(
        torch.min(clip_features).item(), torch.max(clip_features).item(), torch.mean(clip_features).item()))

    clip_features = (clip_features - torch.mean(clip_features)) / torch.std(clip_features)
    clip_features = clip_features.numpy()

    class_avg = []
    for ind, c in zip(start_index, count):
        class_avg.append(np.mean(clip_features[ind:ind + c, :], axis=0))
    class_avg = np.array(class_avg)

    print("\n" + "=" * 50)
    print(f"Concept set: {args.concept_set}  "
          f"({k_real} real + {k_random} random, {args.random_fraction:.0%} injected)")
    print("=" * 50)

    baseline_entropy = compute_entropy(clip_features)
    print(f"Baseline entropy -- min: {np.min(baseline_entropy):.4f}, "
          f"max: {np.max(baseline_entropy):.4f}, mean: {np.mean(baseline_entropy):.4f}")

    # --- Baseline: remove concepts at random, chunk_size at a time ---
    print("\nRandom-removal baseline:")
    keep = list(range(class_avg.shape[1]))
    batches = chunked(k_random, args.chunk_size)
    for i, batch_size in enumerate(batches):
        removed = random.sample(keep, batch_size)
        keep = [j for j in keep if j not in removed]
        entr = compute_entropy(class_avg[:, keep])
        print(f"  step {i + 1}/{len(batches)}: mean entropy={np.mean(entr):.4f}, "
              f"random-concepts-removed-so-far={fraction_random_removed(keep, k_real, k_random):.2%}")

    # --- Entropy-guided pruning (validates that pruning targets the injected concepts) ---
    print("\nEntropy-guided pruning (prune_n_concepts):")
    keep = list(range(class_avg.shape[1]))
    for i, batch_size in enumerate(batches):
        keep = prune_n_concepts(class_avg, batch_size, keep)
        entr = compute_entropy(class_avg[:, keep])
        print(f"  step {i + 1}/{len(batches)}: mean entropy={np.mean(entr):.4f}, "
              f"random-concepts-removed-so-far={fraction_random_removed(keep, k_real, k_random):.2%}")

    print("\nIf entropy-guided pruning is a useful signal, its final "
          "random-concepts-removed fraction should be higher than the random-removal baseline's.")


if __name__ == '__main__':
    args = parser.parse_args()
    run_set_refinement(args)
