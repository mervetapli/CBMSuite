"""
Evaluate a trained CBM (as saved by train_cbm_suite.py / ablations/*.py):
overall accuracy, how many concepts the final layer actually uses, and which
concepts are most associated with each predicted class.
"""
import os
import json
import argparse

import torch

import cbm
import data_utils
import utils

DEFAULT_TOP_K_CONCEPTS = 5


parser = argparse.ArgumentParser(description='Evaluate a trained CBM')
parser.add_argument("--model_dir", type=str, required=True,
                    help="Directory containing a trained CBM (as saved by train_cbm_suite.py)")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=250, help="Batch size for evaluation")
parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
parser.add_argument("--top_k_concepts", type=int, default=DEFAULT_TOP_K_CONCEPTS,
                    help="How many top concepts to show per predicted class")


def evaluate(args):
    with open(os.path.join(args.model_dir, "args.txt"), "r") as f:
        train_args = json.load(f)
    dataset = train_args["dataset"]
    backbone = train_args["backbone"]

    print(f"Loading model from {args.model_dir}  (dataset={dataset}, backbone={backbone})")
    model = cbm.load_cbm(args.model_dir, args.device)

    _, preprocess = data_utils.get_target_model(backbone, args.device)
    val_data = data_utils.get_data(dataset + "_val", preprocess=preprocess)

    with open(data_utils.LABEL_FILES[dataset], "r") as f:
        classes = f.read().split("\n")

    print("\n" + "=" * 50)
    print("Accuracy")
    print("=" * 50)
    accuracy = utils.get_accuracy_cbm(model, val_data, args.device, args.batch_size, args.num_workers)
    print(f"Val accuracy: {accuracy * 100:.2f}%")

    print("\n" + "=" * 50)
    print("Concept usage")
    print("=" * 50)
    weight_contribs = torch.sum(torch.abs(model.final.weight), dim=0)
    used = torch.sum(weight_contribs > 1e-5).item()
    print(f"Concepts with non-zero outgoing weight: {used}/{len(weight_contribs)}")

    if model.concepts is not None:
        print("\n" + "=" * 50)
        print(f"Top {args.top_k_concepts} concepts per predicted class")
        print("=" * 50)
        concept_acts_by_pred = utils.get_concept_activations_by_prediction(
            model, val_data, args.device, args.batch_size, args.num_workers
        )
        k = min(args.top_k_concepts, len(model.concepts))
        for class_idx in range(concept_acts_by_pred.shape[0]):
            top = torch.topk(concept_acts_by_pred[class_idx].abs(), k=k)
            names = [model.concepts[i] for i in top.indices.tolist()]
            class_name = classes[class_idx] if class_idx < len(classes) else str(class_idx)
            print(f"{class_name}: {', '.join(names)}")


if __name__ == '__main__':
    args = parser.parse_args()
    evaluate(args)
