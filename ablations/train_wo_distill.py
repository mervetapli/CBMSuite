"""
Ablation: train a CBM without knowledge distillation.

This is a copy of train_cbm_suite.py with Stage 2 (teacher model) and the
distillation term in the final-layer loss removed. The final layer is fit
directly on concept activations with only the elastic-net loss; no teacher
model is trained, loaded, or saved.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import datetime
import json
import dataclasses
from tqdm import tqdm
from glm_saga.elasticnet import elastic_loss
from torch.utils.data import DataLoader, TensorDataset

import utils
import data_utils
from arg_parser import create_training_parser, get_training_config_from_args
from training_config import TrainingConfig


def train_concept_encoder(model, target_features, vlm_features, val_target_features, val_vlm_features, config: TrainingConfig):
    """
    Train the concept encoder (projection layer) to map backbone features to VLM features.

    Returns:
        model: Trained concept encoder model
        metrics: Dictionary containing best step and validation loss
    """
    opt, scheduler = config.get_optimizer("projection", model.parameters())
    loss_fn = torch.nn.MSELoss()

    indices = [ind for ind in range(len(target_features))]
    best_val_loss = float("inf")
    best_step = 0
    best_encoder_weights = None
    proj_batch_size = min(config.proj_batch_size, len(target_features))

    for i in range(config.proj_steps):
        model.train()
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = model(target_features[batch].to(config.device).detach())
        loss = loss_fn(outs, vlm_features[batch].to(config.device).detach())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if i % 50 == 0 or i == config.proj_steps - 1:
            with torch.no_grad():
                model.eval()
                val_output = model(val_target_features.to(config.device).detach())
                val_loss = loss_fn(val_output, val_vlm_features.to(config.device).detach())

            if i == 0:
                best_val_loss = val_loss
                best_step = i
                best_encoder_weights = model.state_dict()
                print("Step:{}, Avg train loss: {:.4f}, Avg val loss: {:.4f}".format(best_step, loss.cpu(), best_val_loss.cpu()))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_encoder_weights = model.state_dict()
            else:  # stop if val loss starts increasing
                break

    model.load_state_dict(best_encoder_weights)
    print("Best step:{}, Avg val loss:{:.4f}".format(best_step, best_val_loss.cpu()))

    return model, {"best_step": best_step, "best_val_loss": best_val_loss.item()}


def train_final_layer_no_distill(linear, train_loader, val_loader, config: TrainingConfig):
    """
    Train the final linear layer directly on concept activations.

    No teacher model and no distillation term: the loss is just the
    elastic-net loss on the concept -> class mapping.

    Returns:
        linear: Trained final layer model
        metrics: Dictionary containing best step, validation loss, and accuracy
    """
    ALPHA = 0.99
    fn_opt, scheduler = config.get_optimizer("final_layer", linear.parameters())

    best_val_accuracy = 0
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for t in tqdm(range(config.n_iters)):
        linear.train()
        total_loss = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)

            loss = elastic_loss(linear, x, y, config.lam, ALPHA)

            fn_opt.zero_grad()
            loss.backward()
            fn_opt.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            total += y.size(0)

        with torch.no_grad():
            linear.eval()
            val_loss = 0
            correct = 0
            total = 0

            for batch in val_loader:
                x, y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                out = linear(x)
                _, predicted = torch.max(out.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                val_loss += elastic_loss(linear, x, y, config.lam, ALPHA).item() * x.size(0)

            print('Ep: {}, loss: {:.4f}, acc: {:.4f}'.format(t, val_loss / total, correct / total))

            if t == 0:
                best_val_loss = val_loss
                best_val_accuracy = correct / total
                best_step = t
                best_weights = linear.state_dict()
                print("Step:{}, Avg train loss: {:.4f}, Avg val loss: {:.4f}".format(best_step, total_loss / len(train_loader.dataset), best_val_loss / total))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = correct / total
                best_step = t
                best_weights = linear.state_dict()

    linear.load_state_dict(best_weights)
    print("Best step:{}, Avg val loss:{:.4f}, Val Acc:{:.4f}".format(best_step, best_val_loss / total, best_val_accuracy))

    return linear, {"best_step": best_step, "best_val_loss": best_val_loss / total, "best_val_accuracy": best_val_accuracy}


def train_cbm_wo_distill_and_save(config: TrainingConfig):
    """
    Ablation pipeline: concept encoder -> final layer (no teacher, no distillation).
    """
    if config.seed is not None:
        torch.manual_seed(config.seed)
    if config.random_seed is not None:
        random.seed(config.random_seed)

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    if config.concept_set is None:
        config.concept_set = "data/concept_sets/{}_filtered.txt".format(config.dataset)

    d_train = config.dataset + "_train"
    d_val = config.dataset + "_val"

    # Get concept and class sets
    cls_file = data_utils.LABEL_FILES[config.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    with open(config.concept_set) as f:
        concepts = f.read().split("\n")

    # Save activations from backbone and VLM
    print("Saving activations...")
    for d_probe in [d_train, d_val]:
        utils.save_activations(vlm_name=config.vlm_name, target_name=config.backbone,
                               target_layers=[config.feature_layer], d_probe=d_probe,
                               concept_set=config.concept_set, batch_size=config.batch_size,
                               device=config.device, pool_mode="avg", save_dir=config.activation_dir,
                               num_workers=config.num_workers)

    # Get save paths
    target_save_name, clip_save_name, text_save_name = utils.get_save_paths(config.vlm_name, config.backbone,
                                            config.feature_layer, d_train, config.concept_set, "avg", config.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name = utils.get_save_paths(config.vlm_name, config.backbone,
                                            config.feature_layer, d_val, config.concept_set, "avg", config.activation_dir)

    print("Loading activations from: ", clip_save_name, text_save_name)

    # Load and normalize features
    print("Loading and normalizing features...")
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()

        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        vlm_features = image_features @ text_features.T
        val_vlm_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features

    # Normalize VLM features
    P_mean = torch.mean(vlm_features)
    P_std = torch.std(vlm_features)

    vlm_features = (vlm_features - P_mean) / P_std
    val_vlm_features = (val_vlm_features - P_mean) / P_std

    # ========== STAGE 1: Train Concept Encoder ==========
    print("\n" + "="*50)
    print("STAGE 1: Training Concept Encoder")
    print("="*50)

    proj_layer = torch.nn.Sequential(
        torch.nn.Linear(target_features.shape[1], 2 * target_features.shape[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * target_features.shape[1], len(concepts))
    ).to(config.device)

    proj_layer, encoder_metrics = train_concept_encoder(proj_layer, target_features, vlm_features,
                                                         val_target_features, val_vlm_features, config)

    # Generate concept predictions
    print("\nGenerating concept predictions...")
    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)

    proj_layer = proj_layer.to('cpu')
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())

        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)

        train_ds = TensorDataset(train_c, train_y)
        val_ds = TensorDataset(val_c, val_y)

    # Create dataloaders (no teacher features needed -- nothing to distill from)
    train_loader = DataLoader(train_ds, batch_size=config.final_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.final_batch_size, shuffle=False)

    # ========== STAGE 2: Train Final Layer (no teacher, no distillation) ==========
    print("\n" + "="*50)
    print("STAGE 2: Training Final Layer (no distillation)")
    print("="*50)

    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(config.device)
    linear, final_metrics = train_final_layer_no_distill(linear, train_loader, val_loader, config)

    # ========== Save Everything ==========
    print("\n" + "="*50)
    print("Saving Model and Artifacts")
    print("="*50)

    save_name = "{}/{}_{}_wo_distill".format(config.save_dir, config.backbone, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.makedirs(save_name, exist_ok=True)

    # Save normalizations
    torch.save(P_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(P_std, os.path.join(save_name, "proj_std.pt"))

    # Save model weights
    torch.save(proj_layer.state_dict(), os.path.join(save_name, "W_c.pt"))
    torch.save(linear.weight, os.path.join(save_name, "W_g.pt"))
    torch.save(linear.bias, os.path.join(save_name, "b_g.pt"))

    # Save concepts
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n' + concept)

    # Save the full training configuration (cbm.py's load_cbm reads "backbone" back out of this file)
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    # Save training metrics
    metrics = {
        "encoder": encoder_metrics,
        "final_layer": final_metrics
    }
    with open(os.path.join(save_name, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {save_name}")
    print(f"Final Validation Accuracy: {final_metrics['best_val_accuracy']:.4f}")


if __name__=='__main__':
    parser = create_training_parser()
    args = parser.parse_args()
    config = get_training_config_from_args(args)
    train_cbm_wo_distill_and_save(config)
