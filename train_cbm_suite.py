import torch
import os
import random
import utils
import data_utils
import datetime
import json
import dataclasses
from tqdm import tqdm
from glm_saga.elasticnet import IndexedTensorDataset, elastic_loss
from torch.utils.data import DataLoader, TensorDataset

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



def train_teacher_model(model, train_loader, val_loader, config: TrainingConfig):
    """
    Train the teacher model on backbone features.

    Returns:
        model: Trained teacher model
        metrics: Dictionary containing best step, validation loss, and accuracy
    """
    t_opt, scheduler_t = config.get_optimizer("teacher", model.parameters())
    t_loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for t in tqdm(range(config.n_iters)):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            out = model(x)
            loss = t_loss_fn(out, y)

            t_opt.zero_grad()
            loss.backward()
            t_opt.step()
            if scheduler_t is not None:
                scheduler_t.step()
            total_loss += loss.item() * x.size(0)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            for batch in val_loader:
                x, y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                val_loss += t_loss_fn(out, y).item() * x.size(0)

            print('Ep: {}, loss: {:.4f}, acc: {:.4f}'.format(t, val_loss / total, correct / total))

            if t == 0:
                best_val_loss = val_loss
                best_step = t
                best_val_accuracy = correct / total
                best_weights = model.state_dict()
                print("Step:{}, Avg train loss: {:.4f}, Avg val loss: {:.4f}".format(best_step, total_loss / len(train_loader.dataset), best_val_loss / total))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = correct / total
                best_step = t
                best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    print("Best step:{}, Avg val loss:{:.4f}, Val Acc:{:.4f}".format(best_step, best_val_loss / total, best_val_accuracy))

    return model, {"best_step": best_step, "best_val_loss": best_val_loss / total, "best_val_accuracy": best_val_accuracy}

def train_final_layer(linear, linear_pr, indexed_train_loader, train_z, train_c, val_loader, config: TrainingConfig):
    """
    Train the final linear layer with distillation from teacher model.

    Returns:
        linear: Trained final layer model
        metrics: Dictionary containing best step, validation loss, and accuracy
    """
    ALPHA = 0.99
    fn_opt, scheduler = config.get_optimizer("final_layer", linear.parameters())
    loss_soft = torch.nn.KLDivLoss(reduction='batchmean')

    temp = config.distill_temp
    best_val_accuracy = 0
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for t in tqdm(range(config.n_iters)):
        linear.train()
        linear_pr.eval()
        total_loss = 0
        term1_tot = 0
        term2_tot = 0
        total = 0

        for batch in indexed_train_loader:
            x, y, idx = batch
            x = x.to(config.device)
            y = y.to(config.device)
            out = linear(x)

            teacher_logits = linear_pr(train_z[idx].to(config.device))
            student_soft_targets = torch.nn.functional.log_softmax(out / temp, dim=-1)
            teacher_soft_targets = torch.nn.functional.softmax(teacher_logits / temp, dim=-1)

            term1 = elastic_loss(linear, x, y, config.lam, ALPHA)
            term2 = loss_soft(student_soft_targets, teacher_soft_targets)
            loss = term1 + config.distill_weight * term2 * temp ** 2

            fn_opt.zero_grad()
            loss.backward()
            fn_opt.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            term1_tot += term1.item() * x.size(0)
            term2_tot += term2.item() * x.size(0)
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
                print("Step:{}, Avg train loss: {:.4f}, Avg val loss: {:.4f}".format(best_step, total_loss / len(indexed_train_loader.dataset), best_val_loss / total))
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = correct / total
                best_step = t
                best_weights = linear.state_dict()

    linear.load_state_dict(best_weights)
    print("Best step:{}, Avg val loss:{:.4f}, Val Acc:{:.4f}".format(best_step, best_val_loss / total, best_val_accuracy))

    return linear, {"best_step": best_step, "best_val_loss": best_val_loss / total, "best_val_accuracy": best_val_accuracy}




def train_cbm_and_save(config: TrainingConfig):
    """
    Main pipeline to train the CBM model: concept encoder -> teacher model -> final layer.
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

        # Prepare datasets. Only the *training* set for the final layer needs
        # per-sample indices (to look up train_z[idx] for teacher-logit
        # distillation) -- everything else is a plain (x, y) TensorDataset.
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)
        val_ds = TensorDataset(val_c, val_y)

        train_z = target_features.detach()
        val_z = val_target_features.detach()

        train_teacher_ds = TensorDataset(train_z, train_y)
        val_teacher_ds = TensorDataset(val_z, val_y)

    # Create dataloaders
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=config.final_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.final_batch_size, shuffle=False)

    train_teacher_loader = DataLoader(train_teacher_ds, batch_size=config.final_batch_size, shuffle=True)
    val_teacher_loader = DataLoader(val_teacher_ds, batch_size=config.final_batch_size, shuffle=False)

    # ========== STAGE 2: Train or Load Teacher Model ==========
    print("\n" + "="*50)
    print("STAGE 2: Training Teacher Model")
    print("="*50)

    linear_pr = torch.nn.Linear(train_z.shape[1], len(classes)).to(config.device)
    t_model_path = 'teacher_models/{}_{}.pth'.format(config.dataset, config.backbone)

    if os.path.exists(t_model_path):
        print(f"Loading pretrained teacher model from: {t_model_path}")
        linear_pr.load_state_dict(torch.load(t_model_path, map_location=config.device))
    else:
        print(f"Training teacher model (will save to {t_model_path})...")
        linear_pr, teacher_metrics = train_teacher_model(linear_pr, train_teacher_loader,
                                                         val_teacher_loader, config)
        os.makedirs(os.path.dirname(t_model_path), exist_ok=True)
        torch.save(linear_pr.state_dict(), t_model_path)

    # ========== STAGE 3: Train Final Layer ==========
    print("\n" + "="*50)
    print("STAGE 3: Training Final Layer with Distillation")
    print("="*50)

    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(config.device)
    linear, final_metrics = train_final_layer(linear, linear_pr, indexed_train_loader,
                                              train_z, train_c, val_loader, config)

    # ========== Save Everything ==========
    print("\n" + "="*50)
    print("Saving Model and Artifacts")
    print("="*50)

    save_name = "{}/{}_{}".format(config.save_dir, config.backbone, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
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

    # Save the full training configuration (also used by cbm.py's load_cbm,
    # which reads the "backbone" key back out of this file).
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
    train_cbm_and_save(config)
