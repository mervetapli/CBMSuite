"""
Unified command-line argument parser for CBM training.

Consolidates all training arguments across different train scripts,
with support for automatic dataset-based configuration loading.
"""

import argparse
from typing import Optional
from training_config import TrainingConfig, DATASET_TRAINING_DEFAULTS


def create_training_parser() -> argparse.ArgumentParser:
    """
    Create unified argument parser for CBM training.
    
    Returns:
        ArgumentParser instance with all training arguments
    """
    parser = argparse.ArgumentParser(
        description='Unified training script for Concept Bottleneck Models'
    )

    # ========================================================================
    # Dataset and Model Arguments
    # ========================================================================
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (cifar10, cifar100, cub, imagenet_100, places365)"
    )
    parser.add_argument(
        "--concept_set",
        type=str,
        default=None,
        help="Path to concept set file. If None, uses default for dataset"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="clip_RN50",
        help="Backbone model (clip_RN50, resnet18, resnet50, etc.)"
    )
    parser.add_argument(
        "--vlm_name",
        type=str,
        default="ViT-B/16",
        help="Vision-language model (ViT-B/16, ViT-L/14, SAIL, SigLIP, FLAIR)"
    )

    # ========================================================================
    # Training Configuration
    # ========================================================================
    parser.add_argument(
        "--use_dataset_defaults",
        action="store_true",
        default=True,
        help="Load learning rates and scheduler settings from dataset defaults"
    )
    
    # Projection layer optimization
    parser.add_argument(
        "--proj_lr",
        type=float,
        default=None,
        help="Projection layer learning rate (overrides dataset default)"
    )
    parser.add_argument(
        "--proj_scheduler_t_max",
        type=int,
        default=None,
        help="Projection layer scheduler T_max (overrides dataset default)"
    )
    
    # Teacher model optimization
    parser.add_argument(
        "--teacher_lr",
        type=float,
        default=None,
        help="Teacher model learning rate (overrides dataset default)"
    )
    parser.add_argument(
        "--teacher_scheduler_t_max",
        type=int,
        default=None,
        help="Teacher model scheduler T_max (overrides dataset default)"
    )
    
    # Final layer optimization
    parser.add_argument(
        "--final_lr",
        type=float,
        default=None,
        help="Final layer learning rate (overrides dataset default)"
    )
    parser.add_argument(
        "--final_scheduler_t_max",
        type=int,
        default=None,
        help="Final layer scheduler T_max (overrides dataset default)"
    )
    
    # ========================================================================
    # Batch Sizes
    # ========================================================================
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for loading activations"
    )
    parser.add_argument(
        "--final_batch_size",
        type=int,
        default=256,
        help="Batch size for teacher/final layer fitting (SAGA)"
    )
    parser.add_argument(
        "--proj_batch_size",
        type=int,
        default=2048,
        help="Batch size for projection layer training"
    )

    # ========================================================================
    # Training Iterations and Hyperparameters
    # ========================================================================
    parser.add_argument(
        "--proj_steps",
        type=int,
        default=1000,
        help="Number of steps to train projection layer"
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100,
        help="Number of iterations for final layer solver"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.0001,
        help="Sparsity regularization parameter"
    )
    parser.add_argument(
        "--distill_temp",
        type=float,
        default=None,
        help="Softmax temperature for final-layer distillation (overrides dataset default)"
    )
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=None,
        help="Weight on the distillation loss term (overrides dataset default)"
    )

    # ========================================================================
    # Paths
    # ========================================================================
    parser.add_argument(
        "--activation_dir",
        type=str,
        default="saved_activations",
        help="Directory to save activations"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--feature_layer",
        type=str,
        default="layer4",
        help="Feature layer to extract from backbone"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes for activation saving (lower this on machines with limited RAM)"
    )

    # ========================================================================
    # Execution
    # ========================================================================
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PyTorch random seed"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Python random seed"
    )

    return parser


def get_training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Create TrainingConfig from command-line arguments.
    
    Loads dataset defaults first, then overrides with command-line arguments.
    
    Args:
        args: Parsed arguments from create_training_parser()
        
    Returns:
        TrainingConfig instance
    """
    # Build override kwargs
    overrides = {
        "backbone": args.backbone,
        "vlm_name": args.vlm_name,
        "batch_size": args.batch_size,
        "final_batch_size": args.final_batch_size,
        "proj_batch_size": args.proj_batch_size,
        "proj_steps": args.proj_steps,
        "n_iters": args.n_iters,
        "lam": args.lam,
        "activation_dir": args.activation_dir,
        "save_dir": args.save_dir,
        "feature_layer": args.feature_layer,
        "num_workers": args.num_workers,
        "device": args.device,
        "seed": args.seed,
        "random_seed": args.random_seed
    }
    
    if args.concept_set is not None:
        overrides["concept_set"] = args.concept_set
    if args.distill_temp is not None:
        overrides["distill_temp"] = args.distill_temp
    if args.distill_weight is not None:
        overrides["distill_weight"] = args.distill_weight

    # Create config from dataset defaults + overrides
    config = TrainingConfig.from_dataset(args.dataset, **overrides)

    # Learning rate / scheduler overrides apply to the nested OptimizerConfig
    # objects directly, since TrainingConfig has no flat fields for them.
    if args.proj_lr is not None:
        config.projection_optimizer.learning_rate = args.proj_lr
    if args.proj_scheduler_t_max is not None:
        config.projection_optimizer.scheduler_t_max = args.proj_scheduler_t_max

    if args.teacher_lr is not None:
        config.teacher_optimizer.learning_rate = args.teacher_lr
    if args.teacher_scheduler_t_max is not None:
        config.teacher_optimizer.scheduler_t_max = args.teacher_scheduler_t_max

    if args.final_lr is not None:
        config.final_layer_optimizer.learning_rate = args.final_lr
    if args.final_scheduler_t_max is not None:
        config.final_layer_optimizer.scheduler_t_max = args.final_scheduler_t_max

    return config


# Simple example usage
if __name__ == "__main__":
    parser = create_training_parser()
    args = parser.parse_args()
    config = get_training_config_from_args(args)
    print("Training Configuration:")
    print(config.to_dict())
