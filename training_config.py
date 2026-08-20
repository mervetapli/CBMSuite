"""
Unified training configuration for CBM models.

Consolidates training hyperparameters across different datasets and model architectures.
Supports different learning rate schedules for projection layer, teacher model, and final layer.
"""

from dataclasses import dataclass
from typing import Optional

# Dataset-specific training defaults
DATASET_TRAINING_DEFAULTS = {
    "cifar10": {
        "proj_lr": 1e-4,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-3,
        "teacher_scheduler_t_max": 50,
        "final_lr": 1e-3,
        "final_scheduler_t_max": 20,
        "distill_temp": 2.0,
        "distill_weight": 1.0,
    },
    "cifar100": {
        "proj_lr": 1e-4,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-3,
        "teacher_scheduler_t_max": 50,
        "final_lr": 1e-3,
        "final_scheduler_t_max": 20,
        "distill_temp": 2.0,
        "distill_weight": 1.0,
    },
    "cub": {
        "proj_lr": 1e-4,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-3,
        "teacher_scheduler_t_max": 20,
        "final_lr": 1e-3,
        "final_scheduler_t_max": 10,
        "distill_temp": 4.0,
        "distill_weight": 0.5,
    },
    "imagenet_100": {
        "proj_lr": 1e-3,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-3,
        "teacher_scheduler_t_max": 50,
        "final_lr": 1e-4,
        "final_scheduler_t_max": 10,
        "distill_temp": 2.0,
        "distill_weight": 1.0,
    },
    # No old_trains/*.py script ever targeted full (1000-class) ImageNet --
    # these are just imagenet_100's values reused as a starting point, not
    # independently tuned for full ImageNet.
    "imagenet": {
        "proj_lr": 1e-3,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-3,
        "teacher_scheduler_t_max": 50,
        "final_lr": 1e-4,
        "final_scheduler_t_max": 10,
        "distill_temp": 2.0,
        "distill_weight": 1.0,
    },
    "places365": {
        "proj_lr": 1e-3,
        "proj_scheduler_t_max": 20,
        "teacher_lr": 1e-4,
        "teacher_scheduler_t_max": 10,
        "final_lr": 1e-3,
        "final_scheduler_t_max": 10,
        "distill_temp": 2.0,
        "distill_weight": 2.0,
    },
}


@dataclass
class OptimizerConfig:
    """Configuration for a single optimizer."""
    learning_rate: float
    scheduler_type: str = "cosine"  # 'cosine', 'constant', None
    scheduler_t_max: Optional[int] = None  # For cosine annealing
    scheduler_eta_min: float = 0.0001
    optimizer_type: str = "adam"  # 'adam', 'sgd'


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Dataset and model configuration
    dataset: str
    backbone: str
    vlm_name: str
    
    # Optimizer configurations for different components
    projection_optimizer: OptimizerConfig
    teacher_optimizer: OptimizerConfig
    final_layer_optimizer: OptimizerConfig
    
    # Batch sizes
    batch_size: int = 512
    final_batch_size: int = 256
    proj_batch_size: int = 2048
    
    # Training iterations
    proj_steps: int = 1000
    n_iters: int = 1000
    
    # Regularization
    lam: float = 0.0007

    # Final-layer distillation: softmax temperature and the weight on the
    # distillation loss term relative to the elastic-net term.
    distill_temp: float = 4.0
    distill_weight: float = 0.5

    # Paths
    activation_dir: str = "saved_activations"
    save_dir: str = "saved_models"
    feature_layer: str = "layer4"
    concept_set: Optional[str] = None
    num_workers: int = 8

    # Device
    device: str = "cuda"
    
    # Seeds
    seed: Optional[int] = None
    random_seed: Optional[int] = None
    
    @staticmethod
    def from_dataset(dataset: str, **kwargs) -> "TrainingConfig":
        """
        Create training config from dataset name using defaults.
        
        Args:
            dataset: Dataset name (e.g., 'cifar10', 'cub', 'imagenet_100')
            **kwargs: Override specific parameters
            
        Returns:
            TrainingConfig instance
        """
        if dataset not in DATASET_TRAINING_DEFAULTS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_TRAINING_DEFAULTS.keys())}")
        
        defaults = DATASET_TRAINING_DEFAULTS[dataset].copy()
        
        # Create optimizer configs
        projection_optimizer = OptimizerConfig(
            learning_rate=defaults.pop("proj_lr"),
            scheduler_t_max=defaults.pop("proj_scheduler_t_max"),
        )
        
        teacher_optimizer = OptimizerConfig(
            learning_rate=defaults.pop("teacher_lr"),
            scheduler_t_max=defaults.pop("teacher_scheduler_t_max"),
        )
        
        final_layer_optimizer = OptimizerConfig(
            learning_rate=defaults.pop("final_lr"),
            scheduler_t_max=defaults.pop("final_scheduler_t_max"),
        )
        
        # Merge remaining defaults with kwargs
        config_dict = {
            "dataset": dataset,
            "projection_optimizer": projection_optimizer,
            "teacher_optimizer": teacher_optimizer,
            "final_layer_optimizer": final_layer_optimizer,
            **defaults,
            **kwargs,
        }
        
        return TrainingConfig(**config_dict)
    
    def get_optimizer(self, 
                     component: str,
                     parameters,
                     ) -> tuple:
        """
        Create optimizer and scheduler for a component.
        
        Args:
            component: 'projection', 'teacher', or 'final_layer'
            parameters: Model parameters to optimize
            
        Returns:
            Tuple of (optimizer, scheduler or None)
        """
        if component == "projection":
            config = self.projection_optimizer
        elif component == "teacher":
            config = self.teacher_optimizer
        elif component == "final_layer":
            config = self.final_layer_optimizer
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Create optimizer
        if config.optimizer_type.lower() == "adam":
            optimizer = __import__("torch").optim.Adam(
                parameters,
                lr=config.learning_rate
            )
        elif config.optimizer_type.lower() == "sgd":
            optimizer = __import__("torch").optim.SGD(
                parameters,
                lr=config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
        
        # Create scheduler
        scheduler = None
        if config.scheduler_type and config.scheduler_type.lower() == "cosine":
            if config.scheduler_t_max is None:
                raise ValueError(f"scheduler_t_max required for cosine annealing")
            scheduler = __import__("torch").optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.scheduler_t_max,
                eta_min=config.scheduler_eta_min
            )
        
        return optimizer, scheduler
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "vlm_name": self.vlm_name,
            "proj_lr": self.projection_optimizer.learning_rate,
            "teacher_lr": self.teacher_optimizer.learning_rate,
            "final_lr": self.final_layer_optimizer.learning_rate,
            "lam": self.lam,
            "distill_temp": self.distill_temp,
            "distill_weight": self.distill_weight,
        }
