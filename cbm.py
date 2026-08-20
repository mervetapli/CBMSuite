import os
import json
import torch
import data_utils


def _get_backbone(backbone_name, feature_layer, device):
    """
    Build a callable backbone(x) -> flat [B, D] features, matching how
    utils.py's save_target_activations extracted activations at training
    time (a forward hook on `feature_layer`, average-pooled if spatial).

    Args:
        backbone_name: Name of the backbone model
        feature_layer: Dotted attribute path to the layer activations were
            hooked from during training (e.g. "layer4", "features.final_pool").
            Ignored for backbones with their own explicit extraction below.
        device: Device to load model to

    Returns:
        backbone: Callable that returns flat [B, D] features for input x
    """
    model, _ = data_utils.get_target_model(backbone_name, device)

    if backbone_name.startswith("clip_"):
        # TargetModelLoader._load_clip already returns a lambda wrapping
        # .encode_image(...).float() -- no hook needed, feature_layer doesn't apply.
        return model

    if "franca" in backbone_name or "perception_encoder" in backbone_name:
        # Handled explicitly in CBM_model.forward() (matching utils.py's own
        # special-casing for these two); feature_layer doesn't apply.
        return model

    # Generic case (plain torchvision models, dinov2, resnet18_cub, ...):
    # register a forward hook on the named layer, same as training time.
    captured = {}

    def hook(_module, _input, output):
        captured["feat"] = output.mean(dim=[2, 3]) if output.dim() == 4 else output

    layer = model
    for attr in feature_layer.split("."):
        layer = getattr(layer, attr)
    layer.register_forward_hook(hook)

    def backbone(x):
        model(x)
        return captured["feat"]

    return backbone


class CBM_model(torch.nn.Module):
    """
    Concept Bottleneck Model that projects features to concept space, then predicts classes.
    
    Architecture:
    1. Backbone: Extract features from input
    2. Projection layer: Project features to concept space
    3. Final layer: Predict classes from concepts
    """
    
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda", feature_layer="layer4"):
        """
        Initialize CBM model.

        Args:
            backbone_name: Name of the backbone model
            W_c: Projection layer weights (state dict)
            W_g: Final layer weights
            b_g: Final layer bias
            proj_mean: Mean for concept normalization
            proj_std: Std for concept normalization
            device: Device to load model to
            feature_layer: Layer backbone activations were hooked from during
                training (see _get_backbone); ignored for CLIP/franca/perception_encoder
        """
        super().__init__()

        self.backbone = _get_backbone(backbone_name, feature_layer, device)
        self.backbone_name = backbone_name
        
        # Extract dimensions from projection layer weights
        first_layer_weight = W_c["0.weight"]
        input_dim = first_layer_weight.shape[1]
        hidden_dim = first_layer_weight.shape[0]
        output_dim = W_g.shape[1]

        # Build projection layer with optional ReLU if present in W_c
        has_relu = "2.weight" in W_c
        if has_relu:
            self.proj_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            ).to(device)
        else:
            self.proj_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Linear(hidden_dim, output_dim)
            ).to(device)
        self.proj_layer.load_state_dict(W_c)
        
        # Concept normalization parameters
        self.register_buffer('proj_mean', proj_mean)
        self.register_buffer('proj_std', proj_std)
        
        # Final layer (concepts -> logits)
        self.final = torch.nn.Linear(output_dim, W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})

        self.concepts = None
        
    def forward(self, x):
        """
        Forward pass: extract features -> project to concepts -> predict classes.
        
        Args:
            x: Input images
            
        Returns:
            logits: Class predictions
            concepts: Normalized concept vectors
        """
        # Backbone-specific feature extraction
        if 'perception_encoder' in self.backbone_name:
            x = self.backbone.encode_image(x)
        elif 'franca' in self.backbone_name:
            features = self.backbone(x)
            intermediate_output = features[-4:]
            x = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
        else:
            x = self.backbone(x)
        
        x = torch.flatten(x, 1)
        
        # Project to concept space
        x = self.proj_layer(x)
        concepts = (x - self.proj_mean) / self.proj_std
        
        # Predict classes
        logits = self.final(x)
        
        return logits, concepts




def load_cbm(load_dir, device="cuda"):
    """
    Load a trained CBM model from directory.
    
    Args:
        load_dir: Directory containing saved model artifacts
        device: Device to load model to (default: 'cuda')
        
    Returns:
        CBM_model: Loaded and initialized model
    """
    # Load configuration
    with open(os.path.join(load_dir, "args.txt"), 'r') as f:
        args = json.load(f)

    # Load model weights and parameters
    W_c = torch.load(os.path.join(load_dir, "W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    # Create and return model. Fall back to the historical default for
    # checkpoints saved before feature_layer was recorded in args.txt.
    feature_layer = args.get('feature_layer', 'layer4')
    model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device, feature_layer)
    model.eval()

    # Load concept names, if saved alongside the model
    concepts_path = os.path.join(load_dir, "concepts.txt")
    if os.path.exists(concepts_path):
        with open(concepts_path, "r") as f:
            model.concepts = f.read().split("\n")

    return model
