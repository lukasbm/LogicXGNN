import argparse
import sys
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import GINConv, MessagePassing
from typing import Optional

from load_data import load_data
from utils import test
from explainer_pipeline import run_explainer_pipeline


def try_to_load_model_from_checkpoint_path(
        checkpoint_path: str, device: torch.device
) -> torch.nn.Module:
    from graph_learning.utils import load_checkpoint as load_model_checkpoint
    from graph_learning.utils.packaging import load_model_from_bundle

    # try multiple loading methods
    attempts = {
        "torch package bundle": lambda: load_model_from_bundle(
            checkpoint_path, map_location=str(device)
        ),
        "Bare Torch Load": lambda: torch.load(
            checkpoint_path, map_location=device, weights_only=False
        ),
        "Cloudpickle Custom Loader": lambda: load_model_checkpoint(
            checkpoint_path, device=str(device)
        ),
    }

    # Try to import ArchTestingModule if available - for .ckpt files
    try:
        from scripts.arch_study import ArchTestingModule
        attempts = {
            "ArchTestingModule": lambda: ArchTestingModule.load_from_checkpoint(
                checkpoint_path=checkpoint_path, map_location=device
            ).model,
            **attempts
        }
    except (ImportError, AttributeError):
        pass  # Skip this loading method if not available

    # Also try PyTorch Lightning checkpoint loading if available
    try:
        import pytorch_lightning as pl
        # Try loading as a Lightning checkpoint and extracting the model
        def load_from_lightning_ckpt():
            ckpt = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in ckpt:
                # This is a Lightning checkpoint - need to reconstruct the model
                # For now, skip this as we'd need model architecture info
                raise ValueError("Lightning checkpoint detected but model reconstruction not implemented")
            raise ValueError("Not a Lightning checkpoint")

        attempts = {"Lightning Checkpoint": load_from_lightning_ckpt, **attempts}
    except (ImportError, ValueError):
        pass

    last_exc = None
    for attempt_name, attempt in attempts.items():
        try:
            print(f"Trying to load model via: {attempt_name}")
            result = attempt()
            # Validate that we got a real model
            if not isinstance(result, torch.nn.Module):
                raise ValueError(f"Got {type(result)} instead of torch.nn.Module")
            print(f"Successfully loaded model via: {attempt_name}")
            break
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            last_exc = e
    else:
        raise last_exc

    model: torch.nn.Module = result
    return model


def fix_model(model: torch.nn.Module) -> torch.nn.Module:
    # normalize model architecture ... add needed fields
    # basically a stupid monkey path
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, GINConv):
                module.in_channels = module.nn[0].in_features
                module.out_channels = module.nn[-1].out_features
    return model


def load_model(path: str, device: torch.device = torch.device('cpu')) -> Optional[torch.nn.Module]:
    """Load model from a specific checkpoint path."""
    try:
        model = try_to_load_model_from_checkpoint_path(path, device)
        if model:
            model = fix_model(model)
        return model
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None


def main():
    print("hi")
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNNs on graph datasets"
    )

    # dataset and experiment setup
    parser.add_argument(
        "--dataset",
        type=str,
        default="SingleP4",
        choices=[
            "BBBP",
            "Mutagenicity",
            "IMDB-BINARY",
            "NCI1",
            "BAMultiShapes",
            "reddit_threads",
            "twitch_egos",
            "github_stargazers",
            "SingleP4",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--load", action="store_true", help="Load pretrained model instead of training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load pretrained model",
    )
    parser.add_argument(
        "--max_depth", type=int, default=5, help="Maximum depth for decision tree"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["GCN", "GIN", "GAT", "GraphSAGE"],
        default="GCN",
        help="GNN architecture to use",
    )
    parser.add_argument(
        "--plot", type=int, default=0, help="Whether to plot predicates"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load data
    train_dataset, test_dataset, train_loader, test_loader, device = load_data(
        args.dataset, args.seed
    )
    y_labels_flat = []
    for data in train_dataset:
        y_labels_flat.append(data.y.item())
    for data in test_dataset:
        y_labels_flat.append(data.y.item())

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_labels_flat), y=y_labels_flat
    )
    class_weights_map = dict(zip(np.unique(y_labels_flat), class_weights))

    print(f"Correctly calculated class weights: {class_weights_map}")
    
    # Load model from checkpoint if specified
    if args.checkpoint:
        print(f"Loading pretrained model from checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint, device)
        if model is None:
            raise ValueError(f"Failed to load model from checkpoint: {args.checkpoint}")

        # It's a full model
        print("Loaded full model object")
        model = model.to(device)

        test_acc = test(model, test_loader, device)
        print(f"Loaded pretrained model | Test Accuracy: {test_acc:.4f}")
    else:
        print("No checkpoint provided. Please provide --checkpoint argument.")
        return

    # Run the explainer pipeline
    run_explainer_pipeline(
        args, model, device, train_loader, test_loader, train_dataset, test_dataset, class_weights_map
    )


if __name__ == "__main__":
    main()