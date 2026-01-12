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


def main(
        checkpoint_path: str,
        plot: bool = True
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Load data
    train_dataset, test_dataset, train_loader, test_loader, device = load_data(
        dataset, seed
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
    if checkpoint:
        print(f"Loading pretrained model from checkpoint: {checkpoint}")
        model = load_model(checkpoint, device)
        if model is None:
            raise ValueError(f"Failed to load model from checkpoint: {checkpoint}")

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
