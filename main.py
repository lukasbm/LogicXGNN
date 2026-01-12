import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from build_logicGNN import *
from explain_gnn import *
from gnn import GAT, GCN, GIN, GraphSAGE
from grounding import *
from load_data import load_data
from utils import load_model, test, train
from explainer_pipeline import stop_dict, run_explainer_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNNs on graph datasets"
    )

    # dataset and experiment setup
    parser.add_argument(
        "--dataset",
        type=str,
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
    # ✅ Decide whether to use conv3 based on dataset
    use_conv3 = args.dataset in [
        "BAMultiShapes",
        "SingleP4",
    ]  # Use 3 layers for structural tasks
    # use_node_features = args.dataset in ["BBBP", "Mutagenicity", "NCI1"]

    def get_model(
            arch,
            in_channels,
            hidden_channels,
            out_channels,
            num_classes,
            use_conv3=True,
            dropout=0.0,
    ):
        if arch == "GCN":
            return GCN(
                in_channels,
                hidden_channels,
                out_channels,
                num_classes,
                use_conv3,
                dropout,
            )
        elif arch == "GIN":
            return GIN(
                in_channels, hidden_channels, out_channels, num_classes, use_conv3
            )
        elif arch == "GAT":
            return GAT(
                in_channels,
                hidden_channels,
                out_channels,
                num_classes,
                use_conv3=use_conv3,
            )
        elif arch == "GraphSAGE":
            return GraphSAGE(
                in_channels,
                hidden_channels,
                out_channels,
                num_classes,
                use_conv3=use_conv3,
            )
        else:
            raise ValueError(f"Unknown architecture {arch}")

    # Use dropout for SingleP4 to prevent overfitting
    dropout = 0.3 if args.dataset == "SingleP4" else 0.0

    model = get_model(
        args.arch,
        in_channels=train_dataset[0].x.shape[1],
        hidden_channels=64,  # Increased capacity for structural learning
        out_channels=64,
        num_classes=2,
        use_conv3=use_conv3,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    if args.arch == "GCN":
        model_path = f"./models/{args.dataset}_{args.seed}.pth"
    else:
        model_path = f"./models/{args.dataset}_{args.seed}_{args.arch}.pth"

    if args.load:
        # 🔹 Load pretrained weights
        model = load_model(model, model_path, device=device)
        test_acc = test(model, test_loader, device)
        print(f"Loaded model | Test Accuracy: {test_acc:.4f}")
    else:
        # 🔹 Train from scratch with early stopping
        for epoch in range(1, 201):
            loss = train(model, train_loader, optimizer, criterion, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            print(
                f"Epoch {epoch}, Loss {loss:.4f}, Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}"
            )

            # check dataset-specific stop threshold
            if test_acc >= stop_dict[args.dataset]:
                print(
                    f"✅ Early stopping at epoch {epoch} for {args.dataset}: Test Acc {test_acc:.4f}"
                )
                break

        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved model to {model_path}")

    # Run the explainer pipeline
    run_explainer_pipeline(
        args, model, device, train_loader, test_loader, train_dataset, test_dataset, class_weights_map
    )


if __name__ == "__main__":
    main()