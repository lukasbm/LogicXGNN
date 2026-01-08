"""
Add rich structural node features for P4 detection
"""
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class StructuralFeatures(BaseTransform):
    """
    Add multiple structural features to help GNN distinguish cographs from non-cographs:
    - Node degree (1-hot encoded up to max_degree)
    - Degree of neighbors (aggregated stats)
    - Local clustering coefficient
    """

    def __init__(self, max_degree=10):
        self.max_degree = max_degree

    def __call__(self, data):
        row, col = data.edge_index
        deg = degree(col, num_nodes=data.num_nodes, dtype=torch.long)

        # Feature 1: One-hot encoded degree
        deg_onehot = torch.zeros(data.num_nodes, self.max_degree)
        deg_clamped = torch.clamp(deg, 0, self.max_degree - 1)
        deg_onehot[torch.arange(data.num_nodes), deg_clamped] = 1

        # Feature 2: Raw degree (normalized)
        deg_norm = deg.float().unsqueeze(1) / self.max_degree

        # Feature 3: Neighbor degree statistics (helps identify P4 patterns)
        neighbor_deg_sum = torch.zeros(data.num_nodes, 1)
        neighbor_deg_max = torch.zeros(data.num_nodes, 1)

        for i in range(data.num_nodes):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                neighbor_degrees = deg[neighbors].float()
                neighbor_deg_sum[i] = neighbor_degrees.sum() / len(neighbors)
                neighbor_deg_max[i] = neighbor_degrees.max()

        # Combine all features
        data.x = torch.cat([deg_onehot, deg_norm, neighbor_deg_sum, neighbor_deg_max], dim=1)

        return data
