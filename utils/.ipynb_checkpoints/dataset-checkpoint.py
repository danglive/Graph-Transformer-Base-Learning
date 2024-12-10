# utils/dataset.py

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom Dataset for graph data suitable for PyTorch Geometric.

    Args:
        node_features: List of node features.
        edge_indices: List of edge indices.
        edge_attr: List of edge attributes.
        labels: List of labels.
    """

    def __init__(self, node_features, edge_indices, edge_attr, labels):
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_attr = edge_attr
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Data: A PyTorch Geometric Data object containing graph data.
        """
        node_feature = torch.tensor(
            self.node_features[idx], dtype=torch.float
        )
        edge_index = torch.tensor(
            self.edge_indices[idx], dtype=torch.long
        ).t().contiguous()
        edge_attr = torch.tensor(
            self.edge_attr[idx], dtype=torch.float
        )
        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        return Data(
            x=node_feature,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label
        )