# model/model.py

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter_mean
from model.graph_transformer import GraphTransformer


def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.

    Args:
        model: The model to count parameters for.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GraphTransformerClassifier(nn.Module):
    """
    Graph Transformer Model for graph classification tasks.

    Args:
        dim: Dimension of the input features.
        depth: Number of transformer layers.
        num_classes: Number of output classes.
        num_feature_node: Number of node feature dimensions.
        num_feature_edge: Number of edge feature dimensions.
        edge_dim: Dimension of the edge features (optional).
        with_feedforwards: Whether to include feedforward networks.
        gated_residual: Whether to use gated residual connections.
        rel_pos_emb: Whether to use relative positional embeddings.
        device: Device to run the model on.
        dropout_rate: Dropout rate for regularization.
        pretrained_path (str, optional): Path to the pretrained encoder checkpoint.
    """

    def __init__(
        self,
        dim,
        depth,
        num_classes,
        num_feature_node,
        num_feature_edge,
        edge_dim=None,
        with_feedforwards=True,
        gated_residual=True,
        rel_pos_emb=True,
        device=None,
        dropout_rate=0.5,
        activation="silu",
        pretrained_path=None,
    ):
        super(GraphTransformerClassifier, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        edge_dim = edge_dim if edge_dim else dim

        # Encoders for node and edge features
        self.node_feature_encoder = nn.Linear(num_feature_node, dim).to(self.device)
        self.edge_feature_encoder = nn.Linear(num_feature_edge, edge_dim).to(self.device)

        # Initialize the encoder and feature encoders
        self.encoder = GraphTransformer(
            dim=dim,
            depth=depth,
            edge_dim=edge_dim,
            with_feedforwards=with_feedforwards,
            gated_residual=gated_residual,
            rel_pos_emb=rel_pos_emb,
            activation = activation
        ).to(self.device)

        # Load pretrained encoder if path is provided
        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)

        # Prediction heads that output class logits
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(dim, num_classes)
            ).to(self.device)
        ])

    def load_pretrained_encoder(self, pretrained_path):
        """
        Loads the pretrained encoder state_dict.

        Args:
            pretrained_path (str): Path to the pretrained encoder checkpoint.
        """
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        encoder_state_dict = checkpoint['encoder_state_dict']
        self.encoder.load_state_dict(encoder_state_dict)
        print(f"Loaded pretrained encoder from {pretrained_path}")
        
    def encode_features(self, batch):
        """
        Encodes node and edge features.

        Args:
            batch: A batch of graph data.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Encoded nodes, edges, mask, and adjacency matrix.
        """
        z_x = self.node_feature_encoder(batch.x.float())
        z_e = self.edge_feature_encoder(batch.edge_attr.float())
        nodes, mask = to_dense_batch(z_x, batch.batch)
        edges = to_dense_adj(
            batch.edge_index, batch.batch, edge_attr=z_e
        )
        adj_matrix = to_dense_adj(
            batch.edge_index, batch.batch
        ).bool()
        return nodes, edges, mask, adj_matrix

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch: A batch of graph data.

        Returns:
            Tensor: Model predictions.
        """
        nodes, edges, mask, adj_matrix = self.encode_features(batch)
        nodes, edges = self.encoder(
            nodes, edges, adj_mat=adj_matrix, mask=mask
        )
        res = scatter_mean(nodes[mask], batch.batch, dim=0)
        preds = [head(res) for head in self.pred_heads]
        return torch.cat(preds, dim=-1)


class GraphTransformerMAE(nn.Module):
    """
    Graph Transformer Masked Autoencoder for self-supervised learning.

    This model can perform both node-level masking and feature-level masking
    based on the configuration. It learns to reconstruct masked parts of the
    input graph data.

    Args:
        dim (int): Dimension of the model (hidden size).
        depth (int): Number of transformer layers.
        num_feature_node (int): Number of input node feature dimensions.
        num_feature_edge (int): Number of input edge feature dimensions.
        mask_ratio (float): Ratio of nodes to mask (used for node-level masking).
        mask_feature_per_node (int): Number of features to mask per node (used for feature-level masking).
        edge_dim (int, optional): Dimension of the edge features after embedding.
        with_feedforwards (bool): Whether to include feedforward networks.
        gated_residual (bool): Whether to use gated residual connections.
        rel_pos_emb (bool): Whether to use relative positional embeddings.
        device (torch.device, optional): Device to run the model on.
        activation (str): Activation function to use (default is "silu").
    """

    def __init__(
        self,
        dim,
        depth,
        num_feature_node,
        num_feature_edge,
        mask_ratio=0.15,
        mask_feature_per_node=0,
        edge_dim=None,
        with_feedforwards=True,
        gated_residual=True,
        rel_pos_emb=True,
        device=None,
        activation="silu",
    ):
        super(GraphTransformerMAE, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mask_ratio = mask_ratio
        self.mask_feature_per_node = mask_feature_per_node
        edge_dim = edge_dim if edge_dim else dim

        # Ensure that only one masking strategy is active at a time
        if sum([
            self.mask_ratio > 0,
            self.mask_feature_per_node > 0,
        ]) > 1:
            raise ValueError("Only one of mask_ratio or mask_feature_per_node should be greater than 0.")

        # Encoder for node features
        self.node_feature_encoder = nn.Linear(num_feature_node, dim).to(self.device)

        # Encoder for edge features
        self.edge_feature_encoder = nn.Linear(num_feature_edge, edge_dim).to(self.device)

        # Graph Transformer encoder
        self.encoder = GraphTransformer(
            dim=dim,
            depth=depth,
            edge_dim=edge_dim,
            with_feedforwards=with_feedforwards,
            gated_residual=gated_residual,
            rel_pos_emb=rel_pos_emb,
            activation=activation,
        ).to(self.device)

        # Decoder to reconstruct node features
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_feature_node),
        ).to(self.device)

    def mask_nodes_or_features(self, x):
        """
        Masks nodes or features in the input batch based on the configuration.

        Args:
            x (Tensor): Node features tensor of shape [batch_size, num_nodes, num_features].

        Returns:
            Tuple[Tensor, Tensor]: The masked node features and the mask tensor.
                - For node-level masking, the mask tensor has shape [batch_size, num_nodes], where 1 indicates masked node.
                - For feature-level masking, the mask tensor has shape [batch_size, num_nodes, num_features], where 1 indicates masked feature.
        """
        batch_size, num_nodes, num_features = x.size()

        if self.mask_feature_per_node > 0:
            # Feature-level masking: Mask a fixed number of features per node
            mask = torch.zeros((batch_size, num_nodes, num_features), device=self.device)

            for i in range(batch_size):
                for j in range(num_nodes):
                    # Randomly select features to mask for each node
                    feature_indices = torch.randperm(num_features)[:self.mask_feature_per_node]
                    mask[i, j, feature_indices] = 1

            # Mask the node features by zeroing out the masked features
            x_masked = x.clone()
            x_masked[mask.bool()] = 0

            return x_masked, mask

        else:
            # Node-level masking: Mask a percentage of nodes
            len_keep = int(num_nodes * (1 - self.mask_ratio))

            # Generate random indices for nodes
            noise = torch.rand(batch_size, num_nodes, device=self.device)

            # Sort noise to get random node indices
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Keep a subset of nodes
            ids_keep = ids_shuffle[:, :len_keep]

            # Create mask: 1 indicates masked node
            mask = torch.ones([batch_size, num_nodes], device=self.device)
            mask[:, :len_keep] = 0
            # Unshuffle to get original order
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Mask the node features by zeroing out the features of masked nodes
            x_masked = x.clone()
            # Expand mask to match feature dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(x_masked)
            x_masked[mask_expanded.bool()] = 0

            return x_masked, mask

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (Data): A batch of graph data from PyTorch Geometric.

        Returns:
            Tensor: The reconstruction loss for the masked nodes or features.
        """
        # Step 1: Get original node features in dense format
        nodes_orig, mask_nodes = to_dense_batch(batch.x.float(), batch.batch)
        # nodes_orig shape: [batch_size, num_nodes, num_feature_node]

        # Step 2: Mask the original node features
        nodes_masked, mask = self.mask_nodes_or_features(nodes_orig)
        # nodes_masked shape: same as nodes_orig
        # mask shape:
        # - [batch_size, num_nodes, num_feature_node] for feature-level masking
        # - [batch_size, num_nodes] for node-level masking

        # Step 3: Encode the masked node features
        z_x = self.node_feature_encoder(nodes_masked)
        # z_x shape: [batch_size, num_nodes, dim]

        # Encode edge features
        z_e = self.edge_feature_encoder(batch.edge_attr.float())
        # z_e shape: [num_edges, edge_dim]

        # Convert edge data to dense format
        edges = to_dense_adj(batch.edge_index, batch.batch, edge_attr=z_e)
        # edges shape: [batch_size, num_nodes, num_nodes, edge_dim]
        adj_matrix = to_dense_adj(batch.edge_index, batch.batch).bool()
        # adj_matrix shape: [batch_size, num_nodes, num_nodes]

        # Step 4: Pass through the transformer encoder
        nodes_encoded, _ = self.encoder(
            z_x, edges, adj_mat=adj_matrix
        )
        # nodes_encoded shape: [batch_size, num_nodes, dim]

        # Step 5: Decode to reconstruct node features
        pred = self.decoder(nodes_encoded)
        # pred shape: [batch_size, num_nodes, num_feature_node]

        # Step 6: Compute loss between predicted and original features at masked positions
        target = nodes_orig  # Original node features

        if self.mask_feature_per_node > 0:
            # Feature-level masking
            loss = F.mse_loss(pred[mask.bool()], target[mask.bool()])
        else:
            # Node-level masking
            # Expand mask to match feature dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(target)
            loss = F.mse_loss(pred[mask_expanded.bool()], target[mask_expanded.bool()])

        return loss

    def get_encoder(self):
        """
        Returns the encoder model.

        Returns:
            nn.Module: The encoder part of the model.
        """
        return self.encoder

    def save_encoder(self, path):
        """
        Saves the encoder's state_dict to the specified path.

        Args:
            path (str): The path to save the encoder.
        """
        torch.save({'encoder_state_dict': self.encoder.state_dict()}, path)
        print(f"Encoder saved to {path}")