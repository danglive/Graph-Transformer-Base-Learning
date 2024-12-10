# model/graph_transformer.py

import torch
from torch import nn
from model.attention import MultiHeadAttention
from model.rotary_embedding import RotaryEmbedding, default, exists


class PreNorm(nn.Module):
    """
    Applies LayerNorm before the given function.

    Args:
        dim: Dimension of the input features.
        fn: The function to apply after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        """Applies LayerNorm before passing through `fn`."""
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class Residual(nn.Module):
    """Residual connection layer."""

    def forward(self, x, res):
        """Applies a residual connection (skip connection)."""
        return x + res


class GatedResidual(nn.Module):
    """
    Gated residual connection layer.

    Args:
        dim: Dimension of the input features.
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),  # Linear layer to generate the gate
            nn.Sigmoid()  # Sigmoid to constrain the gate between 0 and 1
        )

    def forward(self, x, res):
        """Applies a gated residual connection."""
        gate_input = torch.cat((x, res, x - res), dim=-1)  # Combine the inputs
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)
        

def get_activation_function(name: str):
    """
    Returns an activation function object based on the provided name.
    """
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    elif name.lower() == 'silu':
        return nn.SiLU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")
        

def feed_forward(dim, ff_mult=4, activation='silu'):
    """
    FeedForward network used in transformers.

    Args:
        dim: Dimension of the input features.
        ff_mult: Multiplier for the hidden layer size.

    Returns:
        nn.Sequential: FeedForward network.
    """
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        get_activation_function(activation),
        nn.Linear(dim * ff_mult, dim)
    )


class GraphTransformer(nn.Module):
    """
    Transformer model for graph data.

    Args:
        dim: Dimension of the input features.
        depth: Number of transformer layers.
        dim_head: Dimension of each attention head.
        edge_dim: Dimension of the edge features.
        heads: Number of attention heads.
        gated_residual: Whether to use gated residual connections.
        with_feedforwards: Whether to include feedforward networks.
        norm_edges: Whether to normalize edge features.
        rel_pos_emb: Whether to use relative positional embeddings.
        accept_adjacency_matrix: Whether to accept an adjacency matrix.
    """

    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=16,
        gated_residual=True,
        with_feedforwards=True,
        norm_edges=True,
        rel_pos_emb=True,
        accept_adjacency_matrix=True,
        activation='silu'
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()
        self.adj_emb = (
            nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None
        )

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        # Stacking multiple transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.ModuleList([
                    PreNorm(
                        dim,
                        MultiHeadAttention(
                            dim,
                            pos_emb=pos_emb,
                            edge_dim=edge_dim,
                            dim_head=dim_head,
                            heads=heads
                        )
                    ),
                    GatedResidual(dim) if gated_residual else Residual()
                ]),
                nn.ModuleList([
                    PreNorm(dim, feed_forward(dim, activation=activation)),
                    GatedResidual(dim) if gated_residual else Residual()
                ]) if with_feedforwards else None
            ]))

    def forward(self, nodes, edges=None, adj_mat=None, mask=None):
        """
        Forward pass for graph transformer.

        Args:
            nodes: Node feature tensor of shape (batch_size, num_nodes, feature_dim).
            edges: Edge feature tensor of shape (batch_size, num_nodes, num_nodes, edge_dim).
            adj_mat: Adjacency matrix tensor of shape (batch_size, num_nodes, num_nodes).
            mask: Optional mask tensor.

        Returns:
            Tuple[Tensor, Tensor]: Updated node and edge features.
        """
        batch_size, seq_len, _ = nodes.shape

        # Normalize edges if required
        if exists(edges):
            edges = self.norm_edges(edges)

        # Apply adjacency matrix embeddings if available
        if exists(adj_mat):
            assert adj_mat.shape == (batch_size, seq_len, seq_len)
            assert exists(
                self.adj_emb
            ), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(adj_mat.long())

        # Combine edges and adjacency matrix information
        all_edges = default(edges, 0) + default(adj_mat, 0)

        # Apply each transformer layer
        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            residual_nodes = nodes
            nodes = attn_residual(attn(nodes, all_edges, mask=mask), residual_nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                residual_nodes = nodes
                nodes = ff_residual(ff(nodes), residual_nodes)

        return nodes, edges