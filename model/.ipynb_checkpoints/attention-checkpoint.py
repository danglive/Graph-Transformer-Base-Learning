# model/attention.py

import torch
from typing import Optional
from torch import nn, einsum
from einops import rearrange, repeat
from model.rotary_embedding import apply_rotary_emb, exists, default

def rearrange_tensor(tensor, pattern, h=None):
    """
    Utility function to rearrange tensors for multi-head attention.

    Args:
        tensor (torch.Tensor): The tensor to be rearranged.
        pattern (str): The einops pattern for rearrangement.
        h (int, optional): Number of heads. If provided, it's passed to rearrange.

    Returns:
        torch.Tensor: The rearranged tensor.
    """
    if h is not None:
        return rearrange(tensor, pattern, h=h)
    else:
        return rearrange(tensor, pattern)

class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism for graphs.

    Args:
        dim (int): Dimension of the input features.
        pos_emb (bool): Whether to use positional embeddings (optional).
        dim_head (int): Dimension of each attention head.
        heads (int): Number of attention heads.
        edge_dim (int, optional): Dimension of the edge features.
    """
    def __init__(self, dim, pos_emb=True, dim_head=64, heads=16, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for attention scores
        self.pos_emb = pos_emb  # Optional positional embeddings

        # Linear layers for queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask=None):
        """
        Forward pass for the attention mechanism.

        Args:
            nodes (torch.Tensor): Node feature tensor of shape (batch_size, num_nodes, feature_dim).
            edges (torch.Tensor): Edge feature tensor of shape (batch_size, num_nodes, num_nodes, edge_dim).
            mask (torch.Tensor, optional): Optional mask tensor.

        Returns:
            torch.Tensor: Output features after attention.
        """
        h = self.heads

        # Compute queries, keys, and values
        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim=-1)  # Split into keys and values
        e_kv = self.edges_to_kv(edges)  # Edge information

        # Rearrange for multi-head attention using rearrange_tensor
        q = rearrange_tensor(q, 'b ... (h d) -> (b h) ... d', h=h)
        k = rearrange_tensor(k, 'b ... (h d) -> (b h) ... d', h=h)
        v = rearrange_tensor(v, 'b ... (h d) -> (b h) ... d', h=h)
        e_kv = rearrange_tensor(e_kv, 'b i j (h d) -> (b h) i j d', h=h)

        # Apply positional embeddings if present
        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device=nodes.device))
            freqs = rearrange_tensor(freqs, 'n d -> () n d')  # No need for 'h' here
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # Incorporate edge information into keys and values
        k = rearrange_tensor(k, 'b j d -> b () j d')
        v = rearrange_tensor(v, 'b j d -> b () j d')
        k = k + e_kv
        v = v + e_kv

        # Compute attention scores
        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        # Apply attention mask if provided
        if exists(mask):
            mask = rearrange_tensor(mask, 'b i -> b i ()') & rearrange_tensor(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention weights and output
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange_tensor(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
