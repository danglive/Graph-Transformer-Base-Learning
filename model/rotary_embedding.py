# model/rotary_embedding.py
# Credit: https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

from math import pi
from typing import Literal, Optional

import torch
from torch import nn, einsum, Tensor, broadcast_tensors
from torch.cuda.amp import autocast
from einops import rearrange, repeat


# ================================
# Helper Functions
# ================================

def exists(val: Optional[Tensor]) -> bool:
    """
    Check if a value is not None.

    Args:
        val (Optional[Tensor]): The value to check.

    Returns:
        bool: True if val is not None, False otherwise.
    """
    return val is not None


def default(val: Optional[Tensor], d: Tensor) -> Tensor:
    """
    Return the value if it exists; otherwise, return a default value.

    Args:
        val (Optional[Tensor]): The value to check.
        d (Tensor): The default value to return if val is None.

    Returns:
        Tensor: val if it exists, otherwise d.
    """
    return val if exists(val) else d


def broadcat(tensors: list[Tensor], dim: int = -1) -> Tensor:
    """
    Broadcast a list of tensors to a common shape and concatenate them along a specified dimension.

    Args:
        tensors (list[Tensor]): List of tensors to concatenate.
        dim (int, optional): The dimension along which to concatenate. Defaults to -1.

    Returns:
        Tensor: The concatenated tensor.
    """
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    """
    Slice a tensor along a specified dimension using the provided slice object.

    Args:
        t (Tensor): The input tensor.
        dim_slice (slice): The slice object to apply.
        dim (int): The dimension along which to slice.

    Returns:
        Tensor: The sliced tensor.

    Raises:
        ValueError: If the specified dimension is out of bounds.
    """
    dim = dim + t.ndim if dim < 0 else dim
    if dim < 0 or dim >= t.ndim:
        raise ValueError(
            f"Dimension {dim} is out of bounds for tensor with {t.ndim} dimensions."
        )
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


def rotate_half(x: Tensor) -> Tensor:
    """
    Perform a complex rotation by rotating the last two dimensions of the input tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The rotated tensor.
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


# ================================
# Rotary Embedding Functions
# ================================

@autocast(enabled=False)
def apply_rotary_emb(
    freqs: Tensor,
    t: Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
    freqs_seq_dim: Optional[int] = None
) -> Tensor:
    """
    Apply rotary embeddings to the input tensor using the provided frequencies.

    Args:
        freqs (Tensor): Frequencies for rotary embedding.
        t (Tensor): Input tensor to apply rotary embedding.
        start_index (int, optional): Start index for applying rotary embedding. Defaults to 0.
        scale (float, optional): Scaling factor for rotary embedding. Defaults to 1.0.
        seq_dim (int, optional): Sequence dimension for applying the embedding. Defaults to -2.
        freqs_seq_dim (Optional[int], optional): Sequence dimension for frequencies. Defaults to None.

    Returns:
        Tensor: The tensor after applying rotary embedding.
    """
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        freqs_seq_dim = default(freqs_seq_dim, 0)
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], (
        f'Feature dimension {t.shape[-1]} is not sufficient to rotate in all positions {rot_dim}.'
    )

    # Split tensor into left, middle (to be transformed), and right parts
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    # Concatenate the parts back together
    return torch.cat((t_left, t_transformed, t_right), dim=-1).type(dtype)


def apply_learned_rotations(
    rotations: Tensor,
    t: Tensor,
    start_index: int = 0,
    freq_ranges: Optional[Tensor] = None
) -> Tensor:
    """
    Apply learned rotations to the input tensor.

    Args:
        rotations (Tensor): Learned rotation parameters.
        t (Tensor): Input tensor to apply rotations.
        start_index (int, optional): Start index for applying rotations. Defaults to 0.
        freq_ranges (Optional[Tensor], optional): Frequency ranges for scaling rotations. Defaults to None.

    Returns:
        Tensor: The tensor after applying learned rotations.
    """
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# ================================
# RotaryEmbedding Class
# ================================

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings.

    Rotary embeddings apply a rotation to the query and key vectors in the self-attention mechanism
    based on their positions, enhancing the model's ability to generalize to longer sequences.

    Args:
        dim (int): Dimension of the rotary embedding.
        custom_freqs (Optional[Tensor], optional): Custom frequencies for the embedding. Defaults to None.
        freqs_for (Literal['lang', 'pixel', 'constant'], optional): Type of frequencies. Defaults to 'lang'.
        theta (float, optional): Base frequency for generating rotary embeddings. Defaults to 10000.
        max_freq (int, optional): Maximum frequency for pixel-related embeddings. Defaults to 10.
        num_freqs (int, optional): Number of frequency bands for constant frequencies. Defaults to 1.
        learned_freq (bool, optional): Whether to learn frequencies during training. Defaults to False.
        use_xpos (bool, optional): Whether to use extrapolation for long sequence lengths. Defaults to False.
        xpos_scale_base (int, optional): Scaling base for xpos extrapolation. Defaults to 512.
        interpolate_factor (float, optional): Factor for frequency interpolation. Must be >= 1.0. Defaults to 1.0.
        theta_rescale_factor (float, optional): Rescaling factor for theta. Defaults to 1.0.
        seq_before_head_dim (bool, optional): Whether the sequence dimension is before the head dimension. Defaults to False.
        cache_if_possible (bool, optional): Whether to cache the computed frequencies. Defaults to True.
        cache_max_seq_len (int, optional): Maximum sequence length for caching. Defaults to 8192.
    """

    def __init__(
        self,
        dim: int,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        theta: float = 10000.0,
        max_freq: int = 10,
        num_freqs: int = 1,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: int = 512,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = True,
        cache_max_seq_len: int = 8192
    ):
        super().__init__()

        # Rescale theta based on NTK-aware scaling
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        # Generate frequencies based on the specified type
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown freqs_for option: {freqs_for}")

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        # Initialize cached frequencies for efficiency
        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.register_buffer('cached_freqs_seq_len', torch.tensor(0), persistent=False)

        # Register frequencies as a learnable parameter if specified
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq

        # Dummy tensor for device management
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

        # Set default sequence dimension based on input configuration
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # Ensure interpolation factor is valid
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be >= 1.0")
        self.interpolate_factor = interpolate_factor

        # Setup for extrapolation (xpos) if enabled
        self.use_xpos = use_xpos
        if self.use_xpos:
            scale = (torch.arange(0, dim, 2, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = xpos_scale_base

            self.register_buffer('scale', scale, persistent=False)
            self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent=False)
            self.register_buffer('cached_scales_seq_len', torch.tensor(0), persistent=False)

        # Static method for applying rotary embeddings
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self) -> torch.device:
        """
        Get the device of the embedding.

        Returns:
            torch.device: The device on which the embedding is located.
        """
        return self.dummy.device

    def get_seq_pos(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0
    ) -> Tensor:
        """
        Compute sequence positions for extrapolation (xpos).

        Args:
            seq_len (int): The length of the sequence.
            device (torch.device): The device to place the tensor.
            dtype (torch.dtype): The data type of the tensor.
            offset (int, optional): Position offset. Defaults to 0.

        Returns:
            Tensor: The computed sequence positions.
        """
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(
        self,
        t: Tensor,
        seq_dim: Optional[int] = None,
        offset: int = 0,
        scale: Optional[float] = None
    ) -> Tensor:
        """
        Apply rotary embedding to queries or keys.

        Args:
            t (Tensor): Input tensor (queries or keys).
            seq_dim (Optional[int], optional): Sequence dimension. Defaults to None.
            offset (int, optional): Position offset. Defaults to 0.
            scale (Optional[float], optional): Scaling factor. Defaults to None.

        Returns:
            Tensor: The tensor after applying rotary embedding.
        """
        seq_dim = default(seq_dim, self.default_seq_dim)

        if self.use_xpos:
            if scale is None:
                raise ValueError(
                    'When using xpos, you must provide a scaling factor.'
                )

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)
        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(
        self,
        q: Tensor,
        k: Tensor,
        seq_dim: Optional[int] = None,
        offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        """
        Rotate queries using cached keys.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            seq_dim (Optional[int], optional): Sequence dimension. Defaults to None.
            offset (int, optional): Position offset. Defaults to 0.

        Returns:
            tuple[Tensor, Tensor]: Rotated query and key tensors.
        """
        dtype, device = q.dtype, q.device
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len, "Query length must be less than or equal to key length."

        q_scale = k_scale = 1.0

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)
            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(
            q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(
            k, seq_dim=seq_dim, scale=k_scale ** -1
        )

        return rotated_q.type(q.dtype), rotated_k.type(k.dtype)

    def rotate_queries_and_keys(
        self,
        q: Tensor,
        k: Tensor,
        seq_dim: Optional[int] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Rotate both queries and keys using rotary embeddings.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            seq_dim (Optional[int], optional): Sequence dimension. Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: Rotated query and key tensors.
        """
        seq_dim = default(seq_dim, self.default_seq_dim)

        if not self.use_xpos:
            raise ValueError(
                'Extrapolation (xpos) must be enabled to use rotate_queries_and_keys.'
            )

        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)
        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        return rotated_q.type(q.dtype), rotated_k.type(k.dtype)

    def get_scale(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0
    ) -> Tensor:
        """
        Compute scaling factors for extrapolation.

        Args:
            t (Tensor): Input tensor representing positions.
            seq_len (Optional[int], optional): Sequence length. Defaults to None.
            offset (int, optional): Position offset. Defaults to 0.

        Returns:
            Tensor: Scaling factors.

        Raises:
            AssertionError: If xpos is not enabled.
        """
        assert self.use_xpos, "Extrapolation (xpos) must be enabled to compute scales."

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and
            exists(self.cached_scales) and
            (seq_len + offset) <= self.cached_scales_seq_len.item()
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r=2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(torch.tensor(seq_len, device=self.device))

        return scale

    def get_axial_freqs(self, *dims: int) -> Tensor:
        """
        Compute axial frequencies for multi-dimensional inputs.

        Args:
            *dims (int): Dimensions for which to compute frequencies.

        Returns:
            Tensor: The axial frequencies.
        """
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast(enabled=False)
    def forward(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0
    ) -> Tensor:
        """
        Compute the rotary frequencies based on sequence length.

        Args:
            t (Tensor): Input tensor representing positions.
            seq_len (Optional[int], optional): Sequence length. Defaults to None.
            offset (int, optional): Position offset. Defaults to 0.

        Returns:
            Tensor: Frequencies for rotary embedding.
        """
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and
            exists(self.cached_freqs) and
            (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(torch.tensor(seq_len, device=self.device))

        return freqs