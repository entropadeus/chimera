"""
Chimera: A Novel Hybrid LLM Architecture
=========================================
Combines proven innovations from 2023-2025 research:
- Gated Linear Recurrence (RG-LRU from Griffin) for O(1) memory global context
- Sliding Window GQA for local retrieval precision
- SwiGLU FFN (standard in Llama/Mistral/PaLM)
- RoPE with NTK-aware scaling for long context
- 3:1 Recurrent:Attention layer ratio (proven in Jamba/Griffin)

Target: ~1.5B params for laptop inference (CPU/consumer GPU)

References:
- Griffin: arxiv.org/abs/2402.19427
- Mamba: arxiv.org/abs/2312.00752
- GQA: arxiv.org/abs/2305.13245
- SwiGLU: arxiv.org/abs/2002.05202
- RoPE/YaRN: arxiv.org/abs/2309.00071
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChimeraConfig:
    """Configuration for Chimera model."""
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than query heads
    vocab_size: int = 32000
    max_seq_len: int = 8192
    sliding_window: int = 512  # Local attention window
    ffn_hidden_mult: float = 8/3  # SwiGLU uses 8/3 ratio
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None  # NTK-aware scaling factor
    dropout: float = 0.0
    recurrence_dim: int = None  # Defaults to d_model if None
    attention_every_n: int = 3  # Place attention layer every N layers
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.recurrence_dim is None:
            self.recurrence_dim = self.d_model
        self.ffn_hidden = int(self.d_model * self.ffn_hidden_mult)
        # Round to nearest multiple of 256 for efficiency
        self.ffn_hidden = ((self.ffn_hidden + 255) // 256) * 256
        self.head_dim = self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for stability, then cast back
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with optional NTK-aware scaling."""

    def __init__(self, dim: int, max_seq_len: int = 8192,
                 theta: float = 10000.0, scaling: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Apply NTK-aware scaling if specified
        if scaling is not None and scaling > 1.0:
            # Scale base frequency for NTK-aware interpolation
            inv_freq = inv_freq / (scaling ** (dim / (dim - 2)))

        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute cos/sin cache for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for real/imag pairs
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        """Apply rotary embeddings to input tensor."""
        if seq_len + offset > self.cos_cached.size(0):
            self._build_cache(seq_len + offset)

        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation for FFN - standard in Llama/Mistral/PaLM."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(xW1) ⊙ xW3) W2
        gate = F.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))


class GatedLinearRecurrence(nn.Module):
    """
    Real-Gated Linear Recurrent Unit (RG-LRU) - Griffin-accurate implementation.

    Key innovations from Griffin paper (arxiv.org/abs/2402.19427):
    - TWO gates: recurrence gate (r_t) AND input gate (i_t)
    - Learned base recurrence rate: a = σ(Λ), then a_t = a^(c·r_t)
    - Log-space computation for numerical stability
    - Special initialization for long-range memory (0.9-0.999 range)

    Recurrence: h_t = a_t ⊙ h_{t-1} + √(1-a_t²) ⊙ (i_t ⊙ x_t)
    """

    def __init__(self, d_model: int, expansion: int = 1, c: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * expansion
        self.c = c  # Gate scaling constant (Griffin uses 8)

        # Input projection (for x_t that enters the state)
        self.input_proj = nn.Linear(d_model, self.hidden_dim, bias=False)

        # Input gate i_t - controls WHAT new info enters (Griffin's key insight)
        self.input_gate_proj = nn.Linear(d_model, self.hidden_dim, bias=True)

        # Recurrence gate r_t - controls HOW MUCH old state to keep
        self.recurrence_gate_proj = nn.Linear(d_model, self.hidden_dim, bias=True)

        # Learned base recurrence rate: a = σ(Λ) per dimension
        # Initialize Λ so a^c is uniform in [0.9, 0.999] for long memory
        # a^c in [0.9, 0.999] => a in [0.9^(1/c), 0.999^(1/c)]
        # For c=8: a in [0.987, 0.99987] => Λ in [4.3, 9.1] approx
        # We initialize Λ uniform in this range, then σ(Λ) gives our base rates
        lambda_init = torch.empty(self.hidden_dim).uniform_(4.0, 9.0)
        self.lambda_param = nn.Parameter(lambda_init)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

        # Learnable initial state
        self.h0 = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # Layer norm for stability
        self.norm = RMSNorm(self.hidden_dim)

    def _compute_a_t(self, r_t: torch.Tensor) -> torch.Tensor:
        """
        Compute recurrence coefficient a_t = a^(c·r_t) in log-space for stability.

        a = σ(Λ) is the learned base rate per dimension
        r_t = σ(recurrence_gate) modulates it per timestep
        """
        # Base recurrence rate (learned per dimension)
        a = torch.sigmoid(self.lambda_param)  # [hidden_dim]

        # Compute a^(c·r_t) in log-space: exp(c·r_t·log(a))
        log_a = torch.log(a + 1e-8)  # [hidden_dim]
        # r_t is [batch, seq, hidden_dim], scale by c and multiply by log_a
        a_t = torch.exp(self.c * r_t * log_a)  # [batch, seq, hidden_dim]

        return a_t

    def forward(self, x: torch.Tensor,
                recurrence_state: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional cached state for inference.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            recurrence_state: Previous hidden state [batch, 1, hidden_dim]
            use_cache: Whether to return updated state for caching

        Returns:
            output: [batch, seq_len, d_model]
            new_state: Updated recurrence state if use_cache else None
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)  # [batch, seq, hidden_dim]

        # Compute gates
        i_t = torch.sigmoid(self.input_gate_proj(x))  # Input gate
        r_t = torch.sigmoid(self.recurrence_gate_proj(x))  # Recurrence gate

        # Compute recurrence coefficient a_t = a^(c·r_t)
        a_t = self._compute_a_t(r_t)

        # Initialize hidden state
        if recurrence_state is None:
            h = self.h0.expand(batch_size, 1, -1)
        else:
            h = recurrence_state

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x_proj[:, t:t+1, :]
            i_t_step = i_t[:, t:t+1, :]
            a_t_step = a_t[:, t:t+1, :]

            # Griffin recurrence: h_t = a_t ⊙ h_{t-1} + √(1-a_t²) ⊙ (i_t ⊙ x_t)
            # The input gate i_t selectively filters what new info enters
            # The sqrt(1-a²) factor ensures norm preservation
            gated_input = i_t_step * x_t
            h = a_t_step * h + torch.sqrt(1 - a_t_step.pow(2) + 1e-6) * gated_input
            outputs.append(h)

        # Stack outputs
        output = torch.cat(outputs, dim=1)
        output = self.norm(output)
        output = self.output_proj(output)

        new_state = h if use_cache else None
        return output, new_state

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallelized forward pass for training (no sequential bottleneck).
        Uses associative scan approximation for O(n) parallel computation.
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)

        # Compute gates
        i_t = torch.sigmoid(self.input_gate_proj(x))
        r_t = torch.sigmoid(self.recurrence_gate_proj(x))

        # Compute a_t
        a_t = self._compute_a_t(r_t)

        # Apply input gate to projected input
        gated_input = i_t * x_proj

        # Parallel scan approximation
        # Compute cumulative product of a_t (in log-space for stability)
        log_a = torch.log(a_t + 1e-8)
        cumsum_log_a = torch.cumsum(log_a, dim=1)
        a_cumulative = torch.exp(cumsum_log_a)

        # Weighted sum with sqrt complement
        sqrt_complement = torch.sqrt(1 - a_t.pow(2) + 1e-6)
        weighted_input = sqrt_complement * gated_input

        # Compute via cumsum trick
        output = torch.cumsum(weighted_input / (a_cumulative + 1e-6), dim=1) * a_cumulative

        output = self.norm(output)
        output = self.output_proj(output)

        return output


class SlidingWindowGQA(nn.Module):
    """
    Grouped-Query Attention with Sliding Window.

    Combines:
    - GQA (fewer KV heads) for memory efficiency
    - Sliding window for local attention (no global KV cache needed)
    - RoPE for position encoding
    """

    def __init__(self, config: ChimeraConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.window_size = config.sliding_window

        # GQA: n_heads queries, n_kv_heads key-values
        self.n_rep = self.n_heads // self.n_kv_heads  # Repeat factor

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
            config.rope_scaling
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, self.n_heads, head_dim)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                position_offset: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with sliding window attention and optional KV cache.

        The sliding window limits attention to local context, keeping memory bounded.
        """
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(x, seq_len, position_offset)
        q_rot = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k_rot = k.transpose(1, 2)

        # Reshape for RoPE application
        q_rot = q_rot.reshape(batch_size * self.n_heads, seq_len, self.head_dim)
        k_rot = k_rot.reshape(batch_size * self.n_kv_heads, seq_len, self.head_dim)

        q_rot, k_rot = apply_rotary_pos_emb(q_rot.unsqueeze(0), k_rot.unsqueeze(0), cos, sin)

        q = q_rot.squeeze(0).view(batch_size, self.n_heads, seq_len, self.head_dim)
        k = k_rot.squeeze(0).view(batch_size, self.n_kv_heads, seq_len, self.head_dim)
        v = v.transpose(1, 2)  # [batch, kv_heads, seq, dim]

        # Handle KV cache for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        kv_len = k.size(2)

        # For inference with cache: trim to window size
        # For training: keep full KV, use sliding window mask instead
        if use_cache and kv_len > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
            kv_len = self.window_size

        new_cache = (k, v) if use_cache else None

        # Repeat KV heads to match query heads (GQA)
        k = self._repeat_kv(k.transpose(1, 2)).transpose(1, 2)
        v = self._repeat_kv(v.transpose(1, 2)).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Create sliding window causal mask
        q_len = q.size(2)
        kv_len = k.size(2)

        # Start with causal mask (upper triangle = -inf)
        causal_mask = torch.triu(
            torch.full((q_len, kv_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1
        )

        # Add sliding window constraint (vectorized)
        # Each position i can only attend to positions max(0, i - window_size + 1) to i
        if self.window_size < kv_len:
            # Create row and column indices
            row_idx = torch.arange(q_len, device=x.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=x.device).unsqueeze(0)
            # Mask where column is before the window start for each row
            window_mask = col_idx < (row_idx - self.window_size + 1)
            causal_mask = causal_mask.masked_fill(window_mask, float('-inf'))

        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, new_cache


class ChimeraBlock(nn.Module):
    """
    Single Chimera block - either recurrent or attention based.

    Pattern: Pre-norm residual with either:
    - Gated Linear Recurrence (for global context)
    - Sliding Window GQA (for local precision)

    Both followed by SwiGLU FFN.
    """

    def __init__(self, config: ChimeraConfig, use_attention: bool):
        super().__init__()
        self.use_attention = use_attention

        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)

        if use_attention:
            self.temporal_mix = SlidingWindowGQA(config)
        else:
            self.temporal_mix = GatedLinearRecurrence(config.d_model)

        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(self, x: torch.Tensor,
                cache: Optional[torch.Tensor] = None,
                position_offset: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            cache: Previous state (KV cache for attention, hidden state for recurrence)
            position_offset: Position offset for RoPE
            use_cache: Whether to return cache for next step

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated cache if use_cache else None
        """
        # Temporal mixing (attention or recurrence)
        residual = x
        x = self.norm1(x)

        if self.use_attention:
            x, new_cache = self.temporal_mix(x, cache, position_offset, use_cache)
        else:
            x, new_cache = self.temporal_mix(x, cache, use_cache)

        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_cache


class Chimera(nn.Module):
    """
    Chimera: Hybrid Recurrent-Attention Language Model

    A novel architecture combining:
    - Gated linear recurrence for efficient global context (O(1) memory)
    - Sliding window GQA for local retrieval precision
    - SwiGLU FFN
    - RoPE position embeddings

    Layer pattern: Every Nth layer uses attention, others use recurrence.
    Default: 3:1 ratio (recurrence:attention) proven effective in Griffin/Jamba.
    """

    def __init__(self, config: ChimeraConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Build layers with recurrence/attention pattern
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            # Place attention every N layers (e.g., layers 2, 5, 8, 11... for N=3)
            use_attention = (i + 1) % config.attention_every_n == 0
            self.layers.append(ChimeraBlock(config, use_attention))

        # Final norm and output projection
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (reduces params, standard practice)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Print architecture summary
        n_attention = sum(1 for l in self.layers if l.use_attention)
        n_recurrent = config.n_layers - n_attention
        print(f"Chimera initialized: {config.n_layers} layers "
              f"({n_recurrent} recurrent, {n_attention} attention)")

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                cache: Optional[list] = None,
                position_offset: int = 0,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through Chimera.

        Args:
            input_ids: Token IDs [batch, seq_len]
            cache: List of layer caches for inference
            position_offset: Position offset for RoPE
            use_cache: Whether to return caches

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_cache: List of updated caches if use_cache
        """
        x = self.embed_tokens(input_ids)

        new_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(x, layer_cache, position_offset, use_cache)
            if use_cache:
                new_cache.append(layer_new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_cache

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Model Size Configurations
# ============================================================================

def chimera_small() -> ChimeraConfig:
    """~125M params - for testing and small experiments."""
    return ChimeraConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=32000,
    )

def chimera_medium() -> ChimeraConfig:
    """~350M params - for development."""
    return ChimeraConfig(
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=32000,
    )

def chimera_base() -> ChimeraConfig:
    """~1.5B params - main model for laptop deployment."""
    return ChimeraConfig(
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=32000,
    )

def chimera_large() -> ChimeraConfig:
    """~3B params - for stronger performance with more VRAM."""
    return ChimeraConfig(
        d_model=2560,
        n_layers=32,
        n_heads=20,
        n_kv_heads=4,
        vocab_size=32000,
    )


def chimera_deep() -> ChimeraConfig:
    """
    ~135M params - Deep architecture for creative writing (laptop-optimized).

    16 layers with moderate width (704 d_model).
    Optimized for:
    - 6GB VRAM (RTX 4050 laptop)
    - ~5GB peak memory leaves headroom
    - Style/voice development with good depth
    - Verbose, eloquent prose generation

    Layer pattern: 12 recurrent + 4 attention (3:1 ratio)
    """
    return ChimeraConfig(
        d_model=704,
        n_layers=16,
        n_heads=11,  # 704/11 = 64 head_dim
        n_kv_heads=1,  # Aggressive GQA for memory
        vocab_size=32000,
        max_seq_len=2048,
        sliding_window=512,
        attention_every_n=4,
    )


def chimera_abyss() -> ChimeraConfig:
    """
    ~2.1B params - EXTREME DEPTH for A100 80GB creative writing.

    96 layers with solid width (1280 d_model).
    This is the deepest Chimera variant - pure vertical scaling.

    Optimized for:
    - A100 80GB VRAM (~40-50GB peak usage, plenty of headroom)
    - Maximum depth for complex reasoning chains
    - Rich stylistic development across many layers
    - Nuanced, eloquent prose with deep representations

    Layer pattern: 72 recurrent + 24 attention (3:1 ratio)
    - Recurrent layers: global context, style, voice
    - Attention layers: local precision, coherence
    """
    return ChimeraConfig(
        d_model=1280,
        n_layers=96,
        n_heads=20,       # 1280/20 = 64 head_dim
        n_kv_heads=5,     # 4:1 GQA ratio
        vocab_size=32000,
        max_seq_len=8192, # Long context for A100
        sliding_window=2048,
        attention_every_n=4,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing Chimera architecture...")

    config = chimera_small()
    model = Chimera(config)

    # Count parameters
    num_params = model.get_num_params()
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")
