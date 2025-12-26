"""
Transformer: A Clean Implementation from Scratch
=================================================
A tried-and-true decoder-only transformer with modern best practices.

Architecture:
- Pre-norm residual connections (RMSNorm)
- Multi-head self-attention with causal masking
- Rotary Position Embeddings (RoPE)
- SwiGLU feed-forward network
- Optional Grouped Query Attention (GQA) for efficiency
- Weight tying between embedding and output

Based on the proven architecture from:
- GPT-2/3 (decoder-only autoregressive)
- Llama 1/2/3 (RMSNorm, RoPE, SwiGLU, GQA)
- Mistral (sliding window optional, GQA)

This is the architecture that powers most modern LLMs.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4  # For GQA (set equal to n_heads for standard MHA)
    vocab_size: int = 32000
    max_seq_len: int = 2048
    ffn_hidden_mult: float = 8/3  # SwiGLU uses 8/3 ratio (standard)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        self.ffn_hidden = int(self.d_model * self.ffn_hidden_mult)
        # Round to nearest multiple of 256 for efficiency
        self.ffn_hidden = ((self.ffn_hidden + 255) // 256) * 256
        self.head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


# Backward compatibility alias
ChimeraConfig = TransformerConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    More efficient than LayerNorm (no mean subtraction, no bias).
    Used in Llama, Mistral, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information by rotating query/key vectors.
    Benefits: relative position awareness, extrapolates to longer sequences.
    Used in Llama, Mistral, PaLM, and most modern LLMs.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute cos/sin cache for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        """Return cos/sin for the given sequence length."""
        if seq_len + offset > self.cos_cached.size(0):
            self._build_cache(seq_len + offset)

        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # cos/sin are [seq_len, head_dim], need to broadcast to [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Formula: (Swish(xW1) * xW3) @ W2

    The gated linear unit with Swish activation.
    Empirically better than GELU FFN at same parameter count.
    Used in Llama, Mistral, PaLM.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class Attention(nn.Module):
    """
    Multi-Head Self-Attention with optional Grouped Query Attention (GQA).

    Features:
    - Causal (autoregressive) masking
    - RoPE position encoding
    - GQA for memory efficiency (set n_kv_heads < n_heads)
    - KV-cache for efficient inference

    GQA groups multiple query heads to share the same key-value heads,
    reducing KV-cache memory with minimal quality loss.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repeat factor

        # Projections
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        # RoPE
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        batch, n_kv_heads, seq_len, head_dim = x.shape
        if self.n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV-cache for inference.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Cached (keys, values) from previous steps
            position_offset: Position offset for RoPE (for cached inference)
            use_cache: Whether to return updated KV-cache

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated KV-cache if use_cache else None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(seq_len, position_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        q_len = q.size(2)
        kv_len = k.size(2)
        causal_mask = torch.triu(
            torch.full((q_len, kv_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=kv_len - q_len + 1
        )
        attn_weights = attn_weights + causal_mask

        # Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, new_cache


class TransformerBlock(nn.Module):
    """
    Single Transformer block.

    Architecture (Pre-Norm):
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> +
        |___________________________|  |___________________|
                  residual                   residual

    Pre-norm is more stable for deep networks than post-norm.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            cache: KV-cache from previous steps
            position_offset: Position offset for RoPE
            use_cache: Whether to return updated cache

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated cache if use_cache else None
        """
        # Attention with residual
        residual = x
        x = self.attention_norm(x)
        x, new_cache = self.attention(x, cache, position_offset, use_cache)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_cache


class Transformer(nn.Module):
    """
    Decoder-only Transformer Language Model.

    A clean implementation following modern best practices:
    - Pre-norm with RMSNorm
    - RoPE position encoding
    - SwiGLU FFN
    - Optional GQA for efficiency
    - Weight tying

    This is the architecture that powers GPT, Llama, Mistral, etc.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm and output
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Print summary
        print(f"Transformer initialized: {config.n_layers} layers, "
              f"d_model={config.d_model}, heads={config.n_heads}")

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list] = None,
        position_offset: int = 0,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            cache: List of KV-caches for each layer
            position_offset: Position offset for cached inference
            use_cache: Whether to return updated caches

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


# Backward compatibility alias
Chimera = Transformer
ChimeraBlock = TransformerBlock


# ============================================================================
# Model Size Configurations
# ============================================================================

def transformer_small() -> TransformerConfig:
    """~85M params - for testing and quick experiments."""
    return TransformerConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,  # Standard MHA
        vocab_size=32000,
        max_seq_len=2048,
    )


def transformer_medium() -> TransformerConfig:
    """~350M params - for development and Colab training."""
    return TransformerConfig(
        d_model=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,  # GQA for efficiency
        vocab_size=32000,
        max_seq_len=2048,
    )


def transformer_base() -> TransformerConfig:
    """~1B params - for serious training."""
    return TransformerConfig(
        d_model=1536,
        n_layers=24,
        n_heads=24,
        n_kv_heads=6,  # GQA
        vocab_size=32000,
        max_seq_len=4096,
    )


def transformer_large() -> TransformerConfig:
    """~3B params - for high-quality results."""
    return TransformerConfig(
        d_model=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,  # GQA
        vocab_size=32000,
        max_seq_len=4096,
    )


# Backward compatibility aliases
chimera_small = transformer_small
chimera_medium = transformer_medium
chimera_base = transformer_base
chimera_large = transformer_large


def chimera_deep() -> TransformerConfig:
    """~150M params - deep but narrow for laptop training."""
    return TransformerConfig(
        d_model=640,
        n_layers=20,
        n_heads=10,
        n_kv_heads=2,  # Aggressive GQA
        vocab_size=32000,
        max_seq_len=2048,
    )


def chimera_abyss() -> TransformerConfig:
    """~2B params - large model for A100 training."""
    return TransformerConfig(
        d_model=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=32000,
        max_seq_len=8192,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Transformer Architecture")
    print("=" * 60)

    config = transformer_small()
    model = Transformer(config)

    num_params = model.get_num_params()
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Config: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")

    # Test with cache (inference mode)
    print("\nTesting cached inference...")
    with torch.no_grad():
        # First token
        logits, cache = model(input_ids[:, :1], use_cache=True)
        # Next tokens
        for i in range(1, 5):
            logits, cache = model(input_ids[:, i:i+1], cache=cache,
                                  position_offset=i, use_cache=True)
    print("Cached inference successful!")
