import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from einops import rearrange

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class AetherConfig:
    vocab_size: int = 32000
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1
    num_templates: int = 32
    use_gradient_checkpointing: bool = False
    rope_base: int = 10000
    ffn_mult: float = 4.0
    ffn_multiple_of: int = 256
    template_orthogonal_init: bool = True
    template_normalize: bool = True

# -----------------------------------------------------------------------------
# Compatibility Layer
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm for PyTorch < 2.4 compatibility."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


# -----------------------------------------------------------------------------
# Rotary Embeddings
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE with lazy device-aware caching.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = 0

        # Don't register inv_freq as buffer - compute on demand
        self._inv_freq: Optional[torch.Tensor] = None
        self._freqs_cis: Optional[torch.Tensor] = None
        self._target_device: torch.device = torch.device('cpu')
        self._target_dtype: torch.dtype = torch.float32

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self.max_seq_len_cached and device == self._target_device:
            return

        self._target_device = device
        self._target_dtype = dtype
        self.max_seq_len_cached = seq_len

        inv_freq = self._compute_inv_freq(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self._freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self._build_cache(seq_len, device, dtype)
        return self._freqs_cis[:seq_len]


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE. Inputs: [B, S, H, D]
    """
    orig_dtype = xq.dtype

    # Complex view requires float32
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # [S, D/2] -> [1, S, 1, D/2]
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, -1)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.to(orig_dtype), xk_out.to(orig_dtype)


# -----------------------------------------------------------------------------
# Feed-Forward
# -----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(dim * hidden_mult * 2 / 3)  # SwiGLU correction factor
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -----------------------------------------------------------------------------
# Holographic Attention
# -----------------------------------------------------------------------------

class HolographicAttention(nn.Module):
    """
    Holographic Binding Attention with template-based correlation scoring.

    Key insight: Instead of collapsing binding info via sum, we use a learned
    aggregation that preserves more signal.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        num_templates: int = 32,
        orthogonal_init: bool = True,
        normalize_templates: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_templates = min(self.head_dim, num_templates)
        self.normalize_templates = normalize_templates

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim)

        # Templates: [H, T, D]
        templates = torch.randn(num_heads, self.num_templates, self.head_dim)
        if orthogonal_init and self.num_templates <= self.head_dim:
            # Initialize templates as orthogonal basis
            for h in range(num_heads):
                nn.init.orthogonal_(templates[h])
        else:
            templates *= 0.02
        self.templates = nn.Parameter(templates)

        # Learned aggregation instead of blind sum
        # Projects from D -> 1 per template, but learnable
        self.bind_proj = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, _ = x.shape

        q = rearrange(self.q_proj(x), 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b s (h d) -> b s h d', h=self.num_heads)

        # Apply RoPE
        freqs_cis = self.rotary(s, x.device, x.dtype)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV cache handling
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv_cache = (k, v)

        # Transpose for attention: [B, H, S, D]
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        attn_scores = self._holographic_scores(q, k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h s d -> b s (h d)')

        return self.o_proj(out), new_kv_cache

    def _holographic_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute attention via holographic template binding.

        q, k: [B, H, S, D]
        """
        b, h, s_q, d = q.shape
        s_k = k.size(2)

        # Normalize templates for stable training (optional but recommended)
        templates = self.templates
        if self.normalize_templates:
            templates = F.normalize(templates, dim=-1)

        # Float32 for FFT stability
        q_f = q.float()
        k_f = k.float()
        t_f = templates.float()
        bind_proj = self.bind_proj.float()

        # FFT
        q_freq = torch.fft.rfft(q_f, dim=-1, norm="ortho")
        k_freq = torch.fft.rfft(k_f, dim=-1, norm="ortho")
        t_freq = torch.fft.rfft(t_f, dim=-1, norm="ortho")

        # Bind via correlation: Q ⊛ T*, K ⊛ T*
        # q_freq: [B, H, S_q, D_f] -> [B, H, S_q, 1, D_f]
        # t_freq: [H, T, D_f] -> [1, H, 1, T, D_f]
        t_conj = torch.conj(t_freq).unsqueeze(0).unsqueeze(2)

        q_bind_freq = q_freq.unsqueeze(3) * t_conj
        k_bind_freq = k_freq.unsqueeze(3) * t_conj

        # IFFT back
        q_bind = torch.fft.irfft(q_bind_freq, n=d, dim=-1)  # [B, H, S_q, T, D]
        k_bind = torch.fft.irfft(k_bind_freq, n=d, dim=-1)  # [B, H, S_k, T, D]

        # Learned projection instead of blind sum
        # bind_proj: [H, D] -> [1, H, 1, 1, D]
        proj = bind_proj.view(1, h, 1, 1, d)

        q_sig = (q_bind * proj).sum(dim=-1)  # [B, H, S_q, T]
        k_sig = (k_bind * proj).sum(dim=-1)  # [B, H, S_k, T]

        # Normalize signatures for stable dot products
        q_sig = F.normalize(q_sig, dim=-1)
        k_sig = F.normalize(k_sig, dim=-1)

        # Score: [B, H, S_q, T] @ [B, H, T, S_k] -> [B, H, S_q, S_k]
        scores = torch.matmul(q_sig, k_sig.transpose(-1, -2))

        # Scale by sqrt(num_templates) since we normalized
        return (scores * math.sqrt(self.num_templates) * self.scale).to(q.dtype)


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------

class AetherBlock(nn.Module):
    def __init__(self, config: AetherConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1 = RMSNorm(config.embed_dim)
        self.attn = HolographicAttention(
            config.embed_dim,
            config.num_heads,
            config.dropout,
            config.num_templates,
            config.template_orthogonal_init,
            config.template_normalize,
        )
        self.ln2 = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config.embed_dim, config.ffn_mult, config.ffn_multiple_of)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv_cache = self.attn(self.ln1(x), mask, kv_cache)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, new_kv_cache


# -----------------------------------------------------------------------------
# Full Model
# -----------------------------------------------------------------------------

class AetherLM(nn.Module):
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            AetherBlock(config, i) for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Causal mask cache
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
            persistent=False
        )

        self.apply(self._init_weights)

        # Gradient checkpointing
        self.gradient_checkpointing = config.use_gradient_checkpointing

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or expand causal mask."""
        if seq_len > self.causal_mask.size(0):
            # Expand mask
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            self.register_buffer("causal_mask", mask, persistent=False)
        return self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        b, s = input_ids.shape

        # For cached inference, we only need mask for new tokens attending to all past
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].size(1)
            total_len = past_len + s
            mask = self._get_causal_mask(total_len, input_ids.device)
            mask = mask[:, :, -s:, :total_len]
        else:
            mask = self._get_causal_mask(s, input_ids.device)

        x = self.embed_dropout(self.embed(input_ids))

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None

            if self.gradient_checkpointing and self.training:
                x, new_cache = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, layer_cache, use_reentrant=False
                )
            else:
                x, new_cache = layer(x, mask, layer_cache)

            if use_cache:
                new_kv_cache.append(new_cache)

        x = self.norm(x)
        logits = self.head(x)

        output = {"logits": logits}

        if use_cache:
            output["kv_cache"] = new_kv_cache

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100  # Standard HF convention
            )

        return output

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.
        """
        kv_cache = [None] * self.config.num_layers

        # Prefill: process initial context
        if idx.size(1) > 1:
            output = self(idx[:, :-1], kv_cache=None, use_cache=True)
            kv_cache = output["kv_cache"]
            idx_input = idx[:, -1:]
        else:
            idx_input = idx

        for _ in range(max_new_tokens):
            output = self(idx_input, kv_cache=kv_cache, use_cache=True)
            kv_cache = output["kv_cache"]

            logits = output["logits"][:, -1, :] / max(temperature, 1e-5)

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                indices_to_remove = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            idx_input = idx_next

        return idx

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
        return n


# -----------------------------------------------------------------------------
# Size Configs (matching Chimera param counts)
# -----------------------------------------------------------------------------

def aether_deep() -> AetherConfig:
    """~160M params - Deep for creative writing."""
    return AetherConfig(
        vocab_size=32000,
        embed_dim=768,
        num_layers=24,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.0,
        num_templates=32,
        use_gradient_checkpointing=False,
    )


def aether_medium() -> AetherConfig:
    """~100M params - Medium size."""
    return AetherConfig(
        vocab_size=32000,
        embed_dim=640,
        num_layers=16,
        num_heads=10,
        max_seq_len=2048,
        dropout=0.0,
        num_templates=32,
    )


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AetherLM Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = aether_deep()
    model = AetherLM(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    x = torch.randint(0, config.vocab_size, (2, 512), device=device)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(x, labels=x)

    print(f"Loss: {out['loss'].item():.4f}")
    print(f"NaN: {torch.isnan(out['loss']).item()}")
