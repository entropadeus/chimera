"""
Chimera: A Novel Hybrid LLM Architecture
=========================================

Combines proven innovations from 2023-2025 research:
- Gated Linear Recurrence (RG-LRU from Griffin) for O(1) memory global context
- Sliding Window GQA for local retrieval precision
- SwiGLU FFN (standard in Llama/Mistral/PaLM)
- RoPE with NTK-aware scaling for long context
- 3:1 Recurrent:Attention layer ratio (proven in Jamba/Griffin)

Usage:
    from chimera import Chimera, ChimeraConfig, chimera_small, chimera_base
    from chimera import ChimeraTokenizer
    from chimera import ChimeraGenerator

    # Create model
    config = chimera_small()
    model = Chimera(config)

    # Generate text
    tokenizer = ChimeraTokenizer()
    generator = ChimeraGenerator(model, tokenizer)
    output = generator.generate("Once upon a time")
"""

from .model import (
    Chimera,
    ChimeraConfig,
    ChimeraBlock,
    SwiGLU,
    GatedLinearRecurrence,
    SlidingWindowGQA,
    RMSNorm,
    RotaryEmbedding,
    chimera_small,
    chimera_base,
    chimera_medium,
    chimera_large,
)

from .tokenizer import ChimeraTokenizer

from .generate import ChimeraGenerator, load_model

__version__ = "0.1.0"
__all__ = [
    # Model
    "Chimera",
    "ChimeraConfig",
    "ChimeraBlock",
    "SwiGLU",
    "GatedLinearRecurrence",
    "SlidingWindowGQA",
    "RMSNorm",
    "RotaryEmbedding",
    # Configs
    "chimera_small",
    "chimera_base",
    "chimera_medium",
    "chimera_large",
    # Tokenizer
    "ChimeraTokenizer",
    # Generation
    "ChimeraGenerator",
    "load_model",
]
