# Chimera: Self-Optimizing Hybrid LLM

A novel architecture synthesizing proven innovations from 2023-2025 LLM research, with **evolutionary self-optimization** to discover the best architecture for your hardware and data.

## Architecture

Chimera combines:

| Component | Source | Innovation |
|-----------|--------|------------|
| **Gated Linear Recurrence** | Griffin (Google) | O(1) memory global context modeling |
| **Sliding Window GQA** | Llama/Griffin | Local precision with efficient KV caching |
| **SwiGLU FFN** | Llama/Mistral/PaLM | Highest ROI activation upgrade |
| **RoPE + NTK scaling** | Llama/Phi-3 | Best position encoding, long context |
| **3:1 Layer Ratio** | Jamba/Griffin | Proven hybrid recurrence:attention balance |

### Why This Design?

- **Mamba/SSMs** excel at long context but struggle with precise retrieval
- **Transformers** have quadratic attention but dominate on quality
- **Hybrid approaches** (Jamba, Griffin, RecurrentGemma) prove you can have both
- **GQA** gives near-MHA quality with MQA speed (used by Llama 2/3, Mistral)
- **SwiGLU** is standard in every SOTA model for good reason

## Model Sizes

| Config | Params | Layers | d_model | Target Use |
|--------|--------|--------|---------|------------|
| small | ~125M | 12 | 768 | Testing, experiments |
| base | ~1.5B | 24 | 2048 | Laptop deployment |
| large | ~3B | 32 | 2560 | Strong performance |

## Quick Start

```python
from chimera import Chimera, chimera_small, ChimeraTokenizer, ChimeraGenerator

# Create model
config = chimera_small()
model = Chimera(config)

# Setup generator
tokenizer = ChimeraTokenizer()
generator = ChimeraGenerator(model, tokenizer, device="cpu")

# Generate
output = generator.generate(
    "The future of AI is",
    max_new_tokens=100,
    temperature=0.7,
)
print(output)
```

## Self-Optimizing Evolution

**Let Chimera find the optimal architecture for your setup:**

```bash
# Run 1-hour evolution to find best architecture
python chimera_evolve.py --data-path data/tinystories.txt --budget 1h

# Longer evolution for better results
python chimera_evolve.py --budget 4h --max-generations 20
```

The evolutionary system:
- Mutates architecture genes (dimensions, layers, recurrence ratio, etc.)
- Evaluates candidates through short training runs
- Breeds successful architectures via crossover
- Automatically adapts to your GPU/CPU and data

## Training

**Efficient packed-sequence training:**

```bash
# New efficient trainer with gradient accumulation
python train_packed.py \
    --model-config small \
    --data-path data/tinystories.txt \
    --micro-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-steps 5000

# Legacy trainer
python train.py \
    --model-config small \
    --data-path data/train.txt \
    --batch-size 4 \
    --max-steps 10000
```

## Inference

```bash
# Single prompt
python generate.py --config small --prompt "Hello world"

# Interactive chat
python generate.py --config small --interactive

# With quantization (faster on CPU)
python generate.py --config small --quantize 8 --interactive
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers (for tokenizer, optional)

```bash
pip install -r requirements.txt
```

## Evolvable Architecture Genes

The evolutionary system can mutate these architecture components:

| Gene | Options | Effect |
|------|---------|--------|
| `d_model` | 512-2048 | Model width |
| `n_layers` | 6-24 | Model depth |
| `n_heads` | 4-16 | Attention heads |
| `n_kv_heads` | 1-8 | GQA key-value heads |
| `ffn_multiplier` | 2.0-4.0 | FFN hidden size ratio |
| `recurrence_ratio` | 1-5 | Recurrent:Attention layers |
| `window_size` | 256-2048 | Sliding window size |
| `dropout` | 0.0-0.15 | Regularization |

## Project Structure

```
chimera/
  model.py          # Core architecture (Chimera, GatedLinearRecurrence, SlidingWindowGQA)
  tokenizer.py      # HuggingFace tokenizer wrapper
  train.py          # Basic trainer
  train_packed.py   # Efficient packed-sequence trainer
  evolution.py      # Mutation operators, crossover, fitness evaluation
  chimera_evolve.py # Self-optimizing orchestrator
  generate.py       # Inference and generation
```

## Research References

- [Griffin: Mixing Gated Linear Recurrences with Local Attention](https://arxiv.org/abs/2402.19427)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [GQA: Training Generalized Multi-Query Transformers](https://arxiv.org/abs/2305.13245)
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
