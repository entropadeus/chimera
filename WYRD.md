# Wyrd

**Wyrd** is a conversational AI assistant built on the Chimera architecture. It's the instruction-tuned version of Chimera, fine-tuned to follow instructions and engage in natural dialogue.

## What It Is

- **Base Model**: Chimera (hybrid recurrent-attention LLM)
- **Parameters**: 230M (medium config)
- **Training Data**: TinyStories + OASST conversations
- **Purpose**: Lightweight, local-friendly chat assistant

## Architecture

Wyrd inherits Chimera's novel hybrid design:

| Component | Purpose |
|-----------|---------|
| **RG-LRU Layers** (11) | Gated linear recurrence for O(1) memory global context |
| **GQA Layers** (5) | Sliding window attention for local precision |
| **SwiGLU FFN** | Standard feed-forward (Llama/Mistral style) |
| **RoPE** | Rotary position embeddings |

The 3:1 recurrence-to-attention ratio balances efficiency with retrieval accuracy.

## Training Pipeline

```
TinyStories (raw text)
        │
        ▼
   [Pretraining]  ──►  Chimera (base model)
        │                    │
        │                    ▼
        │            [Fine-tuning on conversations]
        │                    │
        ▼                    ▼
   OASST data  ────────►   Wyrd (chat model)
```

1. **Pretraining**: Next-token prediction on TinyStories (~3000 steps)
2. **Fine-tuning**: ChatML-formatted conversations with loss masking

## Chat Format

Wyrd uses ChatML-style templates:

```
<|im_start|>system
You are Wyrd, a helpful AI assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
```

## Usage

**Generation:**
```bash
python generate.py --checkpoint checkpoints/instruct_final.pt --model-config medium --prompt "Hello!"
```

**Chat UI:**
```bash
python chat_ui.py --checkpoint checkpoints/instruct_final.pt --port 5000
```

## Name Origin

*Wyrd* (Old English: "fate" or "destiny") connects to the Norse concept of interconnected fate. It reflects the model's hybrid nature—weaving together recurrence and attention like threads of destiny.

## Limitations

- Small model (230M) - limited reasoning capacity
- Trained on children's stories - simple vocabulary and concepts
- Best for: simple Q&A, storytelling, basic assistance
- Not for: complex reasoning, factual accuracy, professional use

## Files

| File | Description |
|------|-------------|
| `model.py` | Chimera architecture |
| `train_instruct.py` | Fine-tuning script |
| `generate.py` | Text generation |
| `chat_ui.py` | Web interface |
| `checkpoints/instruct_final.pt` | Wyrd model weights |
