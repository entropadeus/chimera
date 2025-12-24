# Chimera LLM - Development Guide

## Project Overview
Chimera is a hybrid recurrent-attention LLM trained on TinyStories.
After instruction fine-tuning, it becomes **Wyrd** - the chat assistant.

## Current Status (Medium Model - 230M params)
- Pretraining: In progress on Colab (L4 GPU)
- Target: Loss < 2.0, PPL < 7, then fine-tune

## Generation Tests (Step 500)

| Prompt | Output |
|--------|--------|
| "Once upon a time" | *"...there was a little girl named Lily. One day, she went to the park to play with her friends. But when they arrived, she saw a big dog running towards her."* |
| "Who are you?" | *"Who can you help me? I want to be your friend."* (base model doesn't understand questions yet) |
| "Tom found a mysterious key" | *"...he thought it would be fun to open the key and see what was inside."* |
| "Lily and Whiskers went on an adventure" | *"...they found a big, shiny rock. Lily picked it up and took it from Ben."* |

**Observations at step 500:**
- Learning TinyStories patterns (simple sentences, kid names)
- Generations are short but coherent
- Some logic gaps (characters appearing from nowhere)
- By step 2000-3000 quality improves significantly

## Training Pipeline

### Step 1: Pretraining (COMPLETE when loss < 2.0, PPL < 7)
```bash
python -u train_packed.py --resume checkpoints/latest.pt --max-steps 2000
```

### Step 2: Generate Instruction Data
After pretraining completes, generate conversational training data:
```bash
python create_instruct_data.py --input data/tinystories.txt --output data/instruct_data.jsonl --max-stories 5000
```

### Step 3: Instruction Fine-tuning
Fine-tune the pretrained model to follow instructions:
```bash
python train_instruct.py --model-path checkpoints/model_1500.pt --data-path data/instruct_data.jsonl --epochs 2 --lr 2e-5
```

### Step 4: Test Generation
Test the model with the chat UI (Wyrd):
```bash
python chat_ui.py --checkpoint checkpoints/model_1500.pt --port 5000 --host 0.0.0.0
```
- Local: http://127.0.0.1:5000
- Network: http://192.168.0.120:5000
- TTS: Browser-based (Web Speech API)

Or quick generation test:
```bash
python generate.py --checkpoint checkpoints/model_1500.pt --prompt "Once upon a time"
```

## Key Files
- `train_packed.py` - Pretraining script (packed sequences)
- `train_instruct.py` - Instruction fine-tuning
- `create_instruct_data.py` - Generate instruction dataset from TinyStories
- `generate.py` - Text generation / inference
- `chat_ui.py` - Interactive chat interface (Wyrd)
- `model.py` - Chimera architecture definition
- `tts.py` - VibeVoice TTS integration (optional, not currently used)

## Checkpoints
- `checkpoints/latest.pt` - Latest pretraining checkpoint (full state with optimizer)
- `checkpoints/step_*.pt` - Training checkpoints (full state)
- `checkpoints/model_*.pt` - Model weights only (for inference)
- `checkpoints/instruct_final.pt` - Instruction-tuned model (after fine-tuning)

To extract model weights from a training checkpoint:
```python
import torch
from train_packed import TrainConfig
ckpt = torch.load('checkpoints/step_1500.pt', map_location='cpu')
torch.save(ckpt['model'], 'checkpoints/model_1500.pt')
```

## Training Tips
- Use `--compile` flag for 20-40% speedup (PyTorch 2.0+)
- Increase `--micro-batch-size` to use more VRAM (target 85-90%)
- Use `-u` flag with python for unbuffered output
- Pretraining: LR 3e-4, Instruct: LR 2e-5 (10x lower)

## Architecture Summary

### Chimera Medium (current)
- 230M parameters
- 16 layers: 11 recurrent (RG-LRU) + 5 attention (sliding window GQA)
- 1024 hidden dim, 16 heads
- 32k vocab (TinyLlama tokenizer)
- Hybrid design: recurrence for global context, attention for local precision

### Available Configs
| Config | Params | Layers | d_model |
|--------|--------|--------|---------|
| small | 106M | 12 (8R+4A) | 768 |
| medium | 230M | 16 (11R+5A) | 1024 |
| base | 1.5B | 24 (18R+6A) | 2048 |
