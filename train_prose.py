"""
Prose Fine-tuning for Chimera - Stage 1 & 2 Hybrid Training

Stage 1: Train on eloquent prose (establish linguistic register)
Stage 2: Fine-tune on quality narrative (recover story structure)

Optimized for RTX 4050 (6GB VRAM) + 12GB RAM:
- Gradient checkpointing
- Mixed precision (FP16)
- Aggressive gradient accumulation
- Optional CPU offloading for optimizer states
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Generator
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

from model import Chimera, ChimeraConfig, chimera_small, chimera_medium, chimera_base
from tokenizer import ChimeraTokenizer


@dataclass
class ProseConfig:
    """Configuration for prose fine-tuning."""
    # Model
    model_path: str = "checkpoints/model.pt"
    model_config: str = "medium"

    # Data
    data_path: str = "data/prose/eloquent_merged.txt"
    seq_length: int = 512  # Keep shorter for VRAM
    stride: int = 256  # Overlap for sliding window

    # Training - Memory Optimized
    batch_size: int = 1  # Tiny batch for 6GB VRAM
    gradient_accumulation_steps: int = 32  # Effective batch = 32
    learning_rate: float = 1e-5  # Lower than pretraining
    min_lr: float = 1e-7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_epochs: int = 2
    max_steps: int = -1  # -1 = train full epochs
    grad_clip: float = 1.0

    # Memory Optimization
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload_optimizer: bool = False  # Set True if still OOM
    empty_cache_freq: int = 50  # Clear CUDA cache every N steps

    # Output
    output_dir: str = "checkpoints"
    save_every: int = 500
    log_every: int = 10
    eval_ratio: float = 0.05


# =============================================================================
# Gradient Checkpointing Wrapper
# =============================================================================

def enable_gradient_checkpointing(model: Chimera):
    """
    Enable gradient checkpointing to trade compute for memory.
    Recomputes activations during backward pass instead of storing them.
    """
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module._original_forward(*inputs)
        return custom_forward

    for layer in model.layers:
        layer._original_forward = layer.forward

        def make_checkpointed_forward(layer):
            def forward(x, cache=None, position_offset=0, use_cache=False):
                # Can't checkpoint when using cache (inference)
                if use_cache:
                    return layer._original_forward(x, cache, position_offset, use_cache)

                # Checkpoint the forward pass
                def run_layer(x_inner):
                    return layer._original_forward(x_inner, None, position_offset, False)[0]

                output = torch.utils.checkpoint.checkpoint(
                    run_layer, x, use_reentrant=False
                )
                return output, None
            return forward

        layer.forward = make_checkpointed_forward(layer)

    print("Gradient checkpointing enabled")


# =============================================================================
# Dataset
# =============================================================================

class ProseDataset(Dataset):
    """
    Dataset for prose language modeling.
    Uses sliding window with overlap for long texts.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ChimeraTokenizer,
        seq_length: int = 512,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.chunks: List[torch.Tensor] = []

        print(f"Loading prose from {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Text size: {len(text):,} characters")

        # Tokenize entire text
        print("Tokenizing...")
        tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        print(f"Token count: {len(tokens):,}")

        # Create overlapping chunks
        print(f"Creating chunks (seq_length={seq_length}, stride={stride})...")

        for start in range(0, len(tokens) - seq_length, stride):
            chunk = tokens[start:start + seq_length + 1]  # +1 for next-token prediction
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Created {len(self.chunks):,} training chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class StreamingProseDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for very large text files.
    Reads and tokenizes on-the-fly.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ChimeraTokenizer,
        seq_length: int = 512,
        stride: int = 256,
        buffer_size: int = 100000,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        self.buffer_size = buffer_size

    def __iter__(self) -> Generator:
        token_buffer = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            while True:
                text = f.read(self.buffer_size)
                if not text:
                    break

                tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                token_buffer.extend(tokens)

                while len(token_buffer) >= self.seq_length + 1:
                    chunk = token_buffer[:self.seq_length + 1]
                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }
                    token_buffer = token_buffer[self.stride:]


# =============================================================================
# CPU Offload Optimizer (for extreme memory constraints)
# =============================================================================

class CPUOffloadOptimizer:
    """
    Optimizer that keeps states on CPU, only moving to GPU for updates.
    Slower but uses much less VRAM.
    """

    def __init__(self, params, lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95)):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

        # Initialize optimizer states on CPU
        self.state = {}
        for i, p in enumerate(self.params):
            if p.requires_grad:
                self.state[i] = {
                    'm': torch.zeros_like(p, device='cpu'),
                    'v': torch.zeros_like(p, device='cpu'),
                    't': 0,
                }

    def step(self):
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            state = self.state[i]
            state['t'] += 1

            # Move grad to CPU
            grad = p.grad.float().cpu()

            # Update moments on CPU
            state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
            state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            m_hat = state['m'] / (1 - beta1 ** state['t'])
            v_hat = state['v'] / (1 - beta2 ** state['t'])

            # Compute update on CPU
            update = m_hat / (v_hat.sqrt() + 1e-8)

            # Add weight decay
            if self.weight_decay > 0:
                update.add_(p.data.float().cpu(), alpha=self.weight_decay)

            # Apply update (move back to GPU)
            p.data.add_(update.to(p.device).to(p.dtype), alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


# =============================================================================
# Training Loop
# =============================================================================

def evaluate(model, dataloader, device, max_batches: int = 50):
    """Quick evaluation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def train(config: ProseConfig):
    """Main prose fine-tuning loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_mem:.1f} GB")

        # Set memory-efficient settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = ChimeraTokenizer()

    # Model
    model_configs = {"small": chimera_small, "medium": chimera_medium, "base": chimera_base}
    model_cfg = model_configs.get(config.model_config, chimera_medium)()
    model_cfg.vocab_size = tokenizer.vocab_size

    model = Chimera(model_cfg)

    # Load pretrained weights
    print(f"Loading pretrained weights from {config.model_path}...")
    state_dict = torch.load(config.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Enable gradient checkpointing BEFORE moving to GPU
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Memory stats after model load
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after model load: {allocated:.2f} GB")

    # Dataset
    print(f"\nLoading data from {config.data_path}...")
    dataset = ProseDataset(
        config.data_path,
        tokenizer,
        config.seq_length,
        config.stride
    )

    # Train/eval split
    eval_size = max(100, int(len(dataset) * config.eval_ratio))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train chunks: {len(train_dataset):,}")
    print(f"Eval chunks: {len(eval_dataset):,}")

    # Calculate training steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs

    warmup_steps = int(total_steps * config.warmup_ratio)

    print(f"\nTraining plan:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    # Optimizer
    if config.cpu_offload_optimizer:
        print("Using CPU offload optimizer (slower but saves VRAM)")
        optimizer = CPUOffloadOptimizer(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = None  # Can't use with CPU offload
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

    # LR scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(config.min_lr / config.learning_rate, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if not config.cpu_offload_optimizer else None

    # Mixed precision
    use_amp = config.mixed_precision and device.type == 'cuda' and not config.cpu_offload_optimizer
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    autocast_dtype = torch.float16 if use_amp else torch.float32

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Prose Fine-tuning")
    print(f"Stage: {'Eloquent Prose' if 'eloquent' in config.data_path.lower() else 'Narrative Fiction'}")
    print(f"{'='*60}\n")

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    best_eval_loss = float('inf')
    start_time = time.time()

    for epoch in range(config.num_epochs):
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===\n")

        for batch in train_loader:
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * config.gradient_accumulation_steps
            micro_step += 1

            # Optimizer step
            if micro_step % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                elif config.cpu_offload_optimizer:
                    # Manual gradient clipping for CPU offload
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.float().pow(2).sum().item()
                    total_norm = total_norm ** 0.5
                    if total_norm > config.grad_clip:
                        scale = config.grad_clip / total_norm
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.data.mul_(scale)
                    optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                if not config.cpu_offload_optimizer:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                global_step += 1

                # Clear CUDA cache periodically
                if device.type == 'cuda' and global_step % config.empty_cache_freq == 0:
                    torch.cuda.empty_cache()

                # Logging
                if global_step % config.log_every == 0:
                    avg_loss = running_loss / config.log_every
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate

                    # Memory stats
                    if device.type == 'cuda':
                        mem_used = torch.cuda.memory_allocated() / 1e9
                        mem_str = f"| VRAM: {mem_used:.1f}GB"
                    else:
                        mem_str = ""

                    print(f"Step {global_step:5d}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 20)):.2f} | "
                          f"LR: {lr:.2e} {mem_str}")

                    running_loss = 0.0
                    start_time = time.time()

                # Save checkpoint
                if global_step % config.save_every == 0:
                    # Evaluate
                    eval_loss, eval_ppl = evaluate(model, eval_loader, device)
                    print(f"  [Eval] Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_path = output_dir / "prose_best.pt"
                        torch.save(model.state_dict(), best_path)
                        print(f"  [Eval] New best! Saved to {best_path}")

                    # Regular checkpoint
                    save_path = output_dir / f"prose_step_{global_step}.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved: {save_path}")

                    # Clean up old checkpoints (keep last 3)
                    checkpoints = sorted(output_dir.glob("prose_step_*.pt"), key=lambda x: x.stat().st_mtime)
                    for old_ckpt in checkpoints[:-3]:
                        old_ckpt.unlink()

    # Final evaluation
    print("\n" + "="*60)
    final_loss, final_ppl = evaluate(model, eval_loader, device)
    print(f"Final Eval - Loss: {final_loss:.4f} | PPL: {final_ppl:.2f}")

    # Final save
    final_path = output_dir / "prose_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {output_dir / 'prose_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Prose Fine-tuning for Chimera")

    # Model
    parser.add_argument("--model-path", type=str, default="checkpoints/model.pt",
                        help="Path to pretrained model weights")
    parser.add_argument("--model-config", type=str, default="medium",
                        choices=["small", "medium", "base"])

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to prose text file")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256,
                        help="Sliding window stride")

    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)

    # Memory optimization
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload optimizer to CPU (slower but saves VRAM)")
    parser.add_argument("--no-amp", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500)

    args = parser.parse_args()

    config = ProseConfig(
        model_path=args.model_path,
        model_config=args.model_config,
        data_path=args.data_path,
        seq_length=args.seq_length,
        stride=args.stride,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        cpu_offload_optimizer=args.cpu_offload,
        mixed_precision=not args.no_amp,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )

    train(config)


if __name__ == "__main__":
    main()
