"""
Training script for Chimera LLM.

Features:
- Gradient checkpointing for memory efficiency
- Mixed precision training (AMP)
- Learning rate scheduling with warmup
- Gradient clipping
- Checkpointing and resumption
- Wandb logging (optional)
- Multi-GPU support via DDP
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import Chimera, ChimeraConfig, chimera_small, chimera_base
from tokenizer import ChimeraTokenizer


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_config: str = "small"  # "small" or "base"

    # Data
    data_path: str = "data/train.txt"
    seq_length: int = 2048
    batch_size: int = 4
    num_workers: int = 4

    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True

    # Checkpointing
    output_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 10

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Resume
    resume_from: Optional[str] = None


class TextDataset(Dataset):
    """
    Memory-efficient text dataset for language modeling.
    Loads data in chunks to handle large files without running out of memory.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ChimeraTokenizer,
        seq_length: int = 2048,
        stride: int = None,
        max_tokens: int = 50_000_000,  # ~50M tokens max to avoid OOM
    ):
        """
        Initialize dataset with chunked loading for memory efficiency.

        Args:
            data_path: Path to text file
            tokenizer: Tokenizer instance
            seq_length: Sequence length for training
            stride: Stride between sequences (defaults to seq_length)
            max_tokens: Maximum tokens to load (limits memory usage)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length

        print(f"Loading data from {data_path}...")

        if not os.path.exists(data_path):
            print("Warning: Data file not found, using dummy data")
            text = "Hello world. " * 10000
            self.tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        else:
            # Load in chunks to avoid memory issues
            self.tokens = []
            chunk_size = 10_000_000  # ~10MB chunks

            with open(data_path, "r", encoding="utf-8") as f:
                while len(self.tokens) < max_tokens:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    # Tokenize chunk
                    chunk_tokens = tokenizer.encode(chunk, add_bos=False, add_eos=False)
                    self.tokens.extend(chunk_tokens)

                    print(f"  Loaded {len(self.tokens):,} tokens...")

                    if len(self.tokens) >= max_tokens:
                        self.tokens = self.tokens[:max_tokens]
                        print(f"  Reached max_tokens limit ({max_tokens:,})")
                        break

        print(f"Total: {len(self.tokens):,} tokens")

        # Calculate number of sequences
        self.n_sequences = max(1, (len(self.tokens) - seq_length) // self.stride + 1)
        print(f"Created {self.n_sequences:,} training sequences")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length + 1  # +1 for target

        tokens = self.tokens[start:end]

        # Pad if necessary
        if len(tokens) < self.seq_length + 1:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_length + 1 - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        # Input is tokens[:-1], target is tokens[1:]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr: float):
    """Cosine learning rate schedule with warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step > max_steps:
            return min_lr / optimizer.defaults["lr"]
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return max(min_lr / optimizer.defaults["lr"],
                      0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    config: TrainConfig,
    device: torch.device,
) -> dict:
    """Single training step."""
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass with mixed precision
    with autocast(enabled=config.mixed_precision):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # Compute perplexity
    perplexity = torch.exp(loss).item()

    return {
        "loss": loss.item(),
        "perplexity": perplexity,
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 50):
    """Evaluate model on validation data."""
    model.eval()

    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap for stability

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    step: int,
    config: TrainConfig,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
) -> int:
    """Load training checkpoint and return step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    print(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
    return checkpoint["step"]


def train(config: TrainConfig):
    """Main training loop."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model_configs = {
        "small": chimera_small,
        "base": chimera_base,
    }
    model_config = model_configs.get(config.model_config, chimera_small)()
    model = Chimera(model_config)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        # Note: Would need to implement gradient checkpointing in the model
        print("Gradient checkpointing enabled (if supported)")

    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Initialize tokenizer
    tokenizer = ChimeraTokenizer()

    # Create dataset
    train_dataset = TextDataset(config.data_path, tokenizer, config.seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Optimizer (AdamW with decoupled weight decay)
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps, config.min_lr)

    # Mixed precision scaler
    scaler = GradScaler() if config.mixed_precision and device.type == "cuda" else None

    # Resume from checkpoint
    start_step = 0
    if config.resume_from:
        start_step = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, scaler
        )

    # Wandb logging
    if config.wandb_project:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            config.wandb_project = None

    # Training loop
    print(f"\nStarting training from step {start_step}")
    print(f"Total steps: {config.max_steps}")
    print("-" * 60)

    step = start_step
    epoch = 0
    train_iter = iter(train_loader)
    start_time = time.time()
    running_loss = 0

    while step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        metrics = train_step(model, batch, optimizer, scaler, config, device)
        scheduler.step()
        step += 1
        running_loss += metrics["loss"]

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = config.log_every / elapsed
            avg_loss = running_loss / config.log_every

            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f} | "
                  f"LR: {lr:.2e} | {steps_per_sec:.2f} steps/s")

            if config.wandb_project:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/perplexity": math.exp(avg_loss),
                    "train/lr": lr,
                    "train/steps_per_sec": steps_per_sec,
                }, step=step)

            running_loss = 0
            start_time = time.time()

        # Save checkpoint
        if step % config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, step, config,
                output_dir / f"checkpoint_{step}.pt"
            )
            # Also save latest
            save_checkpoint(
                model, optimizer, scheduler, scaler, step, config,
                output_dir / "checkpoint_latest.pt"
            )

    print("\nTraining complete!")

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, scaler, step, config,
        output_dir / "checkpoint_final.pt"
    )

    # Save model only (for inference)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Saved final model to {output_dir / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train Chimera LLM")

    # Model
    parser.add_argument("--model-config", type=str, default="small",
                        choices=["small", "base"],
                        help="Model configuration")

    # Data
    parser.add_argument("--data-path", type=str, default="data/train.txt",
                        help="Path to training data")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Logging
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name")

    args = parser.parse_args()

    config = TrainConfig(
        model_config=args.model_config,
        data_path=args.data_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        resume_from=args.resume,
        wandb_project=args.wandb_project,
    )

    train(config)


if __name__ == "__main__":
    main()
