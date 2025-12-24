"""
Conversational fine-tuning for Chimera → Wyrd.

Key features:
- ChatML-style template with system prompts
- Loss masking (only trains on assistant responses)
- Multi-turn conversation support
- Lower learning rate than pretraining
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import Chimera, chimera_small, chimera_medium, chimera_base
from tokenizer import ChimeraTokenizer


@dataclass
class InstructConfig:
    """Conversational fine-tuning configuration."""
    # Model
    model_path: str = "checkpoints/model.pt"
    model_config: str = "small"

    # Data
    data_path: str = "data/instruct_data.jsonl"
    seq_length: int = 512

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 2
    grad_clip: float = 1.0

    # Efficiency
    mixed_precision: bool = True
    compile_model: bool = False

    # Output
    output_dir: str = "checkpoints"
    save_every: int = 200
    log_every: int = 10
    eval_every: int = 500


# =============================================================================
# ChatML-Style Template
# =============================================================================

class ChatTemplate:
    """
    ChatML-style template for conversation formatting.

    Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    DEFAULT_SYSTEM = "You are Wyrd, a helpful AI assistant."

    @classmethod
    def format_system(cls, content: str) -> str:
        return f"{cls.IM_START}system\n{content}{cls.IM_END}\n"

    @classmethod
    def format_user(cls, content: str) -> str:
        return f"{cls.IM_START}user\n{content}{cls.IM_END}\n"

    @classmethod
    def format_assistant(cls, content: str, add_generation_prompt: bool = False) -> str:
        if add_generation_prompt:
            return f"{cls.IM_START}assistant\n"
        return f"{cls.IM_START}assistant\n{content}{cls.IM_END}\n"

    @classmethod
    def format_conversation(
        cls,
        conversations: List[Dict],
        system: Optional[str] = None,
        add_generation_prompt: bool = False
    ) -> str:
        """Format a full conversation."""
        parts = []

        # System prompt
        sys_content = system or cls.DEFAULT_SYSTEM
        parts.append(cls.format_system(sys_content))

        # Conversation turns
        for i, turn in enumerate(conversations):
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                parts.append(cls.format_user(content))
            elif role == "assistant":
                # If last turn and add_generation_prompt, don't include content
                is_last = (i == len(conversations) - 1)
                if is_last and add_generation_prompt:
                    parts.append(cls.format_assistant("", add_generation_prompt=True))
                else:
                    parts.append(cls.format_assistant(content))

        return "".join(parts)


# =============================================================================
# Dataset
# =============================================================================

class ConversationDataset(Dataset):
    """Dataset for conversational fine-tuning with loss masking."""

    def __init__(
        self,
        data_path: str,
        tokenizer: ChimeraTokenizer,
        seq_length: int = 512,
        template: ChatTemplate = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.template = template or ChatTemplate()
        self.examples = []

        print(f"Loading conversation data from {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if example.get("conversations"):
                        self.examples.append(example)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.examples)} conversations")

        # Analyze sequence lengths (sample first 100)
        if len(self.examples) > 0:
            sample_size = min(100, len(self.examples))
            sample_lengths = []
            for ex in self.examples[:sample_size]:
                text = ChatTemplate.format_conversation(ex["conversations"], ex.get("system"))
                tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
                sample_lengths.append(len(tokens))
            avg_len = sum(sample_lengths) / len(sample_lengths)
            max_len = max(sample_lengths)
            over_limit = sum(1 for l in sample_lengths if l > seq_length)
            print(f"Sequence stats (sample of {sample_size}): avg={avg_len:.0f}, max={max_len}, truncated={over_limit}/{sample_size}")
            if over_limit > sample_size * 0.5:
                print(f"  WARNING: >50% of sequences exceed seq_length={seq_length}. Consider increasing --seq-length.")

    def _tokenize_with_mask(
        self,
        conversations: List[Dict],
        system: Optional[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenize conversation and create aligned loss mask.

        CRITICAL: We build tokens and mask incrementally to ensure perfect alignment.
        BPE tokenization is NOT additive, so we must encode the growing string
        and track position deltas.

        Returns:
            tokens: List of token IDs
            mask: List of 0/1 (1 = compute loss on this position)
        """
        tokens = []
        mask = []

        # System prompt - no loss
        sys_content = system or ChatTemplate.DEFAULT_SYSTEM
        sys_text = ChatTemplate.format_system(sys_content)
        sys_tokens = self.tokenizer.encode(sys_text, add_bos=False, add_eos=False)
        tokens.extend(sys_tokens)
        mask.extend([0] * len(sys_tokens))

        # Build conversation turn by turn
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                turn_text = ChatTemplate.format_user(content)
                turn_tokens = self.tokenizer.encode(turn_text, add_bos=False, add_eos=False)
                tokens.extend(turn_tokens)
                mask.extend([0] * len(turn_tokens))
            else:  # assistant
                # Encode prefix (no loss)
                prefix = f"{ChatTemplate.IM_START}assistant\n"
                prefix_tokens = self.tokenizer.encode(prefix, add_bos=False, add_eos=False)
                tokens.extend(prefix_tokens)
                mask.extend([0] * len(prefix_tokens))

                # Encode content (with loss)
                content_tokens = self.tokenizer.encode(content, add_bos=False, add_eos=False)
                tokens.extend(content_tokens)
                mask.extend([1] * len(content_tokens))

                # Encode suffix (with loss - learn to end responses)
                suffix = f"{ChatTemplate.IM_END}\n"
                suffix_tokens = self.tokenizer.encode(suffix, add_bos=False, add_eos=False)
                tokens.extend(suffix_tokens)
                mask.extend([1] * len(suffix_tokens))

        # Now verify alignment by re-encoding the full text
        full_text = ChatTemplate.format_conversation(conversations, system=system)
        full_tokens = self.tokenizer.encode(full_text, add_bos=False, add_eos=False)

        # If lengths don't match, fall back to safe mode: train on everything after system
        if len(tokens) != len(full_tokens):
            # Mismatch due to BPE boundary effects - use full tokens with approximate mask
            tokens = full_tokens
            # Conservative: mark assistant content regions based on string positions
            mask = self._create_approximate_mask(full_text, conversations, system, len(full_tokens))

        return tokens, mask

    def _create_approximate_mask(
        self,
        full_text: str,
        conversations: List[Dict],
        system: Optional[str],
        num_tokens: int
    ) -> List[int]:
        """
        Fallback: Create approximate mask based on character positions.
        Maps character positions to token positions for robustness.
        """
        # Find character ranges of assistant content
        assistant_ranges = []  # List of (start_char, end_char)

        # Track position in text
        sys_content = system or ChatTemplate.DEFAULT_SYSTEM
        pos = len(ChatTemplate.format_system(sys_content))

        for turn in conversations:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                pos += len(ChatTemplate.format_user(content))
            else:  # assistant
                prefix = f"{ChatTemplate.IM_START}assistant\n"
                pos += len(prefix)
                start_char = pos
                pos += len(content)
                end_char = pos + len(f"{ChatTemplate.IM_END}\n")
                pos = end_char
                assistant_ranges.append((start_char, end_char))

        # Map characters to tokens (approximate)
        # Use a simple heuristic: chars_per_token ≈ len(text) / num_tokens
        text_len = len(full_text)
        chars_per_token = text_len / max(num_tokens, 1)

        mask = [0] * num_tokens
        for start_char, end_char in assistant_ranges:
            start_tok = int(start_char / chars_per_token)
            end_tok = int(end_char / chars_per_token)
            for i in range(max(0, start_tok), min(num_tokens, end_tok)):
                mask[i] = 1

        return mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        conversations = example["conversations"]
        system = example.get("system")

        # Tokenize with aligned mask
        tokens, label_mask = self._tokenize_with_mask(conversations, system)

        # Add BOS token
        tokens = [self.tokenizer.bos_token_id] + tokens
        label_mask = [0] + label_mask  # BOS = no loss

        # Add EOS token (part of last response, should have loss)
        tokens = tokens + [self.tokenizer.eos_token_id]
        label_mask = label_mask + [1]

        # Truncate or pad to seq_length + 1
        target_len = self.seq_length + 1

        if len(tokens) > target_len:
            tokens = tokens[:target_len]
            label_mask = label_mask[:target_len]
        elif len(tokens) < target_len:
            pad_len = target_len - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_len
            label_mask = label_mask + [0] * pad_len

        tokens = torch.tensor(tokens, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.long)

        # Create input/labels
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()

        # Apply mask: -100 means ignore in cross entropy
        labels[label_mask[1:] == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# =============================================================================
# Training
# =============================================================================

def evaluate(model, dataloader, device, max_batches: int = 50):
    """Quick evaluation on a subset of data."""
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
                ignore_index=-100,
                reduction='sum'
            )

            # Count non-ignored tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def train(config: InstructConfig):
    """Main conversational fine-tuning loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = ChimeraTokenizer()

    # Model
    model_configs = {"small": chimera_small, "medium": chimera_medium, "base": chimera_base}
    model_cfg = model_configs.get(config.model_config, chimera_small)()
    model_cfg.vocab_size = tokenizer.vocab_size

    model = Chimera(model_cfg)

    # Load pretrained weights
    print(f"Loading pretrained weights from {config.model_path}...")
    state_dict = torch.load(config.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Optional compile (skip on TPU/XLA - it uses its own compilation)
    is_xla = False
    try:
        import torch_xla
        is_xla = True
    except ImportError:
        pass

    if config.compile_model and hasattr(torch, 'compile') and not is_xla:
        print("Compiling model with torch.compile()...")
        # Use mode="reduce-overhead" for stability, fullgraph=False for flexibility
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        # Warmup compilation with dummy batch before training
        print("Warming up compiled model...")
        with torch.no_grad():
            dummy_input = torch.randint(0, model_cfg.vocab_size, (1, config.seq_length), device=device)
            _ = model(dummy_input)
        print("Compilation warmup complete.")
    elif is_xla:
        print("TPU/XLA detected - skipping torch.compile() (XLA handles compilation)")

    # Dataset
    dataset = ConversationDataset(config.data_path, tokenizer, config.seq_length)

    # Split for eval (5%)
    eval_size = max(100, len(dataset) // 20)
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

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Calculate training steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    print(f"\nTraining plan:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(config.min_lr / config.learning_rate, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = config.mixed_precision and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    autocast_dtype = torch.float16 if use_amp else torch.float32

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting conversational fine-tuning (Chimera → Wyrd)")
    print(f"{'='*60}\n")

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    best_eval_loss = float('inf')
    start_time = time.time()
    tokens_processed = 0

    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===\n")

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / config.gradient_accumulation_steps

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * config.gradient_accumulation_steps
            tokens_processed += (labels != -100).sum().item()
            micro_step += 1

            # Optimizer step
            if micro_step % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % config.log_every == 0:
                    avg_loss = running_loss / config.log_every
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    toks_per_sec = tokens_processed / elapsed

                    print(f"Step {global_step:5d}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss, 20)):.2f} | "
                          f"LR: {lr:.2e} | {toks_per_sec:.0f} tok/s")

                    running_loss = 0.0
                    tokens_processed = 0
                    start_time = time.time()

                # Evaluation
                if global_step % config.eval_every == 0:
                    eval_loss, eval_ppl = evaluate(model, eval_loader, device)
                    print(f"  [Eval] Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_path = output_dir / "instruct_best.pt"
                        torch.save(model.state_dict(), best_path)
                        print(f"  [Eval] New best! Saved to {best_path}")

                # Checkpoint
                if global_step % config.save_every == 0:
                    save_path = output_dir / f"instruct_step_{global_step}.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved: {save_path}")

    # Final evaluation
    print("\n" + "="*60)
    final_loss, final_ppl = evaluate(model, eval_loader, device)
    print(f"Final Eval - Loss: {final_loss:.4f} | PPL: {final_ppl:.2f}")

    # Final save
    final_path = output_dir / "instruct_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {output_dir / 'instruct_best.pt'}")
    print(f"\nTo chat with Wyrd:")
    print(f"  python chat_ui.py --checkpoint {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Conversational Fine-tuning for Chimera")

    parser.add_argument("--model-path", type=str, default="checkpoints/model.pt",
                        help="Path to pretrained model weights")
    parser.add_argument("--model-config", type=str, default="small",
                        choices=["small", "medium", "base"])
    parser.add_argument("--data-path", type=str, default="data/instruct_data.jsonl",
                        help="Path to conversation data (JSONL)")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()

    config = InstructConfig(
        model_path=args.model_path,
        model_config=args.model_config,
        data_path=args.data_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        compile_model=args.compile,
        mixed_precision=not args.no_amp,
    )

    train(config)


if __name__ == "__main__":
    main()
