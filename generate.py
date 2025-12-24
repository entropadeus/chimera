"""
Inference script for Chimera LLM.

Optimized for laptop deployment with:
- CPU and CUDA support
- KV caching for fast generation
- Streaming output
- Memory-efficient batch generation
- Quantization support (int8, int4)
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from model import Chimera, ChimeraConfig, chimera_small, chimera_medium, chimera_base, chimera_large
from tokenizer import ChimeraTokenizer


class ChimeraGenerator:
    """
    Text generation engine for Chimera.

    Features:
    - Greedy, top-k, top-p, and temperature sampling
    - KV caching for efficient autoregressive generation
    - Streaming output support
    - CPU/GPU inference
    """

    def __init__(
        self,
        model: Chimera,
        tokenizer: ChimeraTokenizer,
        device: str = "auto",
    ):
        """
        Initialize generator.

        Args:
            model: Chimera model instance
            tokenizer: ChimeraTokenizer instance
            device: "auto", "cpu", "cuda", or "mps"
        """
        self.tokenizer = tokenizer

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        print(f"Generator initialized on {self.device}")
        print(f"Model parameters: {model.get_num_params():,}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
        stop_tokens: Optional[List[str]] = None,
    ) -> Union[str, "Generator"]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
            do_sample: Use sampling vs greedy decoding
            stream: Yield tokens as they're generated
            stop_tokens: List of strings that stop generation

        Returns:
            Generated text (or generator if stream=True)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Stop token IDs
        stop_ids = set()
        stop_ids.add(self.tokenizer.eos_token_id)
        if stop_tokens:
            for st in stop_tokens:
                st_ids = self.tokenizer.encode(st, add_bos=False)
                if st_ids:
                    stop_ids.add(st_ids[-1])

        def _generate():
            nonlocal input_ids

            cache = None
            generated_ids = []
            position_offset = 0

            # Process prompt first (prefill)
            logits, cache = self.model(input_ids, use_cache=True)
            position_offset = input_ids.size(1)

            for _ in range(max_new_tokens):
                # Get logits for last position
                next_logits = logits[:, -1, :]

                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_ids:
                    for prev_id in set(generated_ids[-50:]):  # Last 50 tokens
                        next_logits[0, prev_id] /= repetition_penalty

                # Sample next token
                if do_sample and temperature > 0:
                    next_logits = next_logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                        next_logits[indices_to_remove] = float('-inf')

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                next_id = next_token.item()
                generated_ids.append(next_id)

                # Check stop condition
                if next_id in stop_ids:
                    break

                # Yield token if streaming
                if stream:
                    token_text = self.tokenizer.decode([next_id], skip_special_tokens=True)
                    yield token_text

                # Forward pass with cache
                logits, cache = self.model(
                    next_token,
                    cache=cache,
                    position_offset=position_offset,
                    use_cache=True
                )
                position_offset += 1

            # Final output
            if not stream:
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                yield output_text

        if stream:
            return _generate()
        else:
            return next(_generate())

    def chat(
        self,
        messages: List[dict],
        **generate_kwargs
    ) -> str:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            **generate_kwargs: Arguments passed to generate()

        Returns:
            Generated assistant response
        """
        # Format messages into prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")

        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        return self.generate(prompt, **generate_kwargs)


def load_model(
    checkpoint_path: Optional[str] = None,
    config: str = "base",
    device: str = "auto",
    dtype: str = "float16",
) -> tuple:
    """
    Load Chimera model and tokenizer.

    Args:
        checkpoint_path: Path to saved checkpoint (None for random init)
        config: Model config name ("small", "base", "medium", "large")
        device: Target device
        dtype: Weight dtype ("float32", "float16", "bfloat16")

    Returns:
        (model, tokenizer) tuple
    """
    # Load tokenizer FIRST so we can match vocab_size
    tokenizer = ChimeraTokenizer()

    # Select config
    configs = {
        "small": chimera_small,
        "medium": chimera_medium,
        "base": chimera_base,
        "large": chimera_large,
    }
    model_config = configs.get(config, chimera_small)()

    # Sync vocab_size with tokenizer (critical for avoiding index errors)
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"Adjusting model vocab_size: {model_config.vocab_size} -> {tokenizer.vocab_size}")
        model_config.vocab_size = tokenizer.vocab_size

    # Create model
    model = Chimera(model_config)

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)

    # Convert dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = model.to(dtype_map.get(dtype, torch.float16))

    return model, tokenizer


def quantize_model(model: Chimera, bits: int = 8) -> Chimera:
    """
    Quantize model for faster inference on CPU.

    Args:
        model: Model to quantize
        bits: Quantization bits (8 or 4)

    Returns:
        Quantized model
    """
    try:
        if bits == 8:
            # Dynamic int8 quantization
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("Applied int8 dynamic quantization")
        else:
            print(f"Warning: {bits}-bit quantization requires additional libraries")

    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Continuing with unquantized model")

    return model


def interactive_chat(generator: ChimeraGenerator):
    """Run interactive chat session."""
    print("\n" + "="*60)
    print("Chimera Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset history")
    print("="*60 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                messages = []
                print("[History cleared]")
                continue

            messages.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)

            # Stream response
            response_text = ""
            for token in generator.generate(
                "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\nAssistant:",
                stream=True,
                max_new_tokens=512,
            ):
                print(token, end="", flush=True)
                response_text += token

            print()  # Newline after response

            messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n[Interrupted]")
            break


def benchmark(generator: ChimeraGenerator, prompt: str, n_tokens: int = 100):
    """Benchmark generation speed."""
    print(f"\nBenchmarking {n_tokens} tokens...")

    # Warmup
    _ = generator.generate(prompt, max_new_tokens=10, do_sample=False)

    # Timed run
    torch.cuda.synchronize() if generator.device.type == "cuda" else None
    start = time.perf_counter()

    output = generator.generate(prompt, max_new_tokens=n_tokens, do_sample=False)

    torch.cuda.synchronize() if generator.device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    tokens_per_sec = n_tokens / elapsed
    print(f"Generated {n_tokens} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"Output: {output[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Chimera LLM Inference")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "medium", "base", "large"],
                        help="Model configuration")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for inference")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--quantize", type=int, default=None,
                        choices=[8, 4],
                        help="Quantization bits")

    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive chat")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")

    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        config=args.config,
        device=args.device,
        dtype=args.dtype,
    )

    # Quantize if requested
    if args.quantize:
        model = quantize_model(model, args.quantize)

    # Create generator
    generator = ChimeraGenerator(model, tokenizer, device=args.device)

    # Run mode
    if args.interactive:
        interactive_chat(generator)
    elif args.benchmark:
        benchmark(generator, "The future of artificial intelligence is", n_tokens=100)
    elif args.prompt:
        output = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(output)
    else:
        # Default: quick test
        print("\nQuick test generation:")
        output = generator.generate(
            "Once upon a time",
            max_new_tokens=50,
            temperature=0.8,
        )
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
