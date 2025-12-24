"""Test different sequence lengths and batch sizes for VRAM."""
import torch
import torch.nn.functional as F
from model import Chimera, chimera_medium
from tokenizer import ChimeraTokenizer

def test_config(model, config, batch_size, seq_len, device):
    """Test a specific configuration."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    try:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1e9
        model.zero_grad(set_to_none=True)
        del input_ids, labels, logits, loss
        torch.cuda.empty_cache()
        return peak, True
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return 0, False
        raise

def main():
    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {total_vram:.1f} GB")
    print(f"Safe limit: ~{total_vram * 0.9:.1f} GB\n")

    tokenizer = ChimeraTokenizer()
    config = chimera_medium()
    config.vocab_size = tokenizer.vocab_size

    model = Chimera(config)
    model = model.to(device)
    model.train()

    print(f"Model: {model.get_num_params()/1e6:.0f}M params")
    print(f"Base VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    # Test configurations
    configs = [
        # (batch_size, seq_len)
        (1, 512),
        (1, 768),
        (1, 1024),
        (1, 1536),
        (1, 2048),
        (2, 512),
        (2, 768),
        (2, 1024),
        (4, 512),
        (4, 768),
    ]

    print(f"{'Batch':>6} {'SeqLen':>8} {'Tokens':>10} {'Peak VRAM':>12} {'Status':>10}")
    print("-" * 52)

    results = []
    for batch_size, seq_len in configs:
        tokens = batch_size * seq_len
        peak, success = test_config(model, config, batch_size, seq_len, device)

        if success:
            status = "OK" if peak < total_vram * 0.85 else "TIGHT"
            print(f"{batch_size:>6} {seq_len:>8} {tokens:>10,} {peak:>10.2f} GB {status:>10}")
            results.append((batch_size, seq_len, tokens, peak))
        else:
            print(f"{batch_size:>6} {seq_len:>8} {tokens:>10,} {'---':>12} {'OOM':>10}")
            break  # No point testing larger

    # Recommend best config
    print("\n" + "="*52)
    safe_configs = [(b, s, t, p) for b, s, t, p in results if p < total_vram * 0.85]
    if safe_configs:
        best = max(safe_configs, key=lambda x: x[2])  # Most tokens
        print(f"RECOMMENDED: --batch-size {best[0]} --seq-length {best[1]}")
        print(f"  Tokens/step: {best[2]:,}")
        print(f"  Peak VRAM: {best[3]:.2f} GB")
        print(f"  With grad_accum=16: {best[2]*16:,} tokens/update")

if __name__ == "__main__":
    main()
