"""Quick VRAM test for RTX 4050."""
import torch
import torch.nn.functional as F
from model import Chimera, chimera_medium
from tokenizer import ChimeraTokenizer

def test_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM: {total:.1f} GB")

    # Load model
    tokenizer = ChimeraTokenizer()
    config = chimera_medium()
    config.vocab_size = tokenizer.vocab_size

    model = Chimera(config)
    print(f"\nModel params: {model.get_num_params():,}")

    # Try loading checkpoint
    try:
        state = torch.load("checkpoints/model_1500.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print("Loaded checkpoint: model_1500.pt")
    except:
        print("No checkpoint found, using random weights")

    model = model.to(device)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"\nAfter model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Test forward + backward with gradient checkpointing
    print("\nTesting forward+backward pass...")

    # Simulate training batch
    batch_size = 1
    seq_len = 512

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Mixed precision
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (batch=1, seq=512): {peak:.2f} GB")

        if peak < 5.5:
            print("\n SUCCESS: Should fit on RTX 4050 (6GB)")
            print("  Recommended settings:")
            print("    --batch-size 1 --gradient-accumulation-steps 32")
        elif peak < 6.0:
            print("\n TIGHT: May work, but enable gradient checkpointing")
            print("  Recommended settings:")
            print("    --batch-size 1 --gradient-accumulation-steps 32")
        else:
            print("\n WARNING: May OOM. Try CPU offload:")
            print("    --cpu-offload --batch-size 1")

    # Clean up
    del model, input_ids, labels, logits, loss
    torch.cuda.empty_cache()
    print("\nMemory test complete.")

if __name__ == "__main__":
    test_memory()
