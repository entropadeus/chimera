"""
VibeVoice TTS integration for Wyrd.
Uses Microsoft's VibeVoice-Realtime-0.5B model.
"""

import os
import io
import torch
import tempfile

# Lazy imports to avoid loading heavy deps if TTS is disabled
TTS_MODEL = None
TTS_PROCESSOR = None
TTS_LOADED = False

MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
SAMPLE_RATE = 24000


def load_tts(device="cuda"):
    """Load the VibeVoice TTS model."""
    global TTS_MODEL, TTS_PROCESSOR, TTS_LOADED

    if TTS_LOADED:
        return True

    print("Loading VibeVoice TTS model...")

    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        # Load processor
        TTS_PROCESSOR = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

        # Load model
        TTS_MODEL = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="sdpa",
        )
        TTS_MODEL.set_ddpm_inference_steps(5)  # Fast inference

        TTS_LOADED = True
        print(f"VibeVoice TTS loaded on {device}")
        return True

    except Exception as e:
        print(f"Failed to load VibeVoice TTS: {e}")
        return False


def generate_speech(text: str, voice: str = "en-US-AvaNeural") -> bytes:
    """
    Generate speech from text using VibeVoice.

    Args:
        text: The text to synthesize
        voice: Voice preset to use

    Returns:
        WAV audio bytes
    """
    global TTS_MODEL, TTS_PROCESSOR

    if not TTS_LOADED:
        raise RuntimeError("TTS model not loaded. Call load_tts() first.")

    # Process input
    inputs = TTS_PROCESSOR(
        text=text,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(TTS_MODEL.device) for k, v in inputs.items()}

    # Generate
    with torch.cuda.amp.autocast():
        outputs = TTS_MODEL.generate(
            **inputs,
            cfg_scale=1.5,
        )

    # Get audio
    audio = outputs.speech_outputs[0].cpu().numpy()

    # Convert to WAV bytes
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='WAV')
    buffer.seek(0)

    return buffer.read()


def is_tts_available() -> bool:
    """Check if TTS is loaded and available."""
    return TTS_LOADED
