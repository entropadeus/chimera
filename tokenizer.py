"""
Tokenizer wrapper for Chimera.

Uses SentencePiece-style tokenization compatible with common models.
Supports loading pre-trained tokenizers from HuggingFace or training custom ones.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import torch


class ChimeraTokenizer:
    """
    Tokenizer for Chimera model.

    Wraps HuggingFace tokenizers for easy integration.
    Falls back to a simple character-level tokenizer if no tokenizer is available.
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ):
        """
        Initialize tokenizer.

        Args:
            tokenizer_path: Path to pre-trained tokenizer (HF format) or None for fallback
            vocab_size: Vocabulary size (used if training new tokenizer)
            pad_token: Padding token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self._tokenizer = None
        self._is_hf = False

        if tokenizer_path:
            self._load_tokenizer(tokenizer_path)
        else:
            # Try to load a good default tokenizer
            self._try_load_default()

    def _load_tokenizer(self, path: str):
        """Load tokenizer from path."""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._is_hf = True
            self.vocab_size = self._tokenizer.vocab_size
            print(f"Loaded HuggingFace tokenizer from {path}")
        except ImportError:
            print("Warning: transformers not installed, using fallback tokenizer")
            self._setup_fallback()
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {path}: {e}")
            self._setup_fallback()

    def _try_load_default(self):
        """Try to load a sensible default tokenizer."""
        # Try tokenizers with vocab_size=32000 first (matches our default model config)
        # Then fall back to others
        default_tokenizers = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 32000 vocab, matches our default
            "meta-llama/Llama-2-7b-hf",  # 32000 vocab (may need auth)
            "mistralai/Mistral-7B-v0.1",  # 32000 vocab (may need auth)
            "gpt2",  # 50257 vocab - will need model vocab adjustment
        ]

        try:
            from transformers import AutoTokenizer

            for name in default_tokenizers:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                    self._is_hf = True
                    self.vocab_size = self._tokenizer.vocab_size
                    print(f"Loaded default tokenizer: {name} (vocab_size={self.vocab_size})")
                    return
                except Exception as e:
                    continue

        except ImportError:
            pass

        print("Using fallback character-level tokenizer")
        self._setup_fallback()

    def _setup_fallback(self):
        """Setup simple character-level fallback tokenizer."""
        # Basic ASCII + common unicode
        chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                    "abcdefghijklmnopqrstuvwxyz{|}~\n\t")

        self._char_to_id = {self.pad_token: 0, self.bos_token: 1,
                           self.eos_token: 2, self.unk_token: 3}
        for i, c in enumerate(chars, start=4):
            self._char_to_id[c] = i

        self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        self.vocab_size = len(self._char_to_id)
        self._is_hf = False

    @property
    def pad_token_id(self) -> int:
        if self._is_hf:
            return self._tokenizer.pad_token_id or 0
        return self._char_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        if self._is_hf:
            return self._tokenizer.bos_token_id or 1
        return self._char_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        if self._is_hf:
            return self._tokenizer.eos_token_id or 2
        return self._char_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        if self._is_hf:
            return self._tokenizer.unk_token_id or 3
        return self._char_to_id[self.unk_token]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            List of token IDs
        """
        if self._is_hf:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback character-level encoding
            ids = [
                self._char_to_id.get(c, self.unk_token_id)
                for c in text
            ]

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        if self._is_hf:
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            # Fallback character-level decoding
            special = {self.pad_token_id, self.bos_token_id,
                      self.eos_token_id, self.unk_token_id}
            chars = []
            for id in ids:
                if skip_special_tokens and id in special:
                    continue
                chars.append(self._id_to_char.get(id, self.unk_token))
            return "".join(chars)

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """
        Tokenize text(s) with optional padding and truncation.

        Args:
            text: Input text or list of texts
            padding: Whether to pad to max length
            truncation: Whether to truncate to max length
            max_length: Maximum sequence length
            return_tensors: "pt" for PyTorch tensors

        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            text = [text]

        # Encode all texts
        all_ids = [self.encode(t) for t in text]

        # Truncate if needed
        if truncation and max_length:
            all_ids = [ids[:max_length] for ids in all_ids]

        # Calculate max length for padding
        if padding:
            if max_length:
                pad_length = max_length
            else:
                pad_length = max(len(ids) for ids in all_ids)
        else:
            pad_length = None

        # Create attention masks and pad
        attention_masks = []
        for i, ids in enumerate(all_ids):
            mask = [1] * len(ids)
            if pad_length and len(ids) < pad_length:
                padding_size = pad_length - len(ids)
                ids.extend([self.pad_token_id] * padding_size)
                mask.extend([0] * padding_size)
            all_ids[i] = ids
            attention_masks.append(mask)

        result = {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])

        return result

    def save_pretrained(self, path: str):
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._is_hf:
            self._tokenizer.save_pretrained(path)
        else:
            # Save fallback tokenizer config
            config = {
                "vocab_size": self.vocab_size,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "char_to_id": self._char_to_id,
            }
            with open(path / "tokenizer_config.json", "w") as f:
                json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "ChimeraTokenizer":
        """Load tokenizer from directory."""
        return cls(tokenizer_path=path)


if __name__ == "__main__":
    # Test tokenizer
    print("Testing ChimeraTokenizer...")

    tokenizer = ChimeraTokenizer()

    test_text = "Hello, world! This is a test of the Chimera tokenizer."

    # Test encode/decode
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {test_text}")
    print(f"Token IDs: {ids[:20]}..." if len(ids) > 20 else f"Token IDs: {ids}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Test batch tokenization
    batch = tokenizer(
        ["Hello", "World"],
        padding=True,
        return_tensors="pt"
    )
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
