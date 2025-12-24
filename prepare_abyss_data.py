"""
Comprehensive Dataset Preparation for Chimera Abyss

Downloads and processes multiple high-quality datasets:
- OpenAssistant (OASST2) - Human conversations
- Dolly 15k - Instruction following
- Project Gutenberg - Classic literature
- WritingPrompts - Creative writing from Reddit
- TinyStories - Narrative coherence
- UltraChat - Diverse conversations

Creates a unified training file with proper formatting and quality filtering.
"""

import os
import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
from collections import defaultdict
import hashlib

# Check for required packages
try:
    from datasets import load_dataset
    from tqdm import tqdm
    import requests
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'datasets', 'tqdm', 'requests'])
    from datasets import load_dataset
    from tqdm import tqdm
    import requests


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    name: str
    hf_path: Optional[str] = None
    hf_split: str = "train"
    hf_subset: Optional[str] = None
    url: Optional[str] = None
    text_field: str = "text"
    max_samples: Optional[int] = None
    min_length: int = 100
    max_length: int = 8000
    category: str = "general"
    weight: float = 1.0


# Dataset configurations
DATASETS = {
    # === CONVERSATION / CHAT ===
    "oasst2": DatasetConfig(
        name="OpenAssistant OASST2",
        hf_path="OpenAssistant/oasst2",
        text_field="text",
        max_samples=50000,
        min_length=50,
        category="chat",
        weight=1.5,
    ),
    "dolly": DatasetConfig(
        name="Databricks Dolly 15k",
        hf_path="databricks/databricks-dolly-15k",
        text_field="instruction,response",  # Will combine these
        max_samples=15000,
        min_length=50,
        category="chat",
        weight=1.2,
    ),
    "ultrachat": DatasetConfig(
        name="UltraChat 200k",
        hf_path="stingning/ultrachat",
        text_field="data",
        max_samples=100000,
        min_length=100,
        category="chat",
        weight=1.3,
    ),

    # === CREATIVE WRITING ===
    "writing_prompts": DatasetConfig(
        name="WritingPrompts",
        hf_path="euclaise/writingprompts",
        text_field="story",
        max_samples=50000,
        min_length=200,
        max_length=10000,
        category="creative",
        weight=1.5,
    ),
    "tinystories": DatasetConfig(
        name="TinyStories",
        hf_path="roneneldan/TinyStories",
        hf_subset="default",
        text_field="text",
        max_samples=100000,
        min_length=100,
        category="creative",
        weight=1.0,
    ),

    # === ELOQUENT PROSE / LITERATURE ===
    "gutenberg": DatasetConfig(
        name="Project Gutenberg",
        hf_path="sedthh/gutenberg_english",
        text_field="TEXT",
        max_samples=5000,  # Fewer but longer
        min_length=500,
        max_length=15000,
        category="eloquent",
        weight=2.0,  # High weight for style
    ),
    "booksum": DatasetConfig(
        name="BookSum",
        hf_path="kmfoda/booksum",
        text_field="chapter",
        max_samples=10000,
        min_length=300,
        max_length=12000,
        category="eloquent",
        weight=1.5,
    ),

    # === GENERAL KNOWLEDGE ===
    "wikipedia": DatasetConfig(
        name="Wikipedia (Simple)",
        hf_path="wikipedia",
        hf_subset="20220301.simple",
        text_field="text",
        max_samples=50000,
        min_length=200,
        max_length=5000,
        category="general",
        weight=0.8,
    ),
    "c4_realnews": DatasetConfig(
        name="C4 RealNews",
        hf_path="allenai/c4",
        hf_subset="realnewslike",
        text_field="text",
        max_samples=50000,
        min_length=200,
        max_length=5000,
        category="general",
        weight=0.7,
    ),
}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove excessive punctuation
    text = re.sub(r'([.!?])\1{3,}', r'\1\1\1', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove Reddit formatting artifacts
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    text = re.sub(r'^\s*\[WP\]|\[OT\]|\[EU\]|\[RF\]|\[SP\]', '', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text.strip()


def is_quality_text(text: str, min_length: int, max_length: int) -> bool:
    """Filter for quality text."""
    if not text:
        return False

    length = len(text)
    if length < min_length or length > max_length:
        return False

    # Check for minimum word count
    words = text.split()
    if len(words) < 20:
        return False

    # Check for excessive repetition
    if len(set(words)) / len(words) < 0.3:
        return False

    # Check for too many special characters
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.6:
        return False

    # Check for gibberish (very long words)
    max_word_len = max(len(w) for w in words)
    if max_word_len > 50:
        return False

    return True


def format_conversation(messages: List[Dict]) -> str:
    """Format a conversation into training text."""
    formatted = []
    for msg in messages:
        role = msg.get('role', msg.get('from', 'user'))
        content = msg.get('content', msg.get('value', ''))

        if role in ['user', 'human']:
            formatted.append(f"Human: {content}")
        elif role in ['assistant', 'gpt', 'bot']:
            formatted.append(f"Assistant: {content}")
        else:
            formatted.append(content)

    return "\n\n".join(formatted)


def process_oasst2(dataset) -> Generator[str, None, None]:
    """Process OpenAssistant dataset."""
    # Build conversation trees
    messages_by_parent = defaultdict(list)
    messages_by_id = {}

    for item in dataset:
        msg_id = item.get('message_id')
        parent_id = item.get('parent_id')
        messages_by_id[msg_id] = item
        if parent_id:
            messages_by_parent[parent_id].append(item)

    # Find root messages and build conversations
    root_messages = [m for m in dataset if not m.get('parent_id')]

    def build_conversation(msg_id: str, depth: int = 0) -> List[Dict]:
        if depth > 10 or msg_id not in messages_by_id:
            return []

        msg = messages_by_id[msg_id]
        conv = [{'role': msg.get('role', 'user'), 'content': msg.get('text', '')}]

        # Get best reply (highest rank or first)
        replies = messages_by_parent.get(msg_id, [])
        if replies:
            replies.sort(key=lambda x: x.get('rank', 999))
            conv.extend(build_conversation(replies[0]['message_id'], depth + 1))

        return conv

    for root in root_messages:
        conv = build_conversation(root['message_id'])
        if len(conv) >= 2:
            text = format_conversation(conv)
            if text:
                yield text


def process_dolly(dataset) -> Generator[str, None, None]:
    """Process Dolly dataset."""
    for item in dataset:
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')

        if context:
            text = f"Human: {instruction}\n\nContext: {context}\n\nAssistant: {response}"
        else:
            text = f"Human: {instruction}\n\nAssistant: {response}"

        yield text


def process_ultrachat(dataset) -> Generator[str, None, None]:
    """Process UltraChat dataset."""
    for item in dataset:
        data = item.get('data', [])
        if isinstance(data, list) and len(data) >= 2:
            messages = []
            for i, msg in enumerate(data):
                role = 'user' if i % 2 == 0 else 'assistant'
                messages.append({'role': role, 'content': msg})
            text = format_conversation(messages)
            if text:
                yield text


def process_writing_prompts(dataset) -> Generator[str, None, None]:
    """Process WritingPrompts dataset."""
    for item in dataset:
        prompt = item.get('prompt', '')
        story = item.get('story', '')

        if prompt and story:
            # Clean the prompt
            prompt = re.sub(r'^\s*\[WP\]|\[OT\]|\[EU\]|\[RF\]|\[SP\]', '', prompt).strip()
            text = f"Prompt: {prompt}\n\n{story}"
            yield text
        elif story:
            yield story


def process_generic(dataset, text_field: str) -> Generator[str, None, None]:
    """Generic processor for simple text datasets."""
    for item in dataset:
        if ',' in text_field:
            # Multiple fields to combine
            fields = [f.strip() for f in text_field.split(',')]
            parts = [str(item.get(f, '')) for f in fields if item.get(f)]
            text = '\n\n'.join(parts)
        else:
            text = item.get(text_field, '')

        if text:
            yield text


def download_and_process_dataset(config: DatasetConfig, output_dir: Path) -> List[str]:
    """Download and process a single dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {config.name}")
    print(f"{'='*60}")

    texts = []

    try:
        # Load from HuggingFace
        if config.hf_subset:
            dataset = load_dataset(config.hf_path, config.hf_subset, split=config.hf_split, streaming=True)
        else:
            dataset = load_dataset(config.hf_path, split=config.hf_split, streaming=True)

        # Select processor
        if 'oasst' in config.hf_path.lower():
            # OASST needs special handling - load non-streaming for tree building
            full_dataset = load_dataset(config.hf_path, split=config.hf_split)
            processor = process_oasst2(full_dataset)
        elif 'dolly' in config.hf_path.lower():
            processor = process_dolly(dataset)
        elif 'ultrachat' in config.hf_path.lower():
            processor = process_ultrachat(dataset)
        elif 'writingprompts' in config.hf_path.lower():
            processor = process_writing_prompts(dataset)
        else:
            processor = process_generic(dataset, config.text_field)

        # Process with progress bar
        count = 0
        pbar = tqdm(processor, desc=f"Processing {config.name}", total=config.max_samples)

        for text in pbar:
            if config.max_samples and count >= config.max_samples:
                break

            text = clean_text(text)

            if is_quality_text(text, config.min_length, config.max_length):
                texts.append(text)
                count += 1
                pbar.set_postfix({'kept': count})

        pbar.close()
        print(f"Collected {len(texts):,} samples from {config.name}")

    except Exception as e:
        print(f"Error processing {config.name}: {e}")
        import traceback
        traceback.print_exc()

    return texts


def create_combined_dataset(
    output_path: str,
    datasets_to_use: List[str] = None,
    shuffle: bool = True,
    seed: int = 42,
):
    """Download, process, and combine all datasets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if datasets_to_use is None:
        datasets_to_use = list(DATASETS.keys())

    all_texts = []
    category_counts = defaultdict(int)

    for name in datasets_to_use:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        config = DATASETS[name]
        texts = download_and_process_dataset(config, output_path.parent)

        # Apply weight by duplicating
        if config.weight > 1.0:
            weight_factor = int(config.weight)
            texts = texts * weight_factor

        all_texts.extend(texts)
        category_counts[config.category] += len(texts)

    # Shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(all_texts)

    # Deduplicate by hash
    print("\nDeduplicating...")
    seen_hashes = set()
    unique_texts = []
    for text in tqdm(all_texts, desc="Deduplicating"):
        text_hash = hashlib.md5(text[:500].encode()).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)

    print(f"Removed {len(all_texts) - len(unique_texts):,} duplicates")
    all_texts = unique_texts

    # Write to file
    print(f"\nWriting {len(all_texts):,} samples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(all_texts, desc="Writing"):
            # Each sample on its own line(s), separated by double newline
            f.write(text.strip() + '\n\n')

    # Stats
    file_size = output_path.stat().st_size / 1e9
    print(f"\n{'='*60}")
    print("DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_texts):,}")
    print(f"File size: {file_size:.2f} GB")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count:,}")
    print(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Chimera Abyss training data")
    parser.add_argument(
        "--output", "-o",
        default="data/abyss_train.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=None,
        help="Specific datasets to use (default: all)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print("="*60)
        for name, config in DATASETS.items():
            print(f"\n{name}:")
            print(f"  Name: {config.name}")
            print(f"  Category: {config.category}")
            print(f"  Max samples: {config.max_samples:,}")
            print(f"  Weight: {config.weight}")
        return

    create_combined_dataset(
        output_path=args.output,
        datasets_to_use=args.datasets,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
