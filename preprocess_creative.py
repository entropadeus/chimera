"""
Preprocess and combine creative writing datasets for training.

Combines:
- Project Gutenberg (classic literature)
- Reddit WritingPrompts (creative fiction)
- Cornell Movie Dialogs (snappy dialogue)
- Poetry (lyrical language)

Creates a single shuffled training file optimized for the Chimera model.
"""

import os
import random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" / "creative"
OUTPUT_FILE = Path(__file__).parent / "data" / "creative_combined.txt"


def load_dataset(path: Path, separator: str = "<|endoftext|>") -> list:
    """Load a dataset split by separator."""
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by separator
    docs = content.split(separator)
    docs = [d.strip() for d in docs if d.strip() and len(d.strip()) > 100]

    return docs


def chunk_long_document(doc: str, max_chars: int = 8000) -> list:
    """
    Split very long documents into chunks for better mixing.
    Tries to split at paragraph boundaries.
    """
    if len(doc) <= max_chars:
        return [doc]

    chunks = []
    paragraphs = doc.split('\n\n')
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(para)
        current_len += para_len + 2  # +2 for \n\n

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def main():
    print("="*60)
    print("PREPROCESSING CREATIVE DATASETS")
    print("="*60)

    # Load datasets
    datasets = {}

    # Gutenberg - classic literature
    gutenberg_path = DATA_DIR / "gutenberg" / "gutenberg_combined.txt"
    gutenberg_docs = load_dataset(gutenberg_path)
    print(f"Gutenberg: {len(gutenberg_docs)} documents")
    datasets['gutenberg'] = gutenberg_docs

    # WritingPrompts - creative fiction
    wp_path = DATA_DIR / "writingprompts" / "writingprompts.txt"
    wp_docs = load_dataset(wp_path)
    print(f"WritingPrompts: {len(wp_docs)} stories")
    datasets['writingprompts'] = wp_docs

    # Movie dialogs - snappy dialogue
    movie_path = DATA_DIR / "movie_dialogs" / "movie_dialogs.txt"
    movie_docs = load_dataset(movie_path)
    print(f"Movie Dialogs: {len(movie_docs)} conversations")
    datasets['movie_dialogs'] = movie_docs

    # Poetry - lyrical language
    poetry_path = DATA_DIR / "poetry" / "poetry.txt"
    poetry_docs = load_dataset(poetry_path)
    print(f"Poetry: {len(poetry_docs)} poems")
    datasets['poetry'] = poetry_docs

    # Data mixing strategy
    # We want to oversample certain sources for style:
    # - Gutenberg: 2x (beautiful prose)
    # - WritingPrompts: 1x (already huge)
    # - Movie Dialogs: 3x (personality, "talks back")
    # - Poetry: 5x (lyrical density)

    print("\n" + "="*60)
    print("MIXING STRATEGY")
    print("="*60)

    all_docs = []

    # Chunk and weight datasets
    for name, docs in datasets.items():
        if name == 'gutenberg':
            weight = 2
        elif name == 'writingprompts':
            weight = 1
        elif name == 'movie_dialogs':
            weight = 3
        elif name == 'poetry':
            weight = 5
        else:
            weight = 1

        chunked = []
        for doc in docs:
            chunked.extend(chunk_long_document(doc))

        # Apply weight by repetition
        weighted = chunked * weight
        all_docs.extend(weighted)

        total_chars = sum(len(d) for d in weighted)
        print(f"{name}: {len(chunked)} chunks x{weight} = {len(weighted)} ({total_chars/1e6:.1f}M chars)")

    print(f"\nTotal documents: {len(all_docs)}")

    # Shuffle thoroughly
    print("\nShuffling...")
    random.seed(42)
    random.shuffle(all_docs)

    # Calculate stats
    total_chars = sum(len(d) for d in all_docs)
    est_tokens = total_chars // 4

    print(f"\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total documents: {len(all_docs):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {est_tokens:,}")
    print(f"Output file: {OUTPUT_FILE}")

    # Write output
    print("\nWriting combined file...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(tqdm(all_docs, desc="Writing")):
            f.write(doc)
            f.write('\n')  # One doc per line for the training script

    final_size = OUTPUT_FILE.stat().st_size
    print(f"\nDone! Output size: {final_size/1e6:.1f} MB")
    print(f"Path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
