"""
Data Download Script for Creative Writer Model
===============================================
Downloads and prepares diverse training data:
- Project Gutenberg (classic literature, elegant prose)
- Reddit WritingPrompts (creative fiction, varied styles)
- Cornell Movie Dialogs (snappy dialogue, personality)
- TinyStories (baseline coherence, already have this)

Target: Verbose, beautiful prose with personality that "talks back"
"""

import os
import json
import requests
import zipfile
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data directory
DATA_DIR = Path(__file__).parent / "data" / "creative"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


# =============================================================================
# Project Gutenberg - Classic Literature
# =============================================================================

# Top 100 most downloaded + literary classics
GUTENBERG_BOOKS = [
    # Classics with beautiful prose
    1342,   # Pride and Prejudice - Jane Austen
    11,     # Alice's Adventures in Wonderland
    1661,   # Sherlock Holmes
    84,     # Frankenstein
    98,     # A Tale of Two Cities
    2701,   # Moby Dick
    1952,   # The Yellow Wallpaper
    174,    # The Picture of Dorian Gray - Oscar Wilde (witty!)
    345,    # Dracula
    16,     # Peter Pan
    74,     # Tom Sawyer
    76,     # Huckleberry Finn
    1232,   # The Prince - Machiavelli
    2600,   # War and Peace
    2591,   # Grimm's Fairy Tales
    1400,   # Great Expectations
    158,    # Emma - Austen
    161,    # Sense and Sensibility
    105,    # Persuasion
    55,     # The Wonderful Wizard of Oz
    120,    # Treasure Island
    35,     # The Time Machine
    36,     # The War of the Worlds
    43,     # Jekyll and Hyde
    1260,   # Jane Eyre
    768,    # Wuthering Heights
    5200,   # Metamorphosis - Kafka
    2554,   # Crime and Punishment
    2814,   # Dubliners - Joyce
    4300,   # Ulysses (advanced prose)
    1184,   # Count of Monte Cristo
    244,    # A Study in Scarlet
    3207,   # Leviathan
    135,    # Les Miserables
    2500,   # Siddhartha
    215,    # The Call of the Wild
    1727,   # The Odyssey
    6130,   # The Iliad
    23,     # Narrative of Frederick Douglass
    514,    # Little Women
    1080,   # A Modest Proposal - Swift (satirical!)
    829,    # Gulliver's Surveys
    730,    # Oliver Twist
    19942,  # Candide - Voltaire
    28054,  # Brothers Karamazov
    5740,   # Scarlet Pimpernel
    1998,   # Thus Spake Zarathustra
    4363,   # Beyond Good and Evil
    996,    # Don Quixote
    3600,   # Essays of Montaigne
    7370,   # Second Treatise of Government
    46,     # A Christmas Carol
    1497,   # Republic - Plato
    30254,  # The Art of War
    2680,   # Meditations - Marcus Aurelius
    45,     # Anne of Green Gables
    64317,  # The Great Gatsby
    110,    # Tess of the d'Urbervilles
    20203,  # Autobiography of Benjamin Franklin
    33,     # The Scarlet Letter
    209,    # Turn of the Screw
    1251,   # Le Morte d'Arthur
    541,    # The Age of Innocence
    140,    # The Jungle
    100,    # Complete Works of Shakespeare
    27827,  # The Kama Sutra (different prose style!)
    160,    # The Awakening - Kate Chopin
    521,    # The Life and Adventures of Robinson Crusoe
    2852,   # The Hound of the Baskervilles
    103,    # Around the World in 80 Days
    779,    # The Adventures of Pinocchio
    219,    # Heart of Darkness
    5230,   # The Idiot - Dostoevsky
    7849,   # Notes from Underground
    600,    # Notes on Democracy - Mencken
    8800,   # The Divine Comedy
    4217,   # A Portrait of the Artist
    3825,   # Pygmalion - Shaw
    1322,   # Leaves of Grass - Whitman
    1321,   # Paradise Lost
    236,    # The Jungle Book
    32325,  # The Sun Also Rises (wait, not public domain yet)
    1023,   # Bleak House
    766,    # David Copperfield
    1946,   # The Golden Bowl - Henry James
    432,    # The Ambassadors
    208,    # Daisy Miller
]


def download_gutenberg():
    """Download books from Project Gutenberg."""
    print("\n" + "="*60)
    print("Downloading Project Gutenberg classics...")
    print("="*60)

    output_dir = DATA_DIR / "gutenberg"
    output_dir.mkdir(exist_ok=True)

    texts = []
    failed = []

    for book_id in tqdm(GUTENBERG_BOOKS, desc="Downloading books"):
        # Try multiple URL patterns (Gutenberg uses different formats)
        urls = [
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        ]

        success = False
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    text = response.text
                    # Clean the text
                    text = clean_gutenberg_text(text)
                    if len(text) > 1000:  # Skip if too short
                        texts.append(text)
                        success = True
                        break
            except:
                continue

        if not success:
            failed.append(book_id)

    print(f"Downloaded {len(texts)} books, {len(failed)} failed")

    # Save combined text
    output_file = output_dir / "gutenberg_combined.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n<|endoftext|>\n\n".join(texts))

    total_chars = sum(len(t) for t in texts)
    print(f"Total characters: {total_chars:,} (~{total_chars//4:,} tokens)")

    return output_file


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg headers/footers and clean text."""
    # Find start of actual content
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "***START OF THE PROJECT GUTENBERG",
    ]

    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    # Find and remove header
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of this line
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
                break

    # Find and remove footer
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    text = text[start_idx:end_idx]

    # Clean up whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# =============================================================================
# Reddit WritingPrompts - Creative Fiction
# =============================================================================

def download_writingprompts():
    """Download WritingPrompts dataset from Hugging Face."""
    print("\n" + "="*60)
    print("Downloading Reddit WritingPrompts...")
    print("="*60)

    output_dir = DATA_DIR / "writingprompts"
    output_dir.mkdir(exist_ok=True)

    try:
        from datasets import load_dataset

        # Load the WritingPrompts dataset
        print("Loading from Hugging Face...")
        dataset = load_dataset("euclaise/writingprompts", split="train")

        texts = []
        for item in tqdm(dataset, desc="Processing stories"):
            prompt = item.get('prompt', '')
            story = item.get('story', '')

            if story and len(story) > 500:  # Skip very short stories
                # Format: prompt followed by story
                text = f"[WP] {prompt}\n\n{story}"
                texts.append(text)

        print(f"Collected {len(texts)} stories")

        # Save
        output_file = output_dir / "writingprompts.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n<|endoftext|>\n\n".join(texts))

        total_chars = sum(len(t) for t in texts)
        print(f"Total characters: {total_chars:,} (~{total_chars//4:,} tokens)")

        return output_file

    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        return download_writingprompts()
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative source...")
        return download_writingprompts_alternative()


def download_writingprompts_alternative():
    """Alternative: download from direct source."""
    output_dir = DATA_DIR / "writingprompts"
    output_dir.mkdir(exist_ok=True)

    # Try the processed version
    url = "https://huggingface.co/datasets/euclaise/writingprompts/resolve/main/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd
        dest = output_dir / "wp.parquet"

        if download_file(url, dest, "WritingPrompts"):
            df = pd.read_parquet(dest)
            texts = []

            for _, row in df.iterrows():
                prompt = row.get('prompt', '')
                story = row.get('story', '')
                if story and len(story) > 500:
                    texts.append(f"[WP] {prompt}\n\n{story}")

            output_file = output_dir / "writingprompts.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n<|endoftext|>\n\n".join(texts))

            return output_file
    except Exception as e:
        print(f"Alternative download failed: {e}")
        return None


# =============================================================================
# Cornell Movie Dialogs - Snappy Dialogue with Personality
# =============================================================================

def download_movie_dialogs():
    """Download Cornell Movie-Dialogs Corpus."""
    print("\n" + "="*60)
    print("Downloading Cornell Movie Dialogs...")
    print("="*60)

    output_dir = DATA_DIR / "movie_dialogs"
    output_dir.mkdir(exist_ok=True)

    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = output_dir / "cornell_movie_dialogs.zip"

    if not zip_path.exists():
        if not download_file(url, zip_path, "Cornell Movie Dialogs"):
            print("Failed to download, trying mirror...")
            # Try Kaggle mirror or other source
            url = "https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip"
            if not download_file(url, zip_path, "Cornell Movie Dialogs (mirror)"):
                return None

    # Extract
    print("Extracting...")
    extract_dir = output_dir / "extracted"

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

    # Find and parse the dialog files
    # Look for movie_lines.txt and movie_conversations.txt
    lines_file = None
    conv_file = None

    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if 'movie_lines' in f.lower() and f.endswith('.txt'):
                lines_file = Path(root) / f
            if 'movie_conversations' in f.lower() and f.endswith('.txt'):
                conv_file = Path(root) / f

    if not lines_file:
        print("Could not find movie_lines.txt")
        return None

    print(f"Found: {lines_file}")

    # Parse lines
    lines_dict = {}
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(lines_file, 'r', encoding=encoding, errors='replace') as f:
                for line in f:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 5:
                        line_id = parts[0]
                        text = parts[4]
                        lines_dict[line_id] = text
            break
        except:
            continue

    print(f"Parsed {len(lines_dict)} dialog lines")

    # Parse conversations if available
    conversations = []
    if conv_file and conv_file.exists():
        for encoding in encodings:
            try:
                with open(conv_file, 'r', encoding=encoding, errors='replace') as f:
                    for line in f:
                        parts = line.strip().split(' +++$+++ ')
                        if len(parts) >= 4:
                            line_ids = eval(parts[3])  # List of line IDs
                            conv_lines = [lines_dict.get(lid, '') for lid in line_ids]
                            conv_lines = [l for l in conv_lines if l]
                            if len(conv_lines) >= 2:
                                conversations.append('\n'.join(conv_lines))
                break
            except:
                continue

    if not conversations:
        # Just use all lines grouped by chunks
        all_lines = list(lines_dict.values())
        conversations = []
        for i in range(0, len(all_lines), 5):
            chunk = all_lines[i:i+5]
            if len(chunk) >= 2:
                conversations.append('\n'.join(chunk))

    print(f"Created {len(conversations)} conversation chunks")

    # Save
    output_file = output_dir / "movie_dialogs.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n<|endoftext|>\n\n".join(conversations))

    total_chars = sum(len(c) for c in conversations)
    print(f"Total characters: {total_chars:,} (~{total_chars//4:,} tokens)")

    return output_file


# =============================================================================
# OpenWebText Sample - General Web Text
# =============================================================================

def download_openwebtext_sample():
    """Download a sample of OpenWebText for diversity."""
    print("\n" + "="*60)
    print("Downloading OpenWebText sample...")
    print("="*60)

    output_dir = DATA_DIR / "openwebtext"
    output_dir.mkdir(exist_ok=True)

    try:
        from datasets import load_dataset

        # Load a small subset (first 50k documents)
        print("Loading from Hugging Face (this may take a while)...")
        dataset = load_dataset("openwebtext", split="train", streaming=True)

        texts = []
        for i, item in enumerate(tqdm(dataset, desc="Downloading", total=50000)):
            if i >= 50000:
                break
            text = item.get('text', '')
            if len(text) > 200:
                texts.append(text)

        print(f"Collected {len(texts)} documents")

        # Save
        output_file = output_dir / "openwebtext_sample.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n<|endoftext|>\n\n".join(texts))

        total_chars = sum(len(t) for t in texts)
        print(f"Total characters: {total_chars:,} (~{total_chars//4:,} tokens)")

        return output_file

    except Exception as e:
        print(f"Error: {e}")
        return None


# =============================================================================
# Poetry - Lyrical, Dense Language
# =============================================================================

def download_poetry():
    """Download poetry for lyrical prose patterns."""
    print("\n" + "="*60)
    print("Downloading Poetry corpus...")
    print("="*60)

    output_dir = DATA_DIR / "poetry"
    output_dir.mkdir(exist_ok=True)

    try:
        from datasets import load_dataset

        # Try multiple poetry datasets
        texts = []

        # PoetryFoundation dataset
        try:
            dataset = load_dataset("merve/poetry", split="train")
            for item in tqdm(dataset, desc="Processing poems"):
                poem = item.get('content', item.get('poem', ''))
                if poem and len(poem) > 50:
                    texts.append(poem)
        except:
            print("PoetryFoundation not available, trying alternative...")

        # Gutenberg poetry books
        poetry_books = [
            1321,   # Paradise Lost
            1322,   # Leaves of Grass
            8800,   # Divine Comedy
            100,    # Shakespeare's Sonnets (in complete works)
        ]

        for book_id in poetry_books:
            try:
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    text = clean_gutenberg_text(response.text)
                    if text:
                        texts.append(text)
            except:
                continue

        if texts:
            output_file = output_dir / "poetry.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n<|endoftext|>\n\n".join(texts))

            total_chars = sum(len(t) for t in texts)
            print(f"Collected {len(texts)} poems/texts")
            print(f"Total characters: {total_chars:,} (~{total_chars//4:,} tokens)")

            return output_file

    except Exception as e:
        print(f"Error: {e}")

    return None


# =============================================================================
# Main Download Orchestration
# =============================================================================

def download_all():
    """Download all datasets."""
    print("="*60)
    print("CREATIVE WRITER DATA DOWNLOAD")
    print("="*60)
    print("\nTarget: Verbose, eloquent model with personality")
    print("Datasets: Gutenberg, WritingPrompts, Movie Dialogs, Poetry")
    print("="*60)

    results = {}

    # Download each dataset
    results['gutenberg'] = download_gutenberg()
    results['writingprompts'] = download_writingprompts()
    results['movie_dialogs'] = download_movie_dialogs()
    results['poetry'] = download_poetry()

    # Skip OpenWebText for now - focus on creative data
    # results['openwebtext'] = download_openwebtext_sample()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    total_size = 0
    for name, path in results.items():
        if path and path.exists():
            size = path.stat().st_size
            total_size += size
            print(f"✓ {name}: {path.name} ({size/1e6:.1f} MB)")
        else:
            print(f"✗ {name}: FAILED")

    print(f"\nTotal data size: {total_size/1e6:.1f} MB")
    print(f"Estimated tokens: ~{total_size//4:,}")

    # Save manifest
    manifest = {
        'datasets': {k: str(v) if v else None for k, v in results.items()},
        'total_size_bytes': total_size,
    }

    with open(DATA_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    return results


if __name__ == "__main__":
    download_all()
