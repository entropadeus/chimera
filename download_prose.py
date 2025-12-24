"""
Download eloquent prose from Project Gutenberg for Stage 1 fine-tuning.

Focuses on authors known for beautiful, sophisticated prose:
- Oscar Wilde
- Virginia Woolf
- Ray Bradbury (public domain stories)
- Ursula K. Le Guin (essays in public domain)
- Henry James
- Edgar Allan Poe
- H.P. Lovecraft
- Lord Dunsany (fantasy prose)
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Tuple
import urllib.request
import urllib.error


# Gutenberg texts with beautiful prose
# Format: (title, gutenberg_id, author)
ELOQUENT_PROSE = [
    # Oscar Wilde - master of elegant prose
    ("The Picture of Dorian Gray", 174, "Oscar Wilde"),
    ("De Profundis", 921, "Oscar Wilde"),
    ("The Happy Prince and Other Tales", 902, "Oscar Wilde"),
    ("A House of Pomegranates", 873, "Oscar Wilde"),

    # Edgar Allan Poe - atmospheric, lyrical
    ("Tales of Mystery and Imagination", 2147, "Edgar Allan Poe"),
    ("The Fall of the House of Usher", 932, "Edgar Allan Poe"),

    # H.P. Lovecraft - dense, evocative prose
    ("The Call of Cthulhu", 68283, "H.P. Lovecraft"),
    ("At the Mountains of Madness", 70652, "H.P. Lovecraft"),
    ("The Shadow Over Innsmouth", 73181, "H.P. Lovecraft"),

    # Lord Dunsany - poetic fantasy prose
    ("The Gods of Pegana", 8395, "Lord Dunsany"),
    ("The Book of Wonder", 7477, "Lord Dunsany"),
    ("A Dreamer's Tales", 57103, "Lord Dunsany"),
    ("The Sword of Welleran", 8129, "Lord Dunsany"),

    # Virginia Woolf - stream of consciousness
    ("Monday or Tuesday", 29220, "Virginia Woolf"),

    # Henry James - complex, elegant
    ("The Turn of the Screw", 209, "Henry James"),
    ("Daisy Miller", 208, "Henry James"),

    # Nathaniel Hawthorne - rich American prose
    ("The Scarlet Letter", 25344, "Nathaniel Hawthorne"),
    ("Twice Told Tales", 13837, "Nathaniel Hawthorne"),

    # Joseph Conrad - dense, atmospheric
    ("Heart of Darkness", 219, "Joseph Conrad"),
    ("The Secret Sharer", 220, "Joseph Conrad"),

    # Jack London - vivid narrative
    ("The Call of the Wild", 215, "Jack London"),
    ("White Fang", 910, "Jack London"),

    # G.K. Chesterton - witty, paradoxical
    ("The Man Who Was Thursday", 1695, "G.K. Chesterton"),
    ("Orthodoxy", 130, "G.K. Chesterton"),

    # William Morris - fantasy prose pioneer
    ("The Wood Beyond the World", 3164, "William Morris"),
    ("The Well at the World's End", 169, "William Morris"),
]

# Quality narrative fiction (Stage 2 candidates)
NARRATIVE_FICTION = [
    # Short story collections
    ("The Arabian Nights", 2841, "Various"),
    ("Grimm's Fairy Tales", 2591, "Brothers Grimm"),

    # Adventure/Fantasy with good pacing
    ("The Princess and the Goblin", 709, "George MacDonald"),
    ("Phantastes", 325, "George MacDonald"),
    ("At the Back of the North Wind", 716, "George MacDonald"),

    # Mystery/Thriller with narrative drive
    ("The Hound of the Baskervilles", 2852, "Arthur Conan Doyle"),
    ("The Adventures of Sherlock Holmes", 1661, "Arthur Conan Doyle"),

    # Sci-Fi
    ("The Time Machine", 35, "H.G. Wells"),
    ("The War of the Worlds", 36, "H.G. Wells"),
    ("The Island of Doctor Moreau", 159, "H.G. Wells"),

    # More Dunsany (narrative-focused)
    ("The King of Elfland's Daughter", 61077, "Lord Dunsany"),
]


def download_gutenberg_text(book_id: int, output_dir: Path) -> str:
    """Download a book from Project Gutenberg."""
    # Try multiple URL formats
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; ChimeraBot/1.0)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                text = response.read().decode('utf-8', errors='ignore')
                return text
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

    return None


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg boilerplate and clean text."""
    # Find start markers
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "***START OF THE PROJECT GUTENBERG",
    ]

    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    # Find content boundaries
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of the marker line
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    text = text[start_idx:end_idx]

    # Clean up
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove chapter markers that are just numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def download_collection(
    books: List[Tuple[str, int, str]],
    output_dir: Path,
    delay: float = 1.0
) -> List[Path]:
    """Download a collection of books."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for title, book_id, author in books:
        safe_title = re.sub(r'[^\w\s-]', '', title).replace(' ', '_')[:50]
        output_path = output_dir / f"{safe_title}_{book_id}.txt"

        if output_path.exists():
            print(f"  [cached] {title}")
            downloaded.append(output_path)
            continue

        print(f"  Downloading: {title} ({author})...")
        text = download_gutenberg_text(book_id, output_dir)

        if text:
            cleaned = clean_gutenberg_text(text)
            if len(cleaned) > 1000:  # Sanity check
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                downloaded.append(output_path)
                print(f"    -> {len(cleaned):,} chars")
            else:
                print(f"    -> Too short after cleaning, skipped")
        else:
            print(f"    -> Failed to download")

        time.sleep(delay)  # Be nice to Gutenberg servers

    return downloaded


def merge_texts(paths: List[Path], output_path: Path, separator: str = "\n\n---\n\n"):
    """Merge multiple text files into one."""
    total_chars = 0

    with open(output_path, 'w', encoding='utf-8') as outf:
        for i, path in enumerate(paths):
            with open(path, 'r', encoding='utf-8') as inf:
                text = inf.read()
                if i > 0:
                    outf.write(separator)
                outf.write(text)
                total_chars += len(text)

    print(f"Merged {len(paths)} texts -> {output_path}")
    print(f"Total: {total_chars:,} characters ({total_chars/1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(description="Download eloquent prose for fine-tuning")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="1=eloquent prose, 2=narrative fiction")
    parser.add_argument("--output-dir", type=str, default="data/prose",
                        help="Directory to save texts")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all texts into single file")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between downloads (be nice to servers)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.stage == 1:
        print("=== Stage 1: Downloading Eloquent Prose ===")
        books = ELOQUENT_PROSE
        subdir = "eloquent"
    else:
        print("=== Stage 2: Downloading Narrative Fiction ===")
        books = NARRATIVE_FICTION
        subdir = "narrative"

    download_dir = output_dir / subdir
    paths = download_collection(books, download_dir, delay=args.delay)

    if args.merge and paths:
        merge_path = output_dir / f"{subdir}_merged.txt"
        merge_texts(paths, merge_path)

    print(f"\nDownloaded {len(paths)} texts to {download_dir}")


if __name__ == "__main__":
    main()
