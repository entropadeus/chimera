"""
Download and prepare OpenAssistant (OASST) dataset for Chimera fine-tuning.

This downloads the high-quality human conversation data and converts it
to our training format.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: Please install datasets library:")
    print("  pip install datasets")
    exit(1)


def build_conversation_trees(messages):
    """
    OASST data is stored as a tree of messages (replies to replies).
    We need to extract linear conversation paths.
    """
    # Build parent-child relationships
    children = defaultdict(list)
    msg_by_id = {}

    for msg in messages:
        msg_by_id[msg["message_id"]] = msg
        parent_id = msg.get("parent_id")
        if parent_id:
            children[parent_id].append(msg["message_id"])

    # Find root messages (no parent)
    roots = [msg["message_id"] for msg in messages if msg.get("parent_id") is None]

    # Extract all conversation paths via DFS
    conversations = []

    def dfs(msg_id, path):
        msg = msg_by_id[msg_id]
        path = path + [msg]

        if not children[msg_id]:
            # Leaf node - this is a complete conversation
            if len(path) >= 2:  # At least user + assistant
                conversations.append(path)
        else:
            # Continue down each child branch
            for child_id in children[msg_id]:
                dfs(child_id, path)

    for root_id in roots:
        dfs(root_id, [])

    return conversations


def convert_to_training_format(conversation_path, max_turns=6):
    """Convert OASST message path to our training format."""
    turns = []

    for msg in conversation_path[:max_turns * 2]:  # Limit turns
        role = "user" if msg["role"] == "prompter" else "assistant"
        content = msg["text"].strip()

        # Skip empty messages
        if not content:
            continue

        turns.append({"role": role, "content": content})

    # Ensure conversation starts with user and alternates
    if not turns or turns[0]["role"] != "user":
        return None

    # Ensure we have at least one exchange
    if len(turns) < 2:
        return None

    # Ensure it ends with assistant (trim if needed)
    if turns[-1]["role"] == "user":
        turns = turns[:-1]

    if len(turns) < 2:
        return None

    return {
        "system": None,  # OASST doesn't have system prompts
        "conversations": turns
    }


def filter_quality(msg):
    """Filter for high-quality messages based on OASST labels."""
    # Check if message was marked as good quality
    labels = msg.get("labels")

    # If no labels, accept by default (many messages don't have labels)
    if labels is None:
        return True

    # Handle different label formats
    if isinstance(labels, dict):
        # Skip if marked as spam, not helpful, or harmful
        if labels.get("spam", 0) > 0.5:
            return False
        if labels.get("fails_task", 0) > 0.5:
            return False
        if labels.get("not_appropriate", 0) > 0.5:
            return False

        # Prefer messages with good quality scores
        quality = labels.get("quality")
        if isinstance(quality, dict):
            avg_quality = quality.get("value", 0.5)
            if avg_quality < 0.3:  # Skip very low quality
                return False
        elif isinstance(quality, (int, float)):
            if quality < 0.3:
                return False

    return True


def download_and_prepare(output_path: str, max_examples: int = 10000, lang: str = "en"):
    """Download OASST and prepare for training."""

    print("Downloading OpenAssistant dataset from HuggingFace...")
    print("(This may take a minute on first run)\n")

    # Load OASST1 dataset
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative: timdettmers/openassistant-guanaco")
        # Fallback to pre-processed guanaco version
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

        # Guanaco is already formatted, just convert
        examples = []
        for item in dataset:
            if len(examples) >= max_examples:
                break

            text = item.get("text", "")
            # Parse the format: ### Human: ... ### Assistant: ...
            parts = text.split("### ")
            turns = []
            for part in parts:
                if part.startswith("Human:"):
                    turns.append({"role": "user", "content": part[6:].strip()})
                elif part.startswith("Assistant:"):
                    turns.append({"role": "assistant", "content": part[10:].strip()})

            if len(turns) >= 2:
                examples.append({
                    "system": None,
                    "conversations": turns
                })

        save_dataset(examples, output_path)
        return

    print(f"Loaded {len(dataset)} messages")

    # Filter by language
    print(f"Filtering for language: {lang}")
    messages = [msg for msg in dataset if msg.get("lang") == lang]
    print(f"After language filter: {len(messages)} messages")

    # Apply quality filter
    print("Applying quality filters...")
    messages = [msg for msg in messages if filter_quality(msg)]
    print(f"After quality filter: {len(messages)} messages")

    # Build conversation trees
    print("Building conversation trees...")
    conversations = build_conversation_trees(messages)
    print(f"Extracted {len(conversations)} conversation paths")

    # Convert to training format
    print("Converting to training format...")
    examples = []
    for conv_path in conversations:
        if len(examples) >= max_examples:
            break

        example = convert_to_training_format(conv_path)
        if example:
            examples.append(example)

    save_dataset(examples, output_path)


def save_dataset(examples, output_path):
    """Save dataset to JSONL file."""
    # Shuffle
    import random
    random.shuffle(examples)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(examples)} conversations to {output_path}")

    # Stats
    total_turns = sum(len(ex["conversations"]) for ex in examples)
    avg_turns = total_turns / len(examples) if examples else 0
    multi_turn = sum(1 for ex in examples if len(ex["conversations"]) > 2)

    print(f"\nDataset stats:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Total turns: {total_turns}")
    print(f"  Avg turns/conversation: {avg_turns:.1f}")
    print(f"  Multi-turn conversations: {multi_turn} ({100*multi_turn/len(examples):.1f}%)")

    # Show samples
    print("\n=== Sample Conversations ===\n")
    import random as rnd
    for ex in rnd.sample(examples, min(2, len(examples))):
        for turn in ex["conversations"][:4]:  # First 4 turns only
            role = turn["role"].upper()
            content = turn["content"][:200] + "..." if len(turn["content"]) > 200 else turn["content"]
            print(f"{role}: {content}")
        if len(ex["conversations"]) > 4:
            print(f"... ({len(ex['conversations']) - 4} more turns)")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Download OpenAssistant dataset")
    parser.add_argument("--output", default="data/oasst_data.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max-examples", type=int, default=10000,
                        help="Maximum number of conversations to extract")
    parser.add_argument("--lang", default="en",
                        help="Language filter (default: en)")
    args = parser.parse_args()

    download_and_prepare(args.output, args.max_examples, args.lang)

    print("\n" + "="*50)
    print("Done! To use this data for training:")
    print(f"  python create_instruct_data.py --external {args.output} --external-format oasst")
    print("\nOr train directly on OASST only:")
    print(f"  python train_instruct.py --data-path {args.output}")


if __name__ == "__main__":
    main()
