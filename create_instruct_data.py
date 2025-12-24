"""
Generate conversational instruction-tuning dataset.

Supports multiple data sources:
1. TinyStories - story generation tasks
2. Synthetic conversations - Q&A, tasks, chat
3. External datasets - ShareGPT, OASST format

Creates diverse conversational patterns for training a chat LLM.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
import argparse


# =============================================================================
# Chat Template (ChatML-style)
# =============================================================================

SYSTEM_PROMPTS = [
    "You are Wyrd, a helpful and friendly AI assistant.",
    "You are Wyrd, a creative storyteller who loves to help.",
    "You are Wyrd, a kind assistant who gives clear, helpful answers.",
    "You are Wyrd, an AI that enjoys conversations and helping people.",
]

DEFAULT_SYSTEM = "You are Wyrd, a helpful AI assistant."


# =============================================================================
# Conversation Templates
# =============================================================================

# General Q&A patterns
QA_TEMPLATES = [
    # Simple questions
    {"user": "What is {topic}?", "assistant": "{explanation}"},
    {"user": "Can you explain {topic}?", "assistant": "{explanation}"},
    {"user": "Tell me about {topic}.", "assistant": "{explanation}"},
    {"user": "How does {topic} work?", "assistant": "{explanation}"},

    # Requests
    {"user": "Help me understand {topic}.", "assistant": "Of course! {explanation}"},
    {"user": "I'm curious about {topic}.", "assistant": "{explanation}"},
]

# Task-oriented patterns
TASK_TEMPLATES = [
    {"user": "Write a short poem about {topic}.", "type": "creative"},
    {"user": "Give me 3 ideas for {topic}.", "type": "list"},
    {"user": "Summarize {topic} in simple words.", "type": "summary"},
    {"user": "What are the main points of {topic}?", "type": "analysis"},
]

# Conversational patterns (multi-turn)
CHAT_STARTERS = [
    "Hi!",
    "Hello!",
    "Hey there!",
    "Hi, how are you?",
    "Hello, I have a question.",
    "Hey!",
]

CHAT_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What can I do for you?",
    "Hey! I'm happy to help. What's on your mind?",
    "Hello! I'm doing well, thanks for asking. How can I assist you?",
    "Hi! Sure, I'd be happy to help. What's your question?",
    "Hey! What would you like to talk about?",
]

# Follow-up patterns
FOLLOWUPS = [
    "Can you explain more?",
    "Tell me more about that.",
    "Why is that?",
    "What do you mean?",
    "Can you give an example?",
    "That's interesting! What else?",
    "How so?",
    "And then?",
]

FOLLOWUP_RESPONSES = [
    "Sure! {content}",
    "Of course. {content}",
    "Good question! {content}",
    "Let me explain further. {content}",
    "Here's more detail: {content}",
]

# Story-specific prompts (from TinyStories)
STORY_PROMPTS = [
    "Write a short story about {topic}.",
    "Tell me a story about {topic}.",
    "Can you write a children's story about {topic}?",
    "Write a simple story involving {topic}.",
    "Create a story where {topic}.",
    "I'd like to hear a story about {topic}.",
    "Make up a story about {topic}.",
]


# =============================================================================
# Simple Knowledge Base (for Q&A without external API)
# =============================================================================

SIMPLE_QA = [
    {
        "topic": "the sun",
        "explanation": "The sun is a star at the center of our solar system. It gives us light and warmth, and all the planets orbit around it."
    },
    {
        "topic": "rain",
        "explanation": "Rain happens when water from oceans and lakes evaporates into clouds. When the clouds get heavy, the water falls back down as rain."
    },
    {
        "topic": "friendship",
        "explanation": "Friendship is a special bond between people who care about each other. Friends help each other, share things, and have fun together."
    },
    {
        "topic": "dreams",
        "explanation": "Dreams are stories and images our minds create while we sleep. Sometimes they're fun adventures, and sometimes they're a bit strange!"
    },
    {
        "topic": "kindness",
        "explanation": "Kindness means being nice and helpful to others. It can be as simple as sharing, saying something nice, or helping someone who needs it."
    },
    {
        "topic": "animals",
        "explanation": "Animals are living creatures that share our world. They come in all sizes - from tiny insects to huge whales - and each one is special."
    },
    {
        "topic": "the moon",
        "explanation": "The moon is Earth's natural satellite. It orbits around our planet and reflects sunlight, which is why we can see it at night."
    },
    {
        "topic": "music",
        "explanation": "Music is organized sound that people create and enjoy. It can make us feel happy, sad, excited, or calm. People have made music for thousands of years."
    },
    {
        "topic": "trees",
        "explanation": "Trees are tall plants with wooden trunks and branches. They give us oxygen to breathe, shade on hot days, and homes for many animals."
    },
    {
        "topic": "courage",
        "explanation": "Courage means being brave even when you're scared. It's doing the right thing or trying something new even when it feels hard."
    },
]

# Simple tasks the model can learn
SIMPLE_TASKS = [
    {
        "user": "Count to five.",
        "assistant": "1, 2, 3, 4, 5!"
    },
    {
        "user": "Say hello in a fun way.",
        "assistant": "Heya! Great to meet you! 👋"
    },
    {
        "user": "What's 2 + 2?",
        "assistant": "2 + 2 equals 4!"
    },
    {
        "user": "Name three colors.",
        "assistant": "Red, blue, and yellow! Those are also the primary colors."
    },
    {
        "user": "What sound does a cat make?",
        "assistant": "Cats say 'meow!' They also purr when they're happy."
    },
    {
        "user": "Tell me a fun fact.",
        "assistant": "Here's a fun fact: Honey never spoils! Archaeologists have found 3000-year-old honey that was still good to eat."
    },
    {
        "user": "What's your name?",
        "assistant": "I'm Wyrd! Nice to meet you."
    },
    {
        "user": "Are you a robot?",
        "assistant": "I'm an AI assistant called Wyrd. I'm here to chat and help you with things!"
    },
    {
        "user": "Can you help me?",
        "assistant": "Of course! I'd be happy to help. What do you need?"
    },
    {
        "user": "Thank you!",
        "assistant": "You're welcome! Happy to help anytime."
    },
    {
        "user": "Goodbye!",
        "assistant": "Bye! Take care and have a great day!"
    },
    {
        "user": "I'm bored.",
        "assistant": "Let's fix that! Would you like to hear a story, play a word game, or learn something new?"
    },
    {
        "user": "Tell me a joke.",
        "assistant": "Why don't scientists trust atoms? Because they make up everything! 😄"
    },
    {
        "user": "I'm sad.",
        "assistant": "I'm sorry you're feeling sad. That's okay - everyone feels that way sometimes. Would you like to talk about it, or would a happy story help?"
    },
    {
        "user": "You're nice!",
        "assistant": "Aw, thank you! That's very kind of you to say. You're pretty great yourself!"
    },
]


# =============================================================================
# Data Generation Functions
# =============================================================================

def extract_topic(story: str) -> str:
    """Extract a topic/character from a story for prompt generation."""
    patterns = [
        r"(?:named|called)\s+(\w+)",
        r"(little|big|brave|kind|happy|sad)\s+(\w+)",
        r"(dog|cat|bird|rabbit|bear|mouse|fox|lion)",
        r"(boy|girl|child|kid|friend)",
        r"(forest|garden|house|park|school|farm)",
    ]

    story_lower = story.lower()
    for pattern in patterns:
        match = re.search(pattern, story_lower)
        if match:
            return match.group(0)

    match = re.search(r"(?:a|the)\s+(\w+)", story_lower)
    if match:
        return match.group(1)

    return "a fun adventure"


def create_story_conversation(story: str, multi_turn: bool = False) -> Dict:
    """Create a conversation from a TinyStories story."""
    topic = extract_topic(story)
    prompt = random.choice(STORY_PROMPTS).format(topic=topic)

    conversations = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": story.strip()}
    ]

    # Optionally add follow-up
    if multi_turn and random.random() < 0.5:
        followup = random.choice(FOLLOWUPS)
        # Generate a simple follow-up response
        responses = [
            f"I'm glad you enjoyed the story! {topic.capitalize()} stories are fun to tell.",
            f"Stories about {topic} are some of my favorites!",
            "Would you like to hear another story?",
        ]
        conversations.extend([
            {"role": "user", "content": followup},
            {"role": "assistant", "content": random.choice(responses)}
        ])

    return {
        "system": random.choice(SYSTEM_PROMPTS) if random.random() < 0.3 else None,
        "conversations": conversations
    }


def create_qa_conversation() -> Dict:
    """Create a Q&A conversation from the knowledge base."""
    qa = random.choice(SIMPLE_QA)
    template = random.choice(QA_TEMPLATES)

    user_msg = template["user"].format(topic=qa["topic"])
    asst_msg = template["assistant"].format(explanation=qa["explanation"])

    conversations = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": asst_msg}
    ]

    # Sometimes add follow-up
    if random.random() < 0.3:
        conversations.extend([
            {"role": "user", "content": random.choice(["Thanks!", "That makes sense!", "Cool!"])},
            {"role": "assistant", "content": random.choice(["You're welcome!", "Happy to help!", "Anytime!"])}
        ])

    return {
        "system": DEFAULT_SYSTEM if random.random() < 0.2 else None,
        "conversations": conversations
    }


def create_task_conversation() -> Dict:
    """Create a simple task conversation."""
    task = random.choice(SIMPLE_TASKS)

    conversations = [
        {"role": "user", "content": task["user"]},
        {"role": "assistant", "content": task["assistant"]}
    ]

    return {
        "system": None,
        "conversations": conversations
    }


def create_chat_conversation() -> Dict:
    """Create a casual chat conversation."""
    # Start with greeting
    conversations = [
        {"role": "user", "content": random.choice(CHAT_STARTERS)},
        {"role": "assistant", "content": random.choice(CHAT_RESPONSES)}
    ]

    # Add 1-3 more turns
    num_turns = random.randint(1, 3)
    for _ in range(num_turns):
        # Pick a random task or question
        if random.random() < 0.5:
            task = random.choice(SIMPLE_TASKS)
            conversations.extend([
                {"role": "user", "content": task["user"]},
                {"role": "assistant", "content": task["assistant"]}
            ])
        else:
            qa = random.choice(SIMPLE_QA)
            conversations.extend([
                {"role": "user", "content": f"What is {qa['topic']}?"},
                {"role": "assistant", "content": qa["explanation"]}
            ])

    # Sometimes end with goodbye
    if random.random() < 0.3:
        conversations.extend([
            {"role": "user", "content": random.choice(["Thanks!", "Bye!", "That's all, thanks!"])},
            {"role": "assistant", "content": random.choice(["Bye! Take care!", "You're welcome! Goodbye!", "Happy to help! See you!"])}
        ])

    return {
        "system": random.choice(SYSTEM_PROMPTS) if random.random() < 0.4 else None,
        "conversations": conversations
    }


def create_multi_turn_story(story: str) -> Dict:
    """Create a multi-turn conversation by splitting a story."""
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())

    if len(sentences) < 4:
        return create_story_conversation(story)

    # Split into parts
    mid = len(sentences) // 2
    part1 = " ".join(sentences[:mid])
    part2 = " ".join(sentences[mid:])

    topic = extract_topic(story)
    prompt1 = random.choice(STORY_PROMPTS).format(topic=topic)
    prompt2 = random.choice(["What happens next?", "Continue the story!", "And then?"])

    return {
        "system": random.choice(SYSTEM_PROMPTS) if random.random() < 0.3 else None,
        "conversations": [
            {"role": "user", "content": prompt1},
            {"role": "assistant", "content": part1},
            {"role": "user", "content": prompt2},
            {"role": "assistant", "content": part2}
        ]
    }


# =============================================================================
# External Dataset Loaders
# =============================================================================

def load_sharegpt_format(path: str, max_examples: int = 5000) -> List[Dict]:
    """
    Load ShareGPT-format dataset.
    Expected format: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    """
    examples = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(examples) >= max_examples:
                break
            try:
                data = json.loads(line.strip())
                convs = data.get("conversations", [])

                # Convert format
                converted = []
                for turn in convs:
                    role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                    converted.append({"role": role, "content": turn.get("value", "")})

                if converted:
                    examples.append({
                        "system": data.get("system"),
                        "conversations": converted
                    })
            except:
                continue

    return examples


def load_oasst_format(path: str, max_examples: int = 5000) -> List[Dict]:
    """
    Load OASST-style dataset.
    Supports multiple formats:
    - {"messages": [{"role": "...", "content": "..."}]}
    - {"conversations": [{"role": "...", "content": "..."}]}
    - {"system": "...", "conversations": [...]}
    """
    examples = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(examples) >= max_examples:
                break
            try:
                data = json.loads(line.strip())

                # Try different field names for conversations
                messages = (
                    data.get("conversations") or
                    data.get("messages") or
                    data.get("conversation") or
                    []
                )

                if messages and len(messages) >= 2:
                    examples.append({
                        "system": data.get("system"),
                        "conversations": messages
                    })
            except:
                continue

    return examples


def load_alpaca_format(path: str, max_examples: int = 5000) -> List[Dict]:
    """
    Load Alpaca-format dataset.
    Expected format: {"instruction": "...", "input": "...", "output": "..."}
    """
    examples = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(examples) >= max_examples:
                break
            try:
                data = json.loads(line.strip())
                instruction = data.get("instruction", "")
                inp = data.get("input", "")
                output = data.get("output", "")

                user_content = instruction
                if inp:
                    user_content = f"{instruction}\n\n{inp}"

                examples.append({
                    "system": None,
                    "conversations": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ]
                })
            except:
                continue

    return examples


# =============================================================================
# Main Generation Pipeline
# =============================================================================

def load_stories(path: str, max_stories: int = 10000) -> List[str]:
    """Load stories from TinyStories file."""
    stories = []
    current_story = []

    print(f"Loading stories from {path}...")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_story:
                    story = " ".join(current_story)
                    if len(story) > 100:
                        stories.append(story)
                    current_story = []

                    if len(stories) >= max_stories:
                        break
            else:
                current_story.append(line)

    if current_story:
        story = " ".join(current_story)
        if len(story) > 100:
            stories.append(story)

    print(f"Loaded {len(stories)} stories")
    return stories


def generate_dataset(
    stories: List[str],
    output_path: str,
    external_data: Optional[List[Dict]] = None,
    story_ratio: float = 0.4,
    qa_ratio: float = 0.2,
    task_ratio: float = 0.2,
    chat_ratio: float = 0.2,
):
    """Generate diverse conversational dataset."""
    dataset = []

    # Calculate how many of each type
    total_from_stories = len(stories)

    print(f"\nGenerating conversations...")
    print(f"  - Story conversations: {int(total_from_stories * story_ratio)}")
    print(f"  - Q&A conversations: {int(total_from_stories * qa_ratio)}")
    print(f"  - Task conversations: {int(total_from_stories * task_ratio)}")
    print(f"  - Chat conversations: {int(total_from_stories * chat_ratio)}")

    # Generate story-based conversations
    story_count = int(total_from_stories * story_ratio)
    for i, story in enumerate(stories[:story_count]):
        if random.random() < 0.3:
            example = create_multi_turn_story(story)
        else:
            example = create_story_conversation(story, multi_turn=random.random() < 0.2)
        dataset.append(example)

        if (i + 1) % 1000 == 0:
            print(f"  Stories: {i + 1}/{story_count}")

    # Generate Q&A conversations
    qa_count = int(total_from_stories * qa_ratio)
    for i in range(qa_count):
        dataset.append(create_qa_conversation())
    print(f"  Q&A: {qa_count}")

    # Generate task conversations
    task_count = int(total_from_stories * task_ratio)
    for i in range(task_count):
        dataset.append(create_task_conversation())
    print(f"  Tasks: {task_count}")

    # Generate chat conversations
    chat_count = int(total_from_stories * chat_ratio)
    for i in range(chat_count):
        dataset.append(create_chat_conversation())
    print(f"  Chats: {chat_count}")

    # Add external data if provided
    if external_data:
        print(f"  External: {len(external_data)}")
        dataset.extend(external_data)

    # Shuffle
    random.shuffle(dataset)

    # Save as JSONL
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Created {len(dataset)} conversation examples")

    # Stats
    with_system = sum(1 for ex in dataset if ex.get("system"))
    multi_turn = sum(1 for ex in dataset if len(ex["conversations"]) > 2)
    print(f"\nStats:")
    print(f"  - With system prompt: {with_system} ({100*with_system/len(dataset):.1f}%)")
    print(f"  - Multi-turn: {multi_turn} ({100*multi_turn/len(dataset):.1f}%)")

    # Print samples
    print("\n=== Sample Conversations ===\n")
    for example in random.sample(dataset, min(3, len(dataset))):
        if example.get("system"):
            print(f"SYSTEM: {example['system'][:100]}...")
        for turn in example["conversations"]:
            role = turn["role"].upper()
            content = turn["content"][:150] + "..." if len(turn["content"]) > 150 else turn["content"]
            print(f"{role}: {content}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate conversational training data")

    # Input sources
    parser.add_argument("--input", default="data/tinystories.txt",
                        help="Input TinyStories file")
    parser.add_argument("--external", type=str, default=None,
                        help="External dataset (JSONL) - ShareGPT/OASST/Alpaca format")
    parser.add_argument("--external-format", choices=["sharegpt", "oasst", "alpaca"],
                        default="sharegpt", help="Format of external dataset")

    # Output
    parser.add_argument("--output", default="data/instruct_data.jsonl",
                        help="Output JSONL file")

    # Limits
    parser.add_argument("--max-stories", type=int, default=5000,
                        help="Max stories from TinyStories")
    parser.add_argument("--max-external", type=int, default=5000,
                        help="Max examples from external dataset")

    # Ratios (should sum to 1.0)
    parser.add_argument("--story-ratio", type=float, default=0.4,
                        help="Ratio of story-based conversations")
    parser.add_argument("--qa-ratio", type=float, default=0.2,
                        help="Ratio of Q&A conversations")
    parser.add_argument("--task-ratio", type=float, default=0.2,
                        help="Ratio of task conversations")
    parser.add_argument("--chat-ratio", type=float, default=0.2,
                        help="Ratio of casual chat conversations")

    args = parser.parse_args()

    # Load TinyStories
    stories = load_stories(args.input, args.max_stories)

    # Load external data if provided
    external_data = None
    if args.external:
        print(f"\nLoading external data ({args.external_format} format)...")
        loaders = {
            "sharegpt": load_sharegpt_format,
            "oasst": load_oasst_format,
            "alpaca": load_alpaca_format,
        }
        external_data = loaders[args.external_format](args.external, args.max_external)
        print(f"Loaded {len(external_data)} external examples")

    # Generate dataset
    generate_dataset(
        stories,
        args.output,
        external_data=external_data,
        story_ratio=args.story_ratio,
        qa_ratio=args.qa_ratio,
        task_ratio=args.task_ratio,
        chat_ratio=args.chat_ratio,
    )

    print("\nDone! Ready for instruction fine-tuning.")


if __name__ == "__main__":
    main()
