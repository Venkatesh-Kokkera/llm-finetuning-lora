import json
import argparse
import random
from pathlib import Path

def load_raw_data(input_dir: str) -> list:
    """Load raw text files from directory."""
    data = []
    for file in Path(input_dir).glob("**/*.txt"):
        with open(file, "r", errors="ignore") as f:
            text = f.read().strip()
            if text:
                data.append({
                    "text": text,
                    "source": file.name
                })
    print(f"Loaded {len(data)} raw documents")
    return data

def format_as_instruction(sample: dict) -> dict:
    """Format raw text as instruction-response pair."""
    text = sample["text"]
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return None

    # Use first sentence as instruction
    instruction = sentences[0] + "."
    # Use rest as response
    response = ". ".join(sentences[1:]) + "."

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "source": sample["source"]
    }

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters
    text = text.replace("\x00", "")
    text = text.replace("\ufffd", "")
    return text.strip()

def deduplicate(data: list) -> list:
    """Remove duplicate entries."""
    seen = set()
    unique = []
    for item in data:
        key = item["instruction"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"After deduplication: {len(unique)} samples")
    return unique

def split_dataset(
    data: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> tuple:
    """Split into train, val, test sets."""
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def save_jsonl(data: list, output_path: str):
    """Save data as JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} samples to {output_path}")

def prepare(args):
    """Main data preparation pipeline."""
    print("Loading raw data...")
    raw_data = load_raw_data(args.input)

    print("Formatting as instructions...")
    formatted = []
    for sample in raw_data:
        sample["text"] = clean_text(sample["text"])
        formatted_sample = format_as_instruction(sample)
        if formatted_sample:
            formatted.append(formatted_sample)

    print(f"Formatted {len(formatted)} samples")

    # Deduplicate
    formatted = deduplicate(formatted)

    # Split
    train, val, test = split_dataset(formatted)

    # Save
    save_jsonl(train, f"{args.output}/train.jsonl")
    save_jsonl(val, f"{args.output}/val.jsonl")
    save_jsonl(test, f"{args.output}/test.jsonl")

    print("✅ Data preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw",
        help="Input directory with raw text files"
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for processed data"
    )
    args = parser.parse_args()
    prepare(args)
