"""
data_loader.py
Downloads the IMDb dataset from HuggingFace and saves cleaned CSVs.
Run this once: python src/data_loader.py
"""

import os
import re
import pandas as pd
from datasets import load_dataset


def clean_text(text: str) -> str:
    """Remove HTML tags, extra spaces, and truncate to 512 chars."""
    # Remove HTML tags (IMDb reviews contain <br /> tags)
    text = re.sub(r'<.*?>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Truncate to 512 characters (BERT's token limit)
    text = text[:512]
    return text


def download_and_save():
    """Download IMDb dataset and save as cleaned CSV files."""
    print("Downloading IMDb dataset from HuggingFace...")
    print("This will download ~80MB. Please wait...")

    # Load dataset (auto-downloads and caches)
    dataset = load_dataset("imdb")

    os.makedirs("data/processed", exist_ok=True)

    for split in ["train", "test"]:
        print(f"\nProcessing {split} split...")

        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[split])

        # Map numeric labels to text
        df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})

        # Clean the text
        df["text"] = df["text"].apply(clean_text)

        # Remove empty rows after cleaning
        df = df[df["text"].str.len() > 10]

        # Keep only what we need
        df = df[["text", "sentiment"]].reset_index(drop=True)

        # Save to CSV
        output_path = f"data/processed/{split}.csv"
        df.to_csv(output_path, index=False)

        # Print summary
        print(f"Saved {output_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Positive: {len(df[df['sentiment']=='positive'])}")
        print(f"  Negative: {len(df[df['sentiment']=='negative'])}")
        print(f"  Sample text: {df['text'].iloc[0][:100]}...")


if __name__ == "__main__":
    download_and_save()
    print("\n✅ Dataset ready! Check data/processed/ folder.")