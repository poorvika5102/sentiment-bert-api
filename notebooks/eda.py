"""
eda.py - Exploratory Data Analysis
Run: python notebooks/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Training samples: {len(train_df)}")
print(f"Test samples    : {len(test_df)}")
print(f"\nTraining label distribution:")
print(train_df["sentiment"].value_counts())

print(f"\nAverage text length (characters): {train_df['text'].str.len().mean():.0f}")
print(f"Min text length: {train_df['text'].str.len().min()}")
print(f"Max text length: {train_df['text'].str.len().max()}")

print(f"\nSample positive review:")
print(train_df[train_df['sentiment']=='positive']['text'].iloc[0][:200])

print(f"\nSample negative review:")
print(train_df[train_df['sentiment']=='negative']['text'].iloc[0][:200])