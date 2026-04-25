# download_model.py
# Run this once: python download_model.py

from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_PATH = "models/bert-sentiment"
os.makedirs(MODEL_PATH, exist_ok=True)

print("Downloading pre-trained sentiment model from HuggingFace...")
print("This will download ~440MB. Please wait...\n")

# This is a BERT model already fine-tuned on SST-2 sentiment dataset
# It achieves ~93% accuracy — perfect for your resume
HF_MODEL = "textattack/bert-base-uncased-SST-2"

print("Step 1/2 — Downloading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(HF_MODEL)
tokenizer.save_pretrained(MODEL_PATH)
print("✅ Tokenizer saved")

print("Step 2/2 — Downloading model weights (~440MB)...")
model = BertForSequenceClassification.from_pretrained(HF_MODEL)
model.save_pretrained(MODEL_PATH)
print("✅ Model saved")

print(f"\n✅ Done! Model saved to: {MODEL_PATH}")
print("Files saved:")
for f in os.listdir(MODEL_PATH):
    size = os.path.getsize(f"{MODEL_PATH}/{f}") / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")

print("\nNow run: uvicorn api.main:app --reload --port 8000")
