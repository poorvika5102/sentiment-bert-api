"""
main.py
The main FastAPI application.
Run locally: uvicorn api.main:app --reload --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv

from api.schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Global model variables (loaded once at startup)
tokenizer: Optional[BertTokenizer] = None
model: Optional[BertForSequenceClassification] = None
device: Optional[torch.device] = None

# Configuration from .env
MODEL_PATH   = os.getenv("MODEL_PATH", "models/bert-sentiment")
MAX_LENGTH   = int(os.getenv("MAX_TEXT_LENGTH", "256"))
ID2LABEL     = {0: "negative", 1: "positive"}
LABEL2ID     = {"negative": 0, "positive": 1}


# ── Lifespan (startup/shutdown) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global tokenizer, model, device

    logger.info(f"Loading model from: {MODEL_PATH}")
    start = time.time()

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.error("Run training first or copy model files to this path")
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()   # Set to inference mode (disables dropout)

    elapsed = time.time() - start
    logger.info(f"✅ Model loaded in {elapsed:.2f}s")

    yield   # App runs here

    # Cleanup on shutdown
    logger.info("Shutting down — cleaning up model from memory")
    del model, tokenizer


# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="""
## BERT-based Sentiment Analysis API

Fine-tuned on 50,000 IMDb movie reviews using bert-base-uncased.

### Features
- Single text prediction with confidence score
- Batch prediction (up to 32 texts)
- Health check endpoint
- Automatic input validation

### Model Performance
- F1 Score (weighted): **92.4%**
- Accuracy: **91.8%**
- Inference latency: **< 200ms**
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins (for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper function ───────────────────────────────────────
def run_inference(text: str) -> PredictResponse:
    """
    Core inference function.
    Takes raw text, returns structured prediction.
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    # Move tensors to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run model (no gradient needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label_id = int(torch.argmax(probs))

    return PredictResponse(
        text=text,
        sentiment=ID2LABEL[label_id],
        confidence=round(float(probs[label_id]), 4),
        positive_score=round(float(probs[1]), 4),
        negative_score=round(float(probs[0]), 4),
        label_id=label_id,
    )


# ── Endpoints ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check if the API is running and model is loaded",
)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        device=str(device),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Get information about the loaded model",
)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    param_count = sum(p.numel() for p in model.parameters())

    return ModelInfoResponse(
        model_name="bert-base-uncased",
        model_path=MODEL_PATH,
        labels=ID2LABEL,
        max_length=MAX_LENGTH,
        device=str(device),
        parameters=f"{param_count:,}",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment of a single text",
    status_code=status.HTTP_200_OK,
)
def predict(req: PredictRequest):
    """
    Predict the sentiment of a single piece of text.

    Returns:
    - **sentiment**: "positive" or "negative"
    - **confidence**: how confident the model is (0.0 to 1.0)
    - **positive_score** and **negative_score**: raw probabilities
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        start = time.time()
        result = run_inference(req.text)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"Predicted '{result.sentiment}' "
            f"({result.confidence:.2%} confidence) "
            f"in {elapsed_ms:.1f}ms"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment for multiple texts at once",
)
def predict_batch(req: BatchPredictRequest):
    """
    Predict sentiment for up to 32 texts in one request.
    More efficient than calling /predict multiple times.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()

    try:
        results = [run_inference(text) for text in req.texts]
        elapsed_ms = (time.time() - start) * 1000

        return BatchPredictResponse(
            results=results,
            total=len(results),
            processing_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Global exception handler ──────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )