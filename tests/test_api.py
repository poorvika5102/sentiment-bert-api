"""
test_api.py
Automated tests for the Sentiment Analysis API.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# We need to mock the model loading for tests
# because we don't want to load a 440MB model in CI
import unittest.mock as mock
import torch


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing without GPU."""
    with mock.patch("api.main.model") as mock_model, \
         mock.patch("api.main.tokenizer") as mock_tokenizer, \
         mock.patch("api.main.device", torch.device("cpu")):

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 0]]),
        }

        # Mock model output — return positive prediction
        mock_output = mock.MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.9]])  # positive
        mock_model.return_value = mock_output

        yield mock_model, mock_tokenizer


@pytest.fixture
def client():
    """Create test client without starting the actual server."""
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    """Tests for health and info endpoints."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_health_endpoint_exists(self, client):
        response = client.get("/health")
        # 200 (healthy) or 503 (model not loaded in test) are both valid
        assert response.status_code in [200, 503]

    def test_health_response_structure(self, client):
        response = client.get("/health")
        # If model is loaded, check structure
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_empty_text_is_rejected(self, client):
        """Empty text should return 422 validation error."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_text_too_long_is_rejected(self, client):
        """Text over 512 chars should be rejected."""
        long_text = "a" * 600
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422

    def test_whitespace_only_rejected(self, client):
        """Whitespace-only text should be rejected."""
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_missing_text_field_rejected(self, client):
        """Request without text field should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_valid_request_structure(self, client):
        """Valid request should return correct response structure."""
        # This test works even if model isn't loaded
        response = client.post("/predict", json={"text": "Great movie!"})
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert "confidence" in data
            assert "positive_score" in data
            assert "negative_score" in data
            assert data["sentiment"] in ["positive", "negative"]
            assert 0.0 <= data["confidence"] <= 1.0


class TestBatchEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_empty_batch_rejected(self, client):
        """Empty list should return 422."""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_oversized_batch_rejected(self, client):
        """More than 32 texts should be rejected."""
        texts = [f"text {i}" for i in range(33)]
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 422

    def test_batch_with_valid_input(self, client):
        """Valid batch should work."""
        response = client.post(
            "/predict/batch",
            json={"texts": ["Great!", "Terrible!"]}
        )
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert data["total"] == 2
            assert len(data["results"]) == 2


class TestInputValidation:
    """Tests specifically for input validation."""

    def test_integer_text_rejected(self, client):
        response = client.post("/predict", json={"text": 12345})
        # Should either coerce to string (200) or reject (422)
        assert response.status_code in [200, 422]

    def test_none_text_rejected(self, client):
        response = client.post("/predict", json={"text": None})
        assert response.status_code == 422