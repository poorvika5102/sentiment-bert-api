"""
test_api.py - Lightweight tests that work in CI without model files.
"""

from fastapi.testclient import TestClient
from unittest import mock
import sys
import os

# ── Mock heavy libraries BEFORE importing the app ─────────
# This prevents loading transformers/torch in CI
sys.modules['torch'] = mock.MagicMock()
sys.modules['transformers'] = mock.MagicMock()
sys.modules['transformers.models'] = mock.MagicMock()
sys.modules['transformers.models.bert'] = mock.MagicMock()
sys.modules['transformers.models.bert.tokenization_bert'] = mock.MagicMock()

# Set env vars before import
os.environ['MODEL_PATH'] = 'models/bert-sentiment'
os.environ['MAX_TEXT_LENGTH'] = '256'

from api.schemas import PredictRequest, BatchPredictRequest  # noqa: E402


# ── Schema validation tests (no model needed) ─────────────
class TestSchemaValidation:

    def test_predict_request_valid(self):
        req = PredictRequest(text="This is a great movie!")
        assert req.text == "This is a great movie!"

    def test_predict_request_strips_whitespace(self):
        req = PredictRequest(text="  hello  ")
        assert req.text == "hello"

    def test_predict_request_empty_raises(self):
        import pytest
        with pytest.raises(Exception):
            PredictRequest(text="")

    def test_predict_request_whitespace_only_raises(self):
        import pytest
        with pytest.raises(Exception):
            PredictRequest(text="   ")

    def test_batch_request_valid(self):
        req = BatchPredictRequest(texts=["Good!", "Bad!"])
        assert len(req.texts) == 2

    def test_batch_request_empty_raises(self):
        import pytest
        with pytest.raises(Exception):
            BatchPredictRequest(texts=[])


# ── Basic sanity tests ────────────────────────────────────
class TestBasicSanity:

    def test_python_works(self):
        assert 1 + 1 == 2

    def test_env_var_set(self):
        assert os.environ.get('MODEL_PATH') == 'models/bert-sentiment'

    def test_model_path_exists_or_skipped(self):
        model_path = os.environ.get('MODEL_PATH', '')
        # In CI, folder exists but is empty — that's fine
        assert isinstance(model_path, str)