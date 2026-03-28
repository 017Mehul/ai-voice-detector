"""Basic setup tests to verify project structure"""

import pytest
from ai_voice_detector.utils.config import get_config
from ai_voice_detector.utils.helpers import validate_base64, create_error_response


def test_package_import():
    """Test that the main package can be imported"""
    import ai_voice_detector
    assert ai_voice_detector.__version__ == "1.0.0"


def test_config_loading():
    """Test that configuration loads correctly"""
    config = get_config()
    assert config.api.port == 8000
    assert config.audio.min_duration_seconds == 2
    assert config.model.confidence_threshold == 0.5


def test_base64_validation():
    """Test Base64 validation helper"""
    # Valid Base64
    assert validate_base64("SGVsbG8gV29ybGQ=") is True
    
    # Invalid Base64
    assert validate_base64("invalid!@#") is False
    assert validate_base64("") is False


def test_error_response_creation():
    """Test error response helper"""
    response = create_error_response("TEST_ERROR", "Test message")
    
    assert response["error"] == "TEST_ERROR"
    assert response["message"] == "Test message"
    assert len(response) == 2