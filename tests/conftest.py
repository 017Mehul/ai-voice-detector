"""
Test configuration and fixtures for AI Voice Detection System tests.

This module provides common test fixtures and configuration for all tests,
including environment setup to disable rate limiting during testing.
"""

import os
import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment configuration.
    
    This fixture runs automatically for all tests and sets environment
    variables to configure the system for testing, including disabling
    rate limiting to prevent test failures due to rapid requests.
    """
    # Disable rate limiting for tests
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    
    # Set other test-friendly configurations
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    # Ensure we're in test mode
    os.environ["TESTING"] = "true"
    
    yield
    
    # Cleanup after all tests
    test_env_vars = [
        "RATE_LIMIT_ENABLED",
        "DEBUG", 
        "LOG_LEVEL",
        "TESTING"
    ]
    
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def disable_rate_limiting():
    """
    Fixture to explicitly disable rate limiting for specific tests.
    
    Use this fixture when you need to ensure rate limiting is disabled
    for a particular test, even if the global configuration might enable it.
    """
    with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "false"}):
        yield


@pytest.fixture
def enable_rate_limiting():
    """
    Fixture to explicitly enable rate limiting for specific tests.
    
    Use this fixture when you need to test rate limiting functionality.
    """
    with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "true"}):
        yield