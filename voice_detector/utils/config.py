"""
Configuration Management for AI Voice Detection System

This module handles loading and managing configuration from environment variables,
configuration files, and default values with proper validation and type conversion.
"""

import os
import logging
import importlib.util
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
import json


logger = logging.getLogger(__name__)


def _is_module_available(module_name: str) -> bool:
    """Check whether an optional dependency is importable."""
    return importlib.util.find_spec(module_name) is not None


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str = "./models/voice_classifier.joblib"
    calibration_path: str = "./models/calibration_params.joblib"
    confidence_threshold: float = 0.5
    enable_feature_importance: bool = True
    enable_batch_processing: bool = False

    @property
    def path(self) -> str:
        """Backward-compatible alias for older code paths."""
        return self.model_path


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    max_size_mb: int = 10
    max_duration_seconds: int = 30
    min_duration_seconds: int = 2
    sample_rate: int = 22050
    n_mfcc: int = 13


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    requests: int = 100
    window: int = 60  # seconds


@dataclass
class MonitoringConfig:
    """Monitoring and health check configuration"""
    health_check_enabled: bool = True
    metrics_enabled: bool = field(default_factory=lambda: _is_module_available("prometheus_client"))
    prometheus_port: int = 9090
    detailed_logging: bool = False


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_cors: bool = True
    max_request_size: int = 50 * 1024 * 1024  # 50MB
    request_timeout: int = 30  # seconds


@dataclass
class AppConfig:
    """Main application configuration"""
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"
    
    # Sub-configurations
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}")
        
        # Validate model path exists if not in debug mode
        if not self.debug and not Path(self.model.model_path).exists():
            logger.warning(f"Model path does not exist: {self.model.model_path}")
        
        # Validate audio configuration
        if self.audio.min_duration_seconds >= self.audio.max_duration_seconds:
            raise ValueError("min_duration_seconds must be less than max_duration_seconds")
        
        if self.audio.max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        
        # Validate rate limiting
        if self.rate_limit.enabled and (self.rate_limit.requests <= 0 or self.rate_limit.window <= 0):
            raise ValueError("Rate limit requests and window must be positive when enabled")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            raise ValueError(f"Invalid API port: {self.api.port}. Must be between 1 and 65535")
        
        if self.api.workers <= 0:
            raise ValueError("API workers must be positive")


class ConfigManager:
    """Configuration manager for loading and managing application configuration"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            env_file: Optional path to .env file
        """
        self.env_file = env_file
        self._config: Optional[AppConfig] = None
        
        # Load environment variables from .env file if provided
        if env_file and Path(env_file).exists():
            self._load_env_file(env_file)
    
    def _load_env_file(self, env_file: str):
        """Load environment variables from .env file"""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            logger.info(f"Loaded environment variables from {env_file}")
        except Exception as e:
            logger.warning(f"Failed to load .env file {env_file}: {e}")
    
    def get_config(self) -> AppConfig:
        """
        Get application configuration, loading it if not already loaded
        
        Returns:
            AppConfig: Application configuration
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> AppConfig:
        """Load configuration from environment variables and defaults"""
        try:
            # Main application config
            config = AppConfig(
                debug=self._get_bool_env("DEBUG", False),
                environment=self._get_env("ENVIRONMENT", "production"),
                log_level=self._get_env("LOG_LEVEL", "INFO"),
            )
            
            # API configuration
            config.api = APIConfig(
                host=self._get_env("API_HOST", "0.0.0.0"),
                port=self._get_int_env("API_PORT", 8000),
                workers=self._get_int_env("API_WORKERS", 4),
                cors_origins=self._get_list_env("CORS_ORIGINS", ["*"]),
                cors_methods=self._get_list_env("CORS_METHODS", ["GET", "POST"]),
                cors_headers=self._get_list_env("CORS_HEADERS", ["*"]),
            )
            
            # Model configuration
            config.model = ModelConfig(
                model_path=self._get_env("MODEL_PATH", "./models/voice_classifier.joblib"),
                calibration_path=self._get_env("CALIBRATION_PATH", "./models/calibration_params.joblib"),
                confidence_threshold=self._get_float_env("MODEL_CONFIDENCE_THRESHOLD", 0.5),
                enable_feature_importance=self._get_bool_env("ENABLE_FEATURE_IMPORTANCE", True),
                enable_batch_processing=self._get_bool_env("ENABLE_BATCH_PROCESSING", False),
            )
            
            # Audio configuration
            config.audio = AudioConfig(
                max_size_mb=self._get_int_env("MAX_AUDIO_SIZE_MB", 10),
                max_duration_seconds=self._get_int_env("MAX_AUDIO_DURATION_SECONDS", 30),
                min_duration_seconds=self._get_int_env("MIN_AUDIO_DURATION_SECONDS", 2),
                sample_rate=self._get_int_env("AUDIO_SAMPLE_RATE", 22050),
                n_mfcc=self._get_int_env("AUDIO_N_MFCC", 13),
            )
            
            # Rate limiting configuration
            config.rate_limit = RateLimitConfig(
                enabled=self._get_bool_env("RATE_LIMIT_ENABLED", True),
                requests=self._get_int_env("RATE_LIMIT_REQUESTS", 100),
                window=self._get_int_env("RATE_LIMIT_WINDOW", 60),
            )
            
            # Monitoring configuration
            metrics_enabled = self._get_bool_env("METRICS_ENABLED", _is_module_available("prometheus_client"))
            if metrics_enabled and not _is_module_available("prometheus_client"):
                logger.warning("prometheus_client not installed - disabling metrics support")
                metrics_enabled = False

            config.monitoring = MonitoringConfig(
                health_check_enabled=self._get_bool_env("HEALTH_CHECK_ENABLED", True),
                metrics_enabled=metrics_enabled,
                prometheus_port=self._get_int_env("PROMETHEUS_PORT", 9090),
                detailed_logging=self._get_bool_env("ENABLE_DETAILED_LOGGING", False),
            )
            
            # Security configuration
            config.security = SecurityConfig(
                enable_cors=self._get_bool_env("ENABLE_CORS", True),
                max_request_size=self._get_int_env("MAX_REQUEST_SIZE", 50 * 1024 * 1024),
                request_timeout=self._get_int_env("REQUEST_TIMEOUT", 30),
            )
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_env(self, key: str, default: str) -> str:
        """Get string environment variable with default"""
        return os.getenv(key, default)
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable with default"""
        try:
            value = os.getenv(key)
            return int(value) if value is not None else default
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable with default"""
        try:
            value = os.getenv(key)
            return float(value) if value is not None else default
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable with default"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_list_env(self, key: str, default: List[str]) -> List[str]:
        """Get list environment variable with default (comma-separated)"""
        value = os.getenv(key)
        if value is None:
            return default
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def reload_config(self):
        """Reload configuration from environment"""
        self._config = None
        return self.get_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for serialization)"""
        config = self.get_config()
        
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        
        return dataclass_to_dict(config)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(env_file: Optional[str] = None) -> ConfigManager:
    """
    Get global configuration manager instance
    
    Args:
        env_file: Optional path to .env file (only used on first call)
        
    Returns:
        ConfigManager: Global configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(env_file)
    return _config_manager


def get_config() -> AppConfig:
    """
    Get application configuration
    
    Returns:
        AppConfig: Application configuration
    """
    return get_config_manager().get_config()


def reload_config():
    """Reload configuration from environment"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload_config()


# Convenience functions for common configuration access
def get_log_level() -> str:
    """Get configured log level"""
    return get_config().log_level


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_config().debug


def get_model_path() -> str:
    """Get model file path"""
    return get_config().model.model_path


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config().api


def get_audio_config() -> AudioConfig:
    """Get audio processing configuration"""
    return get_config().audio
