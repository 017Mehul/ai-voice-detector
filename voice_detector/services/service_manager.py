"""
Service Manager - Dependency Injection and Component Lifecycle Management

This module provides centralized service management with proper dependency
injection and configuration management for all system components.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import warnings

import joblib

from ..utils.config import get_config
from ..utils.logging_config import get_logger
from ..utils.prometheus_metrics import get_prometheus_metrics
from ..audio.processor import AudioProcessor
from ..features.extractor import FeatureExtractor
from ..models.classifier import RandomForestVoiceClassifier, create_default_classifier
from ..models.calibration import ConfidenceCalibrator
from .voice_detection_service import VoiceDetectionService


logger = get_logger(__name__)


class ServiceManager:
    """
    Centralized service manager for dependency injection and component lifecycle.
    
    This class manages the creation, configuration, and lifecycle of all system
    components, ensuring proper dependency injection and resource management.
    """
    
    def __init__(self):
        """Initialize service manager with empty service registry."""
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
        logger.info("ServiceManager initialized")
    
    def initialize(self) -> None:
        """
        Initialize all services with proper dependency injection.
        
        This method creates and configures all system components in the correct
        order, ensuring dependencies are satisfied.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.warning("ServiceManager already initialized")
            return
        
        try:
            logger.info("Initializing system services...")
            
            # Step 1: Initialize core processing components
            self._initialize_audio_processor()
            self._initialize_feature_extractor()
            self._initialize_classification_model()
            self._initialize_confidence_calibrator()
            
            # Step 2: Initialize high-level services
            self._initialize_voice_detection_service()
            
            self._initialized = True
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}", exc_info=True)
            self._cleanup_services()
            raise RuntimeError(f"Failed to initialize services: {str(e)}")
    
    def _initialize_audio_processor(self) -> None:
        """Initialize audio processing component."""
        logger.debug("Initializing AudioProcessor...")
        
        audio_processor = AudioProcessor()
        
        config = get_config()
        
        # Configure audio processor based on config
        if hasattr(config, 'audio'):
            if hasattr(config.audio, 'target_sample_rate'):
                audio_processor.target_sample_rate = config.audio.target_sample_rate
            if hasattr(config.audio, 'min_duration'):
                audio_processor.min_duration = config.audio.min_duration
            if hasattr(config.audio, 'max_duration'):
                audio_processor.max_duration = config.audio.max_duration
            if hasattr(config.audio, 'max_file_size'):
                audio_processor.max_file_size = config.audio.max_file_size
        
        self._services['audio_processor'] = audio_processor
        logger.debug("AudioProcessor initialized")
    
    def _initialize_feature_extractor(self) -> None:
        """Initialize feature extraction component."""
        logger.debug("Initializing FeatureExtractor...")
        
        # Configure feature extractor parameters with performance optimizations
        feature_config = {
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'enable_caching': True,      # Enable caching for performance
            'optimize_for_speed': True   # Enable speed optimizations
        }
        
        config = get_config()
        
        # Override with config values if available
        if hasattr(config, 'features'):
            if hasattr(config.features, 'n_mfcc'):
                feature_config['n_mfcc'] = config.features.n_mfcc
            if hasattr(config.features, 'n_fft'):
                feature_config['n_fft'] = config.features.n_fft
            if hasattr(config.features, 'hop_length'):
                feature_config['hop_length'] = config.features.hop_length
            if hasattr(config.features, 'n_mels'):
                feature_config['n_mels'] = config.features.n_mels
            if hasattr(config.features, 'enable_caching'):
                feature_config['enable_caching'] = config.features.enable_caching
            if hasattr(config.features, 'optimize_for_speed'):
                feature_config['optimize_for_speed'] = config.features.optimize_for_speed
        
        feature_extractor = FeatureExtractor(**feature_config)
        self._services['feature_extractor'] = feature_extractor
        logger.debug(f"FeatureExtractor initialized with config: {feature_config}")
    
    def _initialize_classification_model(self) -> None:
        """Initialize classification model component."""
        logger.debug("Initializing ClassificationModel...")
        
        # Create classifier with default configuration
        classifier = create_default_classifier()
        
        # Try to load pre-trained model if available
        model_path = self._get_model_path()
        if model_path and model_path.exists():
            try:
                classifier.load_model(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {str(e)}")
                logger.info("Using untrained model - training will be required")
        else:
            logger.info("No pre-trained model found - using untrained model")
        
        self._services['classifier'] = classifier
        logger.debug("ClassificationModel initialized")
    
    def _initialize_confidence_calibrator(self) -> None:
        """Initialize confidence calibration component."""
        logger.debug("Initializing ConfidenceCalibrator...")
        
        confidence_calibrator = ConfidenceCalibrator()

        loaded_path = None
        for calibration_path in self._get_calibration_candidates():
            try:
                confidence_calibrator.load_calibrator(calibration_path)
                if confidence_calibrator.has_usable_calibrator():
                    loaded_path = calibration_path
                    break
            except Exception as e:
                logger.warning(f"Failed to load calibration parameters from {calibration_path}: {str(e)}")

        if loaded_path is not None:
            logger.info(f"Loaded calibration parameters from {loaded_path}")
        else:
            logger.info("No usable calibration parameters found - raw model confidence will be returned")
        
        self._services['confidence_calibrator'] = confidence_calibrator
        logger.debug("ConfidenceCalibrator initialized")
    
    def _initialize_voice_detection_service(self) -> None:
        """Initialize high-level voice detection service."""
        logger.debug("Initializing VoiceDetectionService...")
        
        # Get all required dependencies
        audio_processor = self.get_service('audio_processor')
        feature_extractor = self.get_service('feature_extractor')
        classifier = self.get_service('classifier')
        confidence_calibrator = self.get_service('confidence_calibrator')
        
        # Create voice detection service with dependencies
        voice_detection_service = VoiceDetectionService(
            audio_processor=audio_processor,
            feature_extractor=feature_extractor,
            classifier=classifier,
            confidence_calibrator=confidence_calibrator
        )
        
        self._services['voice_detection_service'] = voice_detection_service
        logger.debug("VoiceDetectionService initialized")
    
    def _get_model_path(self) -> Optional[Path]:
        """Get path to pre-trained model file."""
        config = get_config()
        
        # Check config for model path
        if hasattr(config, 'model'):
            configured_path = getattr(config.model, 'model_path', None) or getattr(config.model, 'path', None)
            if configured_path:
                model_path = Path(configured_path)
                return model_path if model_path.exists() else None
        
        # Default model path
        default_path = Path("models/voice_classifier.joblib")
        return default_path if default_path.exists() else None
    
    def _get_calibration_path(self) -> Optional[Path]:
        """Get path to calibration parameters file."""
        config = get_config()
        
        # Check config for calibration path
        if hasattr(config, 'model') and hasattr(config.model, 'calibration_path'):
            calibration_path = Path(config.model.calibration_path)
            return calibration_path if calibration_path.exists() else None
        
        # Default calibration path
        default_path = Path("models/calibration_params.joblib")
        return default_path if default_path.exists() else None

    def _get_calibration_candidates(self) -> list[Path]:
        """Return calibration artifacts ordered by compatibility with the active classifier."""
        candidates = []

        configured_path = self._get_calibration_path()
        if configured_path is not None:
            candidates.append(configured_path)

        common_paths = [
            Path("models/calibrator.joblib"),
            Path("models/kaggle_trained/calibrator.joblib"),
            Path("models/production/calibrator.joblib"),
        ]
        candidates.extend(path for path in common_paths if path.exists())

        backup_root = Path("models/backup")
        if backup_root.exists():
            backup_candidates = sorted(backup_root.glob("*/calibrator.joblib"), reverse=True)
            candidates.extend(backup_candidates)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_candidates.append(candidate)

        unique_candidates.sort(
            key=lambda path: self._score_calibration_candidate(path),
            reverse=True
        )
        return unique_candidates

    def _score_calibration_candidate(self, calibration_path: Path) -> tuple[int, int, str]:
        """Score calibration candidates so the best-matching artifact is preferred first."""
        configured_path = self._get_calibration_path()
        is_configured = int(configured_path is not None and calibration_path.resolve() == configured_path.resolve())

        compatibility_score = 0
        classifier = self._services.get('classifier')
        classifier_model = getattr(classifier, 'model', None)
        classifier_type = type(classifier_model).__name__ if classifier_model is not None else None
        classifier_features = getattr(classifier_model, 'n_features_in_', None)

        sibling_model_paths = [
            calibration_path.with_name('model.joblib'),
            calibration_path.with_name('classifier.joblib')
        ]

        for model_path in sibling_model_paths:
            if not model_path.exists():
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifact = joblib.load(model_path)
            except Exception:
                continue

            if isinstance(artifact, dict):
                sibling_model = artifact.get('model') or artifact.get('classifier')
            else:
                sibling_model = artifact

            if sibling_model is None:
                continue

            sibling_type = type(sibling_model).__name__
            sibling_features = getattr(sibling_model, 'n_features_in_', None)

            if classifier_type and sibling_type == classifier_type:
                compatibility_score += 2
            if classifier_features is not None and sibling_features == classifier_features:
                compatibility_score += 1
            break

        # Reverse lexical order prefers newer timestamped backups once compatibility ties.
        return (compatibility_score, is_configured, str(calibration_path))
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not found
        """
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not found. Available services: {list(self._services.keys())}")
        
        return self._services[service_name]
    
    def is_initialized(self) -> bool:
        """
        Check if service manager is initialized.
        
        Returns:
            bool: True if initialized
        """
        return self._initialized
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all services.
        
        Returns:
            Dictionary containing status information for each service
        """
        status = {}
        
        for service_name, service in self._services.items():
            try:
                if service_name == 'classifier':
                    status[service_name] = {
                        'initialized': True,
                        'trained': service.is_trained(),
                        'info': service.get_model_info()
                    }
                elif service_name == 'confidence_calibrator':
                    status[service_name] = {
                        'initialized': True,
                        'calibrated': service.is_calibrated(),
                        'info': service.get_calibrator_info()
                    }
                elif service_name == 'feature_extractor':
                    status[service_name] = {
                        'initialized': True,
                        'dimensions': service.get_feature_dimensions()
                    }
                else:
                    status[service_name] = {
                        'initialized': True,
                        'type': type(service).__name__
                    }
            except Exception as e:
                status[service_name] = {
                    'initialized': False,
                    'error': str(e)
                }
        
        return status
    
    def _cleanup_services(self) -> None:
        """Clean up all services and resources."""
        logger.info("Cleaning up services...")
        
        for service_name, service in self._services.items():
            try:
                # Clear caches if available
                if hasattr(service, 'clear_cache'):
                    service.clear_cache()
                
                # Call cleanup method if available
                if hasattr(service, 'cleanup'):
                    service.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up {service_name}: {str(e)}")
        
        self._services.clear()
        self._initialized = False
        logger.info("Service cleanup completed")
    
    def clear_caches(self) -> None:
        """
        Clear caches in all services to free memory.
        
        This method should be called periodically during sustained load
        to prevent memory accumulation.
        """
        logger.info("Clearing service caches...")
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'clear_cache'):
                    service.clear_cache()
                    logger.debug(f"Cleared cache for {service_name}")
            except Exception as e:
                logger.warning(f"Error clearing cache for {service_name}: {str(e)}")
        
        logger.info("Service cache clearing completed")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information for all services.
        
        Returns:
            Dictionary containing memory usage information
        """
        memory_info = {}
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'get_cache_info'):
                    memory_info[service_name] = service.get_cache_info()
                else:
                    memory_info[service_name] = {"cache_info": "not_available"}
            except Exception as e:
                memory_info[service_name] = {"error": str(e)}
        
        return memory_info
    
    def shutdown(self) -> None:
        """
        Shutdown service manager and clean up all resources.
        """
        logger.info("Shutting down ServiceManager...")
        self._cleanup_services()
        logger.info("ServiceManager shutdown completed")


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """
    Get the global service manager instance.
    
    Returns:
        ServiceManager: Global service manager instance
    """
    global _service_manager
    
    if _service_manager is None:
        _service_manager = ServiceManager()
    
    return _service_manager


def initialize_services() -> None:
    """
    Initialize the global service manager and all services.
    
    This is a convenience function for application startup.
    
    Raises:
        RuntimeError: If initialization fails
    """
    service_manager = get_service_manager()
    service_manager.initialize()


def shutdown_services() -> None:
    """
    Shutdown the global service manager and clean up all resources.
    
    This is a convenience function for application shutdown.
    """
    global _service_manager
    
    if _service_manager is not None:
        _service_manager.shutdown()
        _service_manager = None
