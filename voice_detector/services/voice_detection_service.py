"""
Voice Detection Service - High-Level Processing Pipeline

This module provides the main voice detection service that orchestrates
the complete processing pipeline from audio input to classification result.
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any

from ..utils.logging_config import get_logger
from ..utils.performance import get_performance_monitor, MemoryOptimizer
from ..models.data_models import DetectionRequest, DetectionResponse, ValidationResult, AudioData, FeatureVector
from ..audio.processor import AudioProcessor, AudioProcessingError
from ..features.extractor import FeatureExtractor
from ..models.classifier import ClassificationModelInterface
from ..models.calibration import ConfidenceCalibrator


logger = get_logger(__name__)


class VoiceDetectionService:
    """
    High-level voice detection service that orchestrates the complete processing pipeline.
    
    This service coordinates all components to provide end-to-end voice detection
    functionality from Base64 audio input to classification result with confidence.
    """
    
    def __init__(self,
                 audio_processor: AudioProcessor,
                 feature_extractor: FeatureExtractor,
                 classifier: ClassificationModelInterface,
                 confidence_calibrator: ConfidenceCalibrator):
        """
        Initialize voice detection service with all required components.
        
        Args:
            audio_processor: Audio processing component
            feature_extractor: Feature extraction component
            classifier: Classification model component
            confidence_calibrator: Confidence calibration component
        """
        self.audio_processor = audio_processor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.confidence_calibrator = confidence_calibrator
        
        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        self.memory_optimizer = MemoryOptimizer(max_memory_mb=1500.0)  # 1.5GB limit
        self.request_count = 0
        
        # Prometheus metrics
        from ..utils.prometheus_metrics import get_prometheus_metrics
        self.metrics = get_prometheus_metrics()
        
        logger.info("VoiceDetectionService initialized with all components")
    
    def detect_voice(self, request: DetectionRequest, request_id: str = "unknown") -> Tuple[DetectionResponse, Optional[ValidationResult]]:
        """
        Perform complete voice detection on audio request.
        
        This method orchestrates the entire processing pipeline:
        1. Audio processing (decode, convert, normalize, validate)
        2. Feature extraction (MFCC, spectral, temporal features)
        3. Classification (AI vs human prediction)
        4. Confidence calibration (calibrated confidence score)
        
        Args:
            request: DetectionRequest containing Base64 audio data
            request_id: Request ID for logging and tracking
            
        Returns:
            Tuple of (DetectionResponse, validation_error_if_any)
            
        Raises:
            ValueError: If request is invalid or processing fails
        """
        self.request_count += 1
        
        # Performance monitoring
        with self.performance_monitor.measure_performance(f"voice_detection_{request_id}") as metrics:
            audio_data = None
            features = None
            
            logger.info(f"Starting voice detection pipeline", extra={"request_id": request_id})
            
            try:
                # Memory optimization: Check if we need to clear caches
                if self.request_count % 50 == 0:  # Every 50 requests
                    if self.memory_optimizer.is_memory_pressure():
                        logger.info("Memory pressure detected, optimizing...")
                        self.memory_optimizer.optimize_memory_if_needed()
                        if hasattr(self.feature_extractor, 'clear_cache'):
                            self.feature_extractor.clear_cache()
                
                # Step 1: Process audio (decode, convert, normalize, validate)
                audio_data, audio_error = self._process_audio(request, request_id)
                if audio_error:
                    logger.warning(
                        f"Audio processing failed",
                        extra={
                            "request_id": request_id,
                            "error": audio_error.error_code,
                            "processing_time": round(metrics.processing_time, 4)
                        }
                    )
                    # Return dummy response with error
                    dummy_response = DetectionResponse(
                        classification="HUMAN",  # Default classification
                        confidence_score=0.0     # Zero confidence for errors
                    )
                    return dummy_response, audio_error
                
                # Step 2: Extract features
                features, feature_error = self._extract_features(audio_data, request_id)
                if feature_error:
                    logger.warning(
                        f"Feature extraction failed",
                        extra={
                            "request_id": request_id,
                            "error": feature_error.error_code,
                            "processing_time": round(metrics.processing_time, 4)
                        }
                    )
                    # Return dummy response with error
                    dummy_response = DetectionResponse(
                        classification="HUMAN",  # Default classification
                        confidence_score=0.0     # Zero confidence for errors
                    )
                    return dummy_response, feature_error
                
                # Step 3: Classify voice
                classification, raw_confidence, classification_error = self._classify_voice(features, request_id)
                if classification_error:
                    logger.warning(
                        f"Voice classification failed",
                        extra={
                            "request_id": request_id,
                            "error": classification_error.error_code,
                            "processing_time": round(metrics.processing_time, 4)
                        }
                    )
                    # Return dummy response with error
                    dummy_response = DetectionResponse(
                        classification="HUMAN",  # Default classification
                        confidence_score=0.0     # Zero confidence for errors
                    )
                    return dummy_response, classification_error
                
                # Step 4: Calibrate confidence
                calibrated_confidence = self._calibrate_confidence(raw_confidence, features, request_id)
                
                # Create final response
                response = DetectionResponse(
                    classification=classification,
                    confidence_score=calibrated_confidence
                )
                
                logger.info(
                    f"Voice detection completed successfully",
                    extra={
                        "request_id": request_id,
                        "classification": classification,
                        "raw_confidence": raw_confidence,
                        "calibrated_confidence": calibrated_confidence,
                        "processing_time": round(metrics.processing_time, 4),
                        "memory_mb": round(metrics.memory_usage_mb, 1)
                    }
                )
                
                return response, None
                
            except Exception as e:
                logger.error(
                    f"Voice detection pipeline failed with unexpected error",
                    extra={
                        "request_id": request_id,
                        "error": str(e),
                        "processing_time": round(metrics.processing_time, 4)
                    },
                    exc_info=True
                )
                
                # Return error response
                dummy_response = DetectionResponse(
                    classification="HUMAN",  # Default classification
                    confidence_score=0.0     # Zero confidence for errors
                )
                error_result = ValidationResult(
                    is_valid=False,
                    error_code="PROCESSING_ERROR",
                    error_message=f"Voice detection pipeline failed: {str(e)}"
                )
                return dummy_response, error_result
            finally:
                # Clean up memory resources to prevent leaks during sustained load
                if audio_data and hasattr(audio_data, 'samples') and audio_data.samples is not None:
                    # Clear large numpy arrays from memory
                    del audio_data.samples
                    audio_data.samples = None
                if features and hasattr(features, 'mfcc_features') and features.mfcc_features is not None:
                    # Clear feature arrays from memory
                    del features.mfcc_features
                    features.mfcc_features = None
                    if hasattr(features.spectral_features, 'centroid') and features.spectral_features.centroid is not None:
                        del features.spectral_features.centroid
                        del features.spectral_features.bandwidth
                        del features.spectral_features.rolloff
                        del features.spectral_features.zero_crossing_rate
                        features.spectral_features.centroid = None
                        features.spectral_features.bandwidth = None
                        features.spectral_features.rolloff = None
                        features.spectral_features.zero_crossing_rate = None
    
    def _process_audio(self, request: DetectionRequest, request_id: str) -> Tuple[AudioData, Optional[ValidationResult]]:
        """
        Process audio through the complete audio processing pipeline.
        
        Args:
            request: DetectionRequest containing Base64 audio
            request_id: Request ID for logging
            
        Returns:
            Tuple of (processed_audio_data, validation_error_if_any)
        """
        try:
            logger.debug(f"Processing audio", extra={"request_id": request_id})
            
            # Track audio processing metrics
            import time
            start_time = time.time()
            
            # Use the complete audio processing pipeline
            audio_data, validation_error = self.audio_processor.process_audio(request.audio_base64)
            
            # Record processing time
            processing_time = time.time() - start_time
            self.metrics.record_audio_processing("complete_pipeline", processing_time)
            
            if validation_error:
                logger.debug(
                    f"Audio processing validation failed",
                    extra={
                        "request_id": request_id,
                        "error": validation_error.error_code,
                        "message": validation_error.error_message
                    }
                )
                # Record error
                self.metrics.record_error(validation_error.error_code, "audio_processor")
                return audio_data, validation_error
            
            # Record audio properties
            if audio_data and audio_data.duration > 0:
                self.metrics.record_audio_properties(audio_data.duration, audio_data.sample_rate)
            
            logger.debug(
                f"Audio processing completed",
                extra={
                    "request_id": request_id,
                    "duration": audio_data.duration,
                    "sample_rate": audio_data.sample_rate
                }
            )
            
            return audio_data, None
            
        except AudioProcessingError as e:
            logger.error(f"Audio processing error", extra={"request_id": request_id, "error": str(e)})
            self.metrics.record_error("AudioProcessingError", "audio_processor")
            dummy_audio = AudioData(samples=[], sample_rate=22050, duration=0.0, channels=0)
            error_result = ValidationResult(
                is_valid=False,
                error_code="AUDIO_PROCESSING_ERROR",
                error_message=str(e)
            )
            return dummy_audio, error_result
        except Exception as e:
            logger.error(f"Unexpected audio processing error", extra={"request_id": request_id, "error": str(e)}, exc_info=True)
            self.metrics.record_error("UnexpectedError", "audio_processor")
            dummy_audio = AudioData(samples=[], sample_rate=22050, duration=0.0, channels=0)
            error_result = ValidationResult(
                is_valid=False,
                error_code="PROCESSING_ERROR",
                error_message=f"Audio processing failed: {str(e)}"
            )
            return dummy_audio, error_result
    
    def _extract_features(self, audio_data: AudioData, request_id: str) -> Tuple[FeatureVector, Optional[ValidationResult]]:
        """
        Extract all features from processed audio data.
        
        Args:
            audio_data: Processed audio data
            request_id: Request ID for logging
            
        Returns:
            Tuple of (feature_vector, validation_error_if_any)
        """
        try:
            logger.debug(f"Extracting features", extra={"request_id": request_id})
            
            # Track feature extraction metrics
            import time
            start_time = time.time()
            
            # Extract all features using the feature extractor
            features = self.feature_extractor.extract_all_features(audio_data)
            
            # Record feature extraction time
            processing_time = time.time() - start_time
            self.metrics.record_feature_extraction("all_features", processing_time)
            
            logger.debug(
                f"Feature extraction completed",
                extra={
                    "request_id": request_id,
                    "feature_info": features.get_feature_info()
                }
            )
            
            return features, None
            
        except ValueError as e:
            logger.error(f"Feature extraction error", extra={"request_id": request_id, "error": str(e)})
            self.metrics.record_error("ValueError", "feature_extractor")
            # Create dummy feature vector for error cases
            import numpy as np
            from ..models.data_models import SpectralFeatures, TemporalFeatures
            
            dummy_features = FeatureVector(
                mfcc_features=np.zeros((13, 1)),
                spectral_features=SpectralFeatures(
                    centroid=np.zeros(1),
                    bandwidth=np.zeros(1),
                    rolloff=np.zeros(1),
                    zero_crossing_rate=np.zeros(1)
                ),
                pitch_variance=0.0,
                energy_variance=0.0,
                temporal_features=TemporalFeatures(
                    frame_consistency=0.0,
                    rhythm_regularity=0.0,
                    pause_patterns=np.zeros(1)
                )
            )
            
            error_result = ValidationResult(
                is_valid=False,
                error_code="FEATURE_EXTRACTION_ERROR",
                error_message=str(e)
            )
            return dummy_features, error_result
        except Exception as e:
            logger.error(f"Unexpected feature extraction error", extra={"request_id": request_id, "error": str(e)}, exc_info=True)
            self.metrics.record_error("UnexpectedError", "feature_extractor")
            # Create dummy feature vector for error cases
            import numpy as np
            from ..models.data_models import SpectralFeatures, TemporalFeatures
            
            dummy_features = FeatureVector(
                mfcc_features=np.zeros((13, 1)),
                spectral_features=SpectralFeatures(
                    centroid=np.zeros(1),
                    bandwidth=np.zeros(1),
                    rolloff=np.zeros(1),
                    zero_crossing_rate=np.zeros(1)
                ),
                pitch_variance=0.0,
                energy_variance=0.0,
                temporal_features=TemporalFeatures(
                    frame_consistency=0.0,
                    rhythm_regularity=0.0,
                    pause_patterns=np.zeros(1)
                )
            )
            
            error_result = ValidationResult(
                is_valid=False,
                error_code="PROCESSING_ERROR",
                error_message=f"Feature extraction failed: {str(e)}"
            )
            return dummy_features, error_result
    
    def _classify_voice(self, features: FeatureVector, request_id: str) -> Tuple[str, float, Optional[ValidationResult]]:
        """
        Classify voice using the trained model.
        
        Args:
            features: Extracted feature vector
            request_id: Request ID for logging
            
        Returns:
            Tuple of (classification, raw_confidence, validation_error_if_any)
        """
        try:
            logger.debug(f"Classifying voice", extra={"request_id": request_id})
            
            # Track model inference metrics
            import time
            start_time = time.time()
            
            # Check if model is trained
            if not self.classifier.is_trained():
                logger.error(
                    "Classifier not trained - classification unavailable",
                    extra={"request_id": request_id}
                )
                self.metrics.record_error("SERVICE_UNAVAILABLE", "classifier")
                error_result = ValidationResult(
                    is_valid=False,
                    error_code="SERVICE_UNAVAILABLE",
                    error_message="Classification model is not trained or unavailable"
                )
                return "HUMAN", 0.0, error_result
            
            # Perform classification
            classification, raw_confidence = self.classifier.predict(features)
            
            # Record inference time and classification metrics
            inference_time = time.time() - start_time
            self.metrics.record_model_inference(inference_time)
            self.metrics.record_classification(classification, raw_confidence)
            
            logger.debug(
                f"Voice classification completed",
                extra={
                    "request_id": request_id,
                    "classification": classification,
                    "raw_confidence": raw_confidence
                }
            )
            
            return classification, raw_confidence, None
            
        except ValueError as e:
            logger.error(f"Classification error", extra={"request_id": request_id, "error": str(e)})
            self.metrics.record_error("ValueError", "classifier")
            error_result = ValidationResult(
                is_valid=False,
                error_code="CLASSIFICATION_ERROR",
                error_message=str(e)
            )
            return "HUMAN", 0.0, error_result
        except Exception as e:
            logger.error(f"Unexpected classification error", extra={"request_id": request_id, "error": str(e)}, exc_info=True)
            self.metrics.record_error("UnexpectedError", "classifier")
            error_result = ValidationResult(
                is_valid=False,
                error_code="PROCESSING_ERROR",
                error_message=f"Classification failed: {str(e)}"
            )
            return "HUMAN", 0.0, error_result
    
    def _calibrate_confidence(self, raw_confidence: float, features: FeatureVector, request_id: str) -> float:
        """
        Calibrate raw confidence score using the confidence calibrator.
        
        Args:
            raw_confidence: Raw confidence from classifier
            features: Feature vector for context-aware calibration
            request_id: Request ID for logging
            
        Returns:
            Calibrated confidence score between 0.0 and 1.0
        """
        try:
            logger.debug(f"Calibrating confidence", extra={"request_id": request_id, "raw_confidence": raw_confidence})
            
            # Calibrate confidence using the calibrator
            calibrated_confidence = self.confidence_calibrator.calibrate_confidence(raw_confidence, features)
            
            logger.debug(
                f"Confidence calibration completed",
                extra={
                    "request_id": request_id,
                    "raw_confidence": raw_confidence,
                    "calibrated_confidence": calibrated_confidence
                }
            )
            
            return calibrated_confidence
            
        except Exception as e:
            logger.warning(f"Confidence calibration failed, using raw confidence", extra={"request_id": request_id, "error": str(e)})
            # Fall back to raw confidence if calibration fails
            return max(0.0, min(1.0, raw_confidence))
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status information for all service components.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "audio_processor": {
                "type": type(self.audio_processor).__name__,
                "target_sample_rate": self.audio_processor.target_sample_rate,
                "min_duration": self.audio_processor.min_duration,
                "max_duration": self.audio_processor.max_duration,
                "optimize_for_speed": getattr(self.audio_processor, 'optimize_for_speed', False)
            },
            "feature_extractor": {
                "type": type(self.feature_extractor).__name__,
                "dimensions": self.feature_extractor.get_feature_dimensions(),
                "cache_info": self.feature_extractor.get_cache_info() if hasattr(self.feature_extractor, 'get_cache_info') else {}
            },
            "classifier": {
                "type": type(self.classifier).__name__,
                "trained": self.classifier.is_trained(),
                "info": self.classifier.get_model_info() if hasattr(self.classifier, 'get_model_info') else {}
            },
            "confidence_calibrator": {
                "type": type(self.confidence_calibrator).__name__,
                "fitted": getattr(self.confidence_calibrator, 'is_fitted', False),
                "usable": self.confidence_calibrator.is_calibrated()
            },
            "performance": {
                "request_count": self.request_count,
                "system_info": self.performance_monitor.get_system_info(),
                "performance_summary": self.performance_monitor.get_performance_summary()
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "system_info": self.performance_monitor.get_system_info(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "memory_status": {
                "current_mb": self.memory_optimizer.get_current_memory_mb(),
                "max_threshold_mb": self.memory_optimizer.max_memory_mb,
                "under_pressure": self.memory_optimizer.is_memory_pressure()
            },
            "request_count": self.request_count
        }
    
    def clear_performance_history(self):
        """Clear performance monitoring history."""
        self.performance_monitor.clear_history()
        logger.info("Performance history cleared")
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Perform performance optimization.
        
        Returns:
            Dictionary containing optimization results
        """
        optimization_results = {
            "memory_optimized": False,
            "caches_cleared": False,
            "gc_forced": False
        }
        
        # Check memory pressure
        if self.memory_optimizer.is_memory_pressure():
            self.memory_optimizer.optimize_memory_if_needed()
            optimization_results["memory_optimized"] = True
            optimization_results["gc_forced"] = True
        
        # Clear feature extractor cache
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
            optimization_results["caches_cleared"] = True
        
        logger.info(f"Performance optimization completed: {optimization_results}")
        return optimization_results
    
    def validate_system_readiness(self) -> ValidationResult:
        """
        Validate that all system components are ready for processing.
        
        Returns:
            ValidationResult indicating system readiness
        """
        try:
            # Check audio processor
            if not hasattr(self.audio_processor, 'process_audio'):
                return ValidationResult(
                    is_valid=False,
                    error_code="COMPONENT_ERROR",
                    error_message="Audio processor is not properly initialized"
                )
            
            # Check feature extractor
            if not hasattr(self.feature_extractor, 'extract_all_features'):
                return ValidationResult(
                    is_valid=False,
                    error_code="COMPONENT_ERROR",
                    error_message="Feature extractor is not properly initialized"
                )
            
            # Check classifier
            if not hasattr(self.classifier, 'predict'):
                return ValidationResult(
                    is_valid=False,
                    error_code="COMPONENT_ERROR",
                    error_message="Classifier is not properly initialized"
                )
            
            if not self.classifier.is_trained():
                return ValidationResult(
                    is_valid=False,
                    error_code="SERVICE_UNAVAILABLE",
                    error_message="Classification model is not trained or unavailable"
                )

            # Check confidence calibrator
            if not hasattr(self.confidence_calibrator, 'calibrate_confidence'):
                return ValidationResult(
                    is_valid=False,
                    error_code="COMPONENT_ERROR",
                    error_message="Confidence calibrator is not properly initialized"
                )

            if not self.confidence_calibrator.is_calibrated():
                logger.warning("Confidence calibrator is not fitted - raw confidence scores will be returned")
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"System readiness validation failed: {str(e)}", exc_info=True)
            return ValidationResult(
                is_valid=False,
                error_code="VALIDATION_ERROR",
                error_message=f"System readiness validation failed: {str(e)}"
            )
