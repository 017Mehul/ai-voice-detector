"""Model Layer - Classification models and confidence calibration"""

from .data_models import (
    DetectionRequest,
    DetectionResponse,
    ErrorResponse,
    AudioData,
    SpectralFeatures,
    TemporalFeatures,
    FeatureVector,
    ValidationResult
)

from .classifier import (
    ClassificationModelInterface,
    RandomForestVoiceClassifier,
    create_default_classifier
)

from .calibration import (
    ConfidenceCalibrator,
    create_default_calibrator
)

from .training import (
    ModelTrainingPipeline,
    create_training_pipeline
)

__all__ = [
    'DetectionRequest',
    'DetectionResponse', 
    'ErrorResponse',
    'AudioData',
    'SpectralFeatures',
    'TemporalFeatures',
    'FeatureVector',
    'ValidationResult',
    'ClassificationModelInterface',
    'RandomForestVoiceClassifier',
    'create_default_classifier',
    'ConfidenceCalibrator',
    'create_default_calibrator',
    'ModelTrainingPipeline',
    'create_training_pipeline'
]