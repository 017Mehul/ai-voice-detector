"""
Classification Model Layer - AI vs Human Voice Classification

This module implements the classification model interface and base implementation
for distinguishing between AI-generated and human voices using machine learning.
"""

import numpy as np
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from ..models.data_models import FeatureVector

logger = logging.getLogger(__name__)


class ClassificationModelInterface(ABC):
    """
    Abstract interface for classification models.
    
    This interface defines the contract that all classification models
    must implement for AI vs human voice detection.
    """
    
    @abstractmethod
    def predict(self, features: FeatureVector) -> Tuple[str, float]:
        """
        Predict classification and raw confidence for given features.
        
        Args:
            features: FeatureVector containing extracted audio features
            
        Returns:
            Tuple of (classification, raw_confidence) where:
            - classification: Either "AI_GENERATED" or "HUMAN"
            - raw_confidence: Raw model confidence score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def predict_batch(self, features_list: list[FeatureVector]) -> list[Tuple[str, float]]:
        """
        Predict classifications for a batch of feature vectors.
        
        Args:
            features_list: List of FeatureVector objects
            
        Returns:
            List of (classification, raw_confidence) tuples
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, features: FeatureVector) -> Dict[str, float]:
        """
        Get feature importance scores for explainability.
        
        Args:
            features: FeatureVector to analyze
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Check if the model has been trained and is ready for inference.
        
        Returns:
            bool: True if model is trained and ready
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        pass


class RandomForestVoiceClassifier(ClassificationModelInterface):
    """
    Random Forest implementation for AI vs human voice classification.
    
    This classifier uses Random Forest as the primary ML approach due to its
    interpretability, robustness, and good performance on structured features.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Initialize Random Forest classifier with configuration.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        # Initialize the sklearn model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=1  # Single-threaded runtime is more reliable in sandboxed Windows environments
        )
        
        self._is_trained = False
        self._feature_names = None
        
        logger.info(f"RandomForestVoiceClassifier initialized with {n_estimators} estimators")

    def _prepare_model_for_runtime(self) -> None:
        """Apply safe runtime settings to loaded estimators when supported."""
        if hasattr(self.model, 'n_jobs'):
            try:
                self.model.n_jobs = 1
            except Exception:
                logger.debug("Could not force estimator to single-threaded mode")

    def _feature_vector_to_model_array(self, features: FeatureVector) -> np.ndarray:
        """Convert a FeatureVector into the compact model input representation."""
        if hasattr(features, 'to_model_array'):
            return features.to_model_array()
        return features.to_array()
    
    def predict(self, features: FeatureVector) -> Tuple[str, float]:
        """
        Predict classification and raw confidence for given features.
        
        Args:
            features: FeatureVector containing extracted audio features
            
        Returns:
            Tuple of (classification, raw_confidence)
            
        Raises:
            ValueError: If model is not trained or features are invalid
        """
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")
        
        if not features.validate():
            raise ValueError("Invalid feature vector provided")
        
        try:
            # Convert features to array format
            feature_array = self._feature_vector_to_model_array(features).reshape(1, -1)
            
            # Check for feature dimension mismatch
            expected_features = getattr(self.model, 'n_features_in_', None)
            actual_features = feature_array.shape[1]
            
            if expected_features is not None and actual_features != expected_features:
                logger.error(f"Feature dimension mismatch: model expects {expected_features} features, "
                           f"but received {actual_features} features. This indicates the model was trained "
                           f"with a different feature extraction configuration.")
                raise ValueError(f"Feature dimension mismatch: model expects {expected_features} features, "
                               f"got {actual_features}. Model needs retraining with current feature format.")
            
            # Get prediction and probability
            self._prepare_model_for_runtime()
            prediction = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]
            
            # Convert numeric prediction to string format
            classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
            
            # Get confidence as maximum probability
            raw_confidence = float(np.max(probabilities))
            
            logger.debug(f"Prediction: {classification}, Confidence: {raw_confidence:.3f}")
            return classification, raw_confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
    
    def predict_batch(self, features_list: list[FeatureVector]) -> list[Tuple[str, float]]:
        """
        Predict classifications for a batch of feature vectors.
        
        Args:
            features_list: List of FeatureVector objects
            
        Returns:
            List of (classification, raw_confidence) tuples
            
        Raises:
            ValueError: If model is not trained or any features are invalid
        """
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")
        
        if not features_list:
            return []
        
        try:
            # Validate all features and convert to array format
            feature_arrays = []
            for i, features in enumerate(features_list):
                if not features.validate():
                    raise ValueError(f"Invalid feature vector at index {i}")
                feature_arrays.append(self._feature_vector_to_model_array(features))
            
            # Stack into batch array
            batch_array = np.vstack(feature_arrays)

            expected_features = getattr(self.model, 'n_features_in_', None)
            if expected_features is not None and batch_array.shape[1] != expected_features:
                raise ValueError(
                    f"Feature dimension mismatch: model expects {expected_features} features, "
                    f"got {batch_array.shape[1]}"
                )
            
            # Get batch predictions and probabilities
            self._prepare_model_for_runtime()
            predictions = self.model.predict(batch_array)
            probabilities = self.model.predict_proba(batch_array)
            
            # Convert to required format
            results = []
            for pred, probs in zip(predictions, probabilities):
                classification = "AI_GENERATED" if pred == 1 else "HUMAN"
                raw_confidence = float(np.max(probs))
                results.append((classification, raw_confidence))
            
            logger.debug(f"Batch prediction completed for {len(features_list)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise ValueError(f"Failed to make batch predictions: {str(e)}")
    
    def get_feature_importance(self, features: FeatureVector) -> Dict[str, float]:
        """
        Get feature importance scores for explainability.
        
        Args:
            features: FeatureVector to analyze (used for feature structure)
            
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained():
            raise ValueError("Model must be trained to get feature importance")
        
        try:
            # Get feature importances from the trained model
            importances = self.model.feature_importances_
            
            # Create feature names based on the feature vector structure
            feature_names = self._generate_feature_names(features)
            
            # Create importance dictionary
            importance_dict = {}
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
            
            # Sort by importance (descending)
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            logger.debug(f"Generated feature importance for {len(sorted_importance)} features")
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            raise ValueError(f"Failed to get feature importance: {str(e)}")
    
    def _generate_feature_names(self, features: FeatureVector) -> list[str]:
        """
        Generate descriptive names for features based on feature vector structure.
        
        Args:
            features: FeatureVector to analyze structure
            
        Returns:
            List of feature names
        """
        names = []
        
        # Fixed-size MFCC summary names used by the compact model representation.
        for i in range(13):
            names.extend([
                f"mfcc_{i}_mean",
                f"mfcc_{i}_std",
                f"mfcc_{i}_min",
                f"mfcc_{i}_max"
            ])
        
        # Spectral feature names (mean and std for each)
        spectral_features = ['centroid', 'bandwidth', 'rolloff', 'zero_crossing_rate']
        for feature in spectral_features:
            names.extend([f"spectral_{feature}_mean", f"spectral_{feature}_std"])
        
        # Variance feature names
        names.extend(['pitch_variance', 'energy_variance'])
        
        # Temporal feature names
        names.extend(['frame_consistency', 'rhythm_regularity', 'pause_patterns_mean'])
        
        return names
    
    def is_trained(self) -> bool:
        """
        Check if the model has been trained and is ready for inference.
        
        Returns:
            bool: True if model is trained and ready
        """
        return self._is_trained
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the Random Forest model on provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels (0 for HUMAN, 1 for AI_GENERATED)
            
        Returns:
            Dictionary containing training metrics
            
        Raises:
            ValueError: If training data is invalid
        """
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature and label counts don't match: {X.shape[0]} vs {y.shape[0]}")
        
        try:
            logger.info(f"Training Random Forest on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Train the model
            self._prepare_model_for_runtime()
            self.model.fit(X, y)
            self._prepare_model_for_runtime()
            self._is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X)
            train_probabilities = self.model.predict_proba(X)
            
            metrics = {
                'accuracy': float(accuracy_score(y, train_predictions)),
                'precision': float(precision_score(y, train_predictions, average='weighted')),
                'recall': float(recall_score(y, train_predictions, average='weighted')),
                'f1_score': float(f1_score(y, train_predictions, average='weighted'))
            }
            
            logger.info(f"Training completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self._is_trained = False
            raise ValueError(f"Failed to train model: {str(e)}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X: Test feature matrix
            y: Test target labels
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If model is not trained or test data is invalid
        """
        if not self.is_trained():
            raise ValueError("Model must be trained before evaluation")
        
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Test data cannot be empty")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature and label counts don't match: {X.shape[0]} vs {y.shape[0]}")
        
        try:
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y, predictions)),
                'precision': float(precision_score(y, predictions, average='weighted')),
                'recall': float(recall_score(y, predictions, average='weighted')),
                'f1_score': float(f1_score(y, predictions, average='weighted'))
            }
            
            logger.info(f"Evaluation completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise ValueError(f"Failed to evaluate model: {str(e)}")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation metrics
            
        Raises:
            ValueError: If data is invalid
        """
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Data cannot be empty")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature and label counts don't match: {X.shape[0]} vs {y.shape[0]}")
        
        try:
            logger.info(f"Performing {cv}-fold cross-validation on {X.shape[0]} samples")
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            
            metrics = {
                'cv_accuracy_mean': float(np.mean(cv_scores)),
                'cv_accuracy_std': float(np.std(cv_scores)),
                'cv_accuracy_min': float(np.min(cv_scores)),
                'cv_accuracy_max': float(np.max(cv_scores))
            }
            
            logger.info(f"Cross-validation completed. Mean accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise ValueError(f"Failed to perform cross-validation: {str(e)}")
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where to save the model
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained():
            raise ValueError("Cannot save untrained model")
        
        try:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save using joblib for sklearn models
            joblib.dump({
                'model': self.model,
                'is_trained': self._is_trained,
                'config': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'random_state': self.random_state
                }
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise ValueError(f"Failed to save model: {str(e)}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Raises:
            ValueError: If model file is invalid or corrupted
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
            
            # Load using joblib
            saved_data = joblib.load(model_path)

            if isinstance(saved_data, dict) and 'model' in saved_data:
                # Native package serialization format
                self.model = saved_data['model']
                self._is_trained = saved_data.get('is_trained', True)

                config = saved_data.get('config', {})
                self.n_estimators = config.get('n_estimators', self.n_estimators)
                self.max_depth = config.get('max_depth', self.max_depth)
                self.min_samples_split = config.get('min_samples_split', self.min_samples_split)
                self.min_samples_leaf = config.get('min_samples_leaf', self.min_samples_leaf)
                self.random_state = config.get('random_state', self.random_state)
            elif hasattr(saved_data, 'predict') and hasattr(saved_data, 'predict_proba'):
                # Support raw sklearn/xgboost estimators saved directly.
                self.model = saved_data
                self._is_trained = True
                self.n_estimators = getattr(saved_data, 'n_estimators', self.n_estimators)
                self.max_depth = getattr(saved_data, 'max_depth', self.max_depth)
            else:
                raise ValueError("Unsupported model file format")

            self._prepare_model_for_runtime()
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            self._is_trained = False
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration and state.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': type(self.model).__name__,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'is_trained': self._is_trained,
            'n_features': getattr(self.model, 'n_features_in_', None) if self._is_trained else None,
            'n_classes': getattr(self.model, 'n_classes_', None) if self._is_trained else None
        }


def create_default_classifier() -> RandomForestVoiceClassifier:
    """
    Create a Random Forest classifier with default configuration optimized
    for AI vs human voice classification.
    
    Returns:
        RandomForestVoiceClassifier: Configured classifier instance
    """
    return RandomForestVoiceClassifier(
        n_estimators=100,
        max_depth=None,  # Allow trees to grow deep
        min_samples_split=5,  # Require at least 5 samples to split
        min_samples_leaf=2,   # Require at least 2 samples at leaf
        random_state=42
    )
