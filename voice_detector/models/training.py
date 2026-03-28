"""
Model Training and Evaluation Utilities

This module provides utilities for training, evaluating, and managing
machine learning models for AI vs human voice classification.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from ..models.data_models import FeatureVector
from ..models.classifier import RandomForestVoiceClassifier, ClassificationModelInterface
from ..models.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Complete training pipeline for AI voice detection models.
    
    This class handles data preparation, model training, hyperparameter tuning,
    evaluation, and model persistence for the voice classification system.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize training pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.calibrator = None
        self.training_history = []
        self.best_params = None
        self.feature_names = None
        
        logger.info(f"ModelTrainingPipeline initialized with random_state={random_state}")
    
    def prepare_training_data(self, 
                            features_list: List[FeatureVector],
                            labels: List[str],
                            test_size: float = 0.2,
                            validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare training data with train/validation/test splits.
        
        Args:
            features_list: List of FeatureVector objects
            labels: List of labels ("AI_GENERATED" or "HUMAN")
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Dictionary containing prepared datasets
            
        Raises:
            ValueError: If data preparation fails
        """
        if len(features_list) == 0 or len(labels) == 0:
            raise ValueError("Features and labels cannot be empty")
        
        if len(features_list) != len(labels):
            raise ValueError(f"Features and labels count mismatch: {len(features_list)} vs {len(labels)}")

        if not 0.0 <= test_size < 1.0:
            raise ValueError("test_size must be in the range [0.0, 1.0)")

        if not 0.0 <= validation_size < 1.0:
            raise ValueError("validation_size must be in the range [0.0, 1.0)")
        
        try:
            logger.info(f"Preparing training data: {len(features_list)} samples")
            
            # Validate all feature vectors
            valid_indices = []
            for i, features in enumerate(features_list):
                if features.validate():
                    valid_indices.append(i)
                else:
                    logger.warning(f"Invalid feature vector at index {i}, skipping")
            
            if len(valid_indices) == 0:
                raise ValueError("No valid feature vectors found")
            
            # Filter to valid samples only
            valid_features = [features_list[i] for i in valid_indices]
            valid_labels = [labels[i] for i in valid_indices]
            
            # Convert features to arrays
            X = np.array([features.to_model_array() for features in valid_features])
            
            # Convert labels to binary (0 for HUMAN, 1 for AI_GENERATED)
            y = np.array([1 if label == "AI_GENERATED" else 0 for label in valid_labels])
            
            # Store feature names for later use
            if len(valid_features) > 0:
                self.feature_names = self._generate_feature_names(valid_features[0])

            def get_stratify_labels(targets: np.ndarray) -> Optional[np.ndarray]:
                class_counts = np.bincount(targets, minlength=2)
                if np.count_nonzero(class_counts) < 2 or np.min(class_counts[class_counts > 0]) < 2:
                    return None
                return targets

            n_features = X.shape[1]
            X_temp = X
            y_temp = y

            if test_size > 0.0:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=get_stratify_labels(y)
                )
            else:
                X_test = np.empty((0, n_features), dtype=X.dtype)
                y_test = np.array([], dtype=y.dtype)

            if validation_size > 0.0 and len(X_temp) > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=validation_size,
                    random_state=self.random_state,
                    stratify=get_stratify_labels(y_temp)
                )
            else:
                X_train = X_temp
                y_train = y_temp
                X_val = np.empty((0, n_features), dtype=X.dtype)
                y_val = np.array([], dtype=y.dtype)

            def calculate_distribution(targets: np.ndarray) -> Dict[str, float]:
                if len(targets) == 0:
                    return {'HUMAN': 0.0, 'AI_GENERATED': 0.0}
                counts = np.bincount(targets, minlength=2) / len(targets)
                return {'HUMAN': float(counts[0]), 'AI_GENERATED': float(counts[1])}
            
            data_info = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': self.feature_names,
                'n_features': X.shape[1],
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'train_distribution': calculate_distribution(y_train),
                'val_distribution': calculate_distribution(y_val),
                'test_distribution': calculate_distribution(y_test)
            }
            
            logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            logger.info(f"Feature dimensions: {X.shape[1]}")
            
            return data_info
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise ValueError(f"Failed to prepare training data: {str(e)}")
    
    def _generate_feature_names(self, features: FeatureVector) -> List[str]:
        """Generate feature names based on feature vector structure."""
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
    
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a Random Forest model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            model_params: Optional model parameters
            
        Returns:
            Dictionary containing training results
            
        Raises:
            ValueError: If training fails
        """
        try:
            logger.info(f"Training model on {len(X_train)} samples")
            
            # Use provided parameters or defaults
            if model_params is None:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state
                }
            
            # Create and train model
            self.model = RandomForestVoiceClassifier(**model_params)
            training_metrics = self.model.train(X_train, y_train)
            
            # Evaluate on validation set if provided
            validation_metrics = {}
            if X_val is not None and y_val is not None:
                validation_metrics = self.model.evaluate(X_val, y_val)
                logger.info(f"Validation metrics: {validation_metrics}")
            
            # Perform cross-validation on training data
            cv_metrics = self.model.cross_validate(X_train, y_train, cv=5)
            
            # Store training history
            training_record = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_params': model_params,
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics,
                'cv_metrics': cv_metrics,
                'training_size': len(X_train),
                'validation_size': len(X_val) if X_val is not None else 0
            }
            
            self.training_history.append(training_record)
            
            logger.info(f"Model training completed successfully")
            return training_record
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise ValueError(f"Failed to train model: {str(e)}")
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            param_grid: Optional[Dict[str, List]] = None,
                            cv: int = 5,
                            scoring: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary containing tuning results
            
        Raises:
            ValueError: If hyperparameter tuning fails
        """
        try:
            logger.info(f"Starting hyperparameter tuning with {cv}-fold CV")
            
            # Default parameter grid if none provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            
            # Create base model for grid search
            base_model = RandomForestVoiceClassifier(random_state=self.random_state)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best parameters
            self.best_params = grid_search.best_params_
            
            # Train model with best parameters
            best_model_params = {**self.best_params, 'random_state': self.random_state}
            self.model = RandomForestVoiceClassifier(**best_model_params)
            training_metrics = self.model.train(X_train, y_train)
            
            tuning_results = {
                'best_params': self.best_params,
                'best_score': float(grid_search.best_score_),
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'training_metrics': training_metrics
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.3f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            return tuning_results
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise ValueError(f"Failed to perform hyperparameter tuning: {str(e)}")
    
    def train_calibrator(self, 
                        X_val: np.ndarray, 
                        y_val: np.ndarray,
                        features_val: Optional[List[FeatureVector]] = None,
                        method: str = 'platt') -> Dict[str, Any]:
        """
        Train confidence calibrator on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            features_val: Optional validation feature vectors for quality assessment
            method: Calibration method ('platt' or 'isotonic')
            
        Returns:
            Dictionary containing calibration results
            
        Raises:
            ValueError: If calibrator training fails
        """
        if self.model is None or not self.model.is_trained():
            raise ValueError("Model must be trained before calibrating confidence")
        
        try:
            logger.info(f"Training confidence calibrator with method: {method}")
            
            # Get raw confidence scores from model
            raw_confidences = []
            for i in range(len(X_val)):
                features_array = X_val[i:i+1]
                probabilities = self.model.model.predict_proba(features_array)[0]
                raw_confidence = float(np.max(probabilities))
                raw_confidences.append(raw_confidence)
            
            raw_confidences = np.array(raw_confidences)
            
            # Create and fit calibrator
            self.calibrator = ConfidenceCalibrator(method=method)
            self.calibrator.fit(raw_confidences, y_val, features_val)
            
            # Evaluate calibration quality
            calibrated_confidences = self.calibrator.calibrate_batch(raw_confidences, features_val)
            
            calibration_results = {
                'method': method,
                'n_samples': len(X_val),
                'raw_confidence_stats': {
                    'mean': float(np.mean(raw_confidences)),
                    'std': float(np.std(raw_confidences)),
                    'min': float(np.min(raw_confidences)),
                    'max': float(np.max(raw_confidences))
                },
                'calibrated_confidence_stats': {
                    'mean': float(np.mean(calibrated_confidences)),
                    'std': float(np.std(calibrated_confidences)),
                    'min': float(np.min(calibrated_confidences)),
                    'max': float(np.max(calibrated_confidences))
                }
            }
            
            logger.info(f"Confidence calibrator trained successfully")
            return calibration_results
            
        except Exception as e:
            logger.error(f"Calibrator training failed: {str(e)}")
            raise ValueError(f"Failed to train calibrator: {str(e)}")
    
    def evaluate_model(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      features_test: Optional[List[FeatureVector]] = None,
                      save_plots: bool = True,
                      plots_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            features_test: Optional test feature vectors
            save_plots: Whether to save evaluation plots
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary containing comprehensive evaluation results
            
        Raises:
            ValueError: If evaluation fails
        """
        if self.model is None or not self.model.is_trained():
            raise ValueError("Model must be trained before evaluation")
        
        try:
            logger.info(f"Evaluating model on {len(X_test)} test samples")
            
            # Get predictions and probabilities
            predictions = self.model.model.predict(X_test)
            probabilities = self.model.model.predict_proba(X_test)
            
            # Get raw confidences
            raw_confidences = np.max(probabilities, axis=1)
            
            # Get calibrated confidences if calibrator is available
            calibrated_confidences = raw_confidences
            if self.calibrator is not None and self.calibrator.is_fitted:
                calibrated_confidences = self.calibrator.calibrate_batch(raw_confidences, features_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Calculate AUC if we have both classes
            auc_score = None
            if len(np.unique(y_test)) == 2:
                auc_score = roc_auc_score(y_test, probabilities[:, 1])
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            # Classification report
            class_report = classification_report(y_test, predictions, 
                                               target_names=['HUMAN', 'AI_GENERATED'],
                                               output_dict=True)
            
            # Feature importance
            feature_importance = {}
            if self.feature_names and hasattr(self.model.model, 'feature_importances_'):
                importances = self.model.model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True))
            
            evaluation_results = {
                'test_size': len(X_test),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc_score) if auc_score is not None else None,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'confidence_stats': {
                    'raw_mean': float(np.mean(raw_confidences)),
                    'raw_std': float(np.std(raw_confidences)),
                    'calibrated_mean': float(np.mean(calibrated_confidences)),
                    'calibrated_std': float(np.std(calibrated_confidences))
                }
            }
            
            # Generate and save plots if requested
            if save_plots:
                plots_info = self._generate_evaluation_plots(
                    y_test, predictions, probabilities, cm, 
                    feature_importance, plots_dir
                )
                evaluation_results['plots'] = plots_info
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise ValueError(f"Failed to evaluate model: {str(e)}")
    
    def _generate_evaluation_plots(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 probabilities: np.ndarray,
                                 confusion_matrix: np.ndarray,
                                 feature_importance: Dict[str, float],
                                 plots_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
        """Generate and save evaluation plots."""
        if plots_dir is None:
            plots_dir = Path("evaluation_plots")
        else:
            plots_dir = Path(plots_dir)
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plots_info = {}
        
        try:
            # Confusion Matrix Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['HUMAN', 'AI_GENERATED'],
                       yticklabels=['HUMAN', 'AI_GENERATED'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = plots_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_info['confusion_matrix'] = str(cm_path)
            
            # ROC Curve (if binary classification)
            if len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
                auc = roc_auc_score(y_true, probabilities[:, 1])
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                roc_path = plots_dir / 'roc_curve.png'
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_info['roc_curve'] = str(roc_path)
            
            # Feature Importance Plot (top 20 features)
            if feature_importance:
                top_features = dict(list(feature_importance.items())[:20])
                
                plt.figure(figsize=(10, 8))
                features = list(top_features.keys())
                importances = list(top_features.values())
                
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importances')
                plt.gca().invert_yaxis()
                
                importance_path = plots_dir / 'feature_importance.png'
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_info['feature_importance'] = str(importance_path)
            
            logger.info(f"Evaluation plots saved to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {str(e)}")
        
        return plots_info
    
    def save_pipeline(self, pipeline_dir: Union[str, Path]) -> None:
        """
        Save the complete training pipeline (model, calibrator, history).
        
        Args:
            pipeline_dir: Directory to save pipeline components
            
        Raises:
            ValueError: If saving fails
        """
        try:
            pipeline_dir = Path(pipeline_dir)
            pipeline_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if self.model is not None and self.model.is_trained():
                model_path = pipeline_dir / 'model.joblib'
                self.model.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Save calibrator
            if self.calibrator is not None and self.calibrator.is_fitted:
                calibrator_path = pipeline_dir / 'calibrator.joblib'
                self.calibrator.save_calibrator(calibrator_path)
                logger.info(f"Calibrator saved to {calibrator_path}")
            
            # Save training history and metadata
            metadata = {
                'random_state': self.random_state,
                'training_history': self.training_history,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'pipeline_version': '1.0'
            }
            
            metadata_path = pipeline_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training pipeline saved to {pipeline_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline saving failed: {str(e)}")
            raise ValueError(f"Failed to save pipeline: {str(e)}")
    
    def load_pipeline(self, pipeline_dir: Union[str, Path]) -> None:
        """
        Load a complete training pipeline from disk.
        
        Args:
            pipeline_dir: Directory containing pipeline components
            
        Raises:
            ValueError: If loading fails
        """
        try:
            pipeline_dir = Path(pipeline_dir)
            if not pipeline_dir.exists():
                raise ValueError(f"Pipeline directory not found: {pipeline_dir}")
            
            # Load metadata
            metadata_path = pipeline_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.random_state = metadata.get('random_state', 42)
                self.training_history = metadata.get('training_history', [])
                self.best_params = metadata.get('best_params')
                self.feature_names = metadata.get('feature_names')
            
            # Load model
            model_path = pipeline_dir / 'model.joblib'
            if model_path.exists():
                self.model = RandomForestVoiceClassifier()
                self.model.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            
            # Load calibrator
            calibrator_path = pipeline_dir / 'calibrator.joblib'
            if calibrator_path.exists():
                self.calibrator = ConfidenceCalibrator()
                self.calibrator.load_calibrator(calibrator_path)
                logger.info(f"Calibrator loaded from {calibrator_path}")
            
            logger.info(f"Training pipeline loaded from {pipeline_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline loading failed: {str(e)}")
            raise ValueError(f"Failed to load pipeline: {str(e)}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline state.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            'random_state': self.random_state,
            'model_trained': self.model is not None and self.model.is_trained(),
            'calibrator_fitted': self.calibrator is not None and self.calibrator.is_fitted,
            'training_runs': len(self.training_history),
            'best_params': self.best_params,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'model_info': self.model.get_model_info() if self.model else None,
            'calibrator_info': self.calibrator.get_calibrator_info() if self.calibrator else None
        }


def create_training_pipeline(random_state: int = 42) -> ModelTrainingPipeline:
    """
    Create a model training pipeline with default configuration.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        ModelTrainingPipeline: Configured training pipeline
    """
    return ModelTrainingPipeline(random_state=random_state)
