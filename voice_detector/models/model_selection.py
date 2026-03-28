"""
Model Selection Utilities for AI Voice Detection

This module provides utilities for comparing different model architectures
and selecting the best performing model for AI vs human voice classification.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from ..models.data_models import FeatureVector
from ..models.classifier import RandomForestVoiceClassifier, ClassificationModelInterface

logger = logging.getLogger(__name__)


class ModelSelectionError(Exception):
    """Custom exception for model selection errors."""
    pass


class ModelCandidate:
    """
    Represents a model candidate for selection with its configuration and performance.
    """
    
    def __init__(self, 
                 name: str,
                 model_class: type,
                 params: Dict[str, Any],
                 description: Optional[str] = None):
        """
        Initialize model candidate.
        
        Args:
            name: Unique name for this model candidate
            model_class: Class of the model to instantiate
            params: Parameters for model initialization
            description: Optional description of the model
        """
        self.name = name
        self.model_class = model_class
        self.params = params
        self.description = description or f"{model_class.__name__} with params {params}"
        
        # Performance tracking
        self.cv_scores = None
        self.training_time = None
        self.inference_time = None
        self.model_instance = None
        self.is_trained = False
        
        logger.debug(f"Created model candidate: {self.name}")
    
    def create_model(self) -> ClassificationModelInterface:
        """
        Create an instance of the model with configured parameters.
        
        Returns:
            ClassificationModelInterface: Model instance
        """
        try:
            self.model_instance = self.model_class(**self.params)
            return self.model_instance
        except Exception as e:
            logger.error(f"Failed to create model {self.name}: {str(e)}")
            raise ModelSelectionError(f"Failed to create model {self.name}: {str(e)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about this model candidate.
        
        Returns:
            Dictionary containing model candidate summary
        """
        return {
            'name': self.name,
            'model_class': self.model_class.__name__,
            'params': self.params,
            'description': self.description,
            'is_trained': self.is_trained,
            'cv_scores': self.cv_scores,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }


class ModelSelector:
    """
    Model selection utility that compares different model architectures
    and selects the best performing one based on specified criteria.
    """
    
    def __init__(self, 
                 selection_metric: str = 'f1_weighted',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize model selector.
        
        Args:
            selection_metric: Metric to use for model selection
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.selection_metric = selection_metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Model candidates and results
        self.candidates = []
        self.selection_results = {}
        self.best_model = None
        
        # Supported metrics
        self.supported_metrics = {
            'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
            'precision_macro', 'recall_macro', 'f1_macro'
        }
        
        if selection_metric not in self.supported_metrics:
            raise ModelSelectionError(f"Unsupported selection metric: {selection_metric}")
        
        logger.info(f"ModelSelector initialized with metric={selection_metric}, cv_folds={cv_folds}")
    
    def add_candidate(self, candidate: ModelCandidate) -> None:
        """
        Add a model candidate to the selection process.
        
        Args:
            candidate: ModelCandidate to add
        """
        # Check for duplicate names
        if any(c.name == candidate.name for c in self.candidates):
            raise ModelSelectionError(f"Model candidate with name '{candidate.name}' already exists")
        
        self.candidates.append(candidate)
        logger.info(f"Added model candidate: {candidate.name}")
    
    def add_random_forest_candidates(self, param_variations: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add Random Forest model candidates with different parameter configurations.
        
        Args:
            param_variations: List of parameter dictionaries for different RF configurations
        """
        if param_variations is None:
            param_variations = [
                # Baseline configuration
                {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': self.random_state
                },
                # Conservative configuration (less overfitting)
                {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state
                },
                # Aggressive configuration (more complex)
                {
                    'n_estimators': 300,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': self.random_state
                },
                # Balanced configuration
                {
                    'n_estimators': 150,
                    'max_depth': 25,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'max_features': 'log2',
                    'random_state': self.random_state
                }
            ]
        
        for i, params in enumerate(param_variations):
            candidate = ModelCandidate(
                name=f"random_forest_{i+1}",
                model_class=RandomForestVoiceClassifier,
                params=params,
                description=f"Random Forest variant {i+1}: {params}"
            )
            self.add_candidate(candidate)
        
        logger.info(f"Added {len(param_variations)} Random Forest candidates")
    
    def evaluate_candidate(self, 
                          candidate: ModelCandidate,
                          X: np.ndarray,
                          y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a single model candidate using cross-validation.
        
        Args:
            candidate: Model candidate to evaluate
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model candidate: {candidate.name}")
        
        try:
            # Create model instance
            model = candidate.create_model()
            
            # Measure training time
            start_time = time.time()
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                model.model,  # Use the underlying sklearn model
                X, y,
                cv=cv,
                scoring=self.selection_metric,
                n_jobs=1
            )
            
            training_time = time.time() - start_time
            
            # Measure inference time (approximate)
            model.model.fit(X, y)  # Quick fit for inference timing
            start_time = time.time()
            _ = model.model.predict(X[:100])  # Predict on small sample
            inference_time = (time.time() - start_time) / 100  # Per-sample inference time
            
            # Store results in candidate
            candidate.cv_scores = cv_scores
            candidate.training_time = training_time
            candidate.inference_time = inference_time
            candidate.is_trained = True
            
            # Calculate summary statistics
            evaluation_results = {
                'candidate_name': candidate.name,
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': float(np.mean(cv_scores)),
                'std_cv_score': float(np.std(cv_scores)),
                'min_cv_score': float(np.min(cv_scores)),
                'max_cv_score': float(np.max(cv_scores)),
                'training_time': training_time,
                'inference_time_per_sample': inference_time,
                'selection_metric': self.selection_metric
            }
            
            logger.info(f"Candidate {candidate.name} - Mean {self.selection_metric}: {evaluation_results['mean_cv_score']:.4f} ± {evaluation_results['std_cv_score']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed for candidate {candidate.name}: {str(e)}")
            raise ModelSelectionError(f"Failed to evaluate candidate {candidate.name}: {str(e)}")
    
    def run_model_selection(self, 
                           X: np.ndarray,
                           y: np.ndarray,
                           include_detailed_metrics: bool = True) -> Dict[str, Any]:
        """
        Run model selection process on all candidates.
        
        Args:
            X: Feature matrix
            y: Target labels
            include_detailed_metrics: Whether to include detailed metrics for all candidates
            
        Returns:
            Dictionary containing selection results
        """
        if len(self.candidates) == 0:
            raise ModelSelectionError("No model candidates added for selection")
        
        logger.info(f"Starting model selection with {len(self.candidates)} candidates")
        
        try:
            # Evaluate all candidates
            candidate_results = []
            
            for candidate in self.candidates:
                evaluation_result = self.evaluate_candidate(candidate, X, y)
                candidate_results.append(evaluation_result)
            
            # Find best candidate based on mean CV score
            best_idx = np.argmax([result['mean_cv_score'] for result in candidate_results])
            best_candidate = self.candidates[best_idx]
            best_result = candidate_results[best_idx]
            
            self.best_model = best_candidate
            
            # Create selection summary
            selection_summary = {
                'selection_metric': self.selection_metric,
                'cv_folds': self.cv_folds,
                'n_candidates': len(self.candidates),
                'best_candidate': {
                    'name': best_candidate.name,
                    'model_class': best_candidate.model_class.__name__,
                    'params': best_candidate.params,
                    'mean_cv_score': best_result['mean_cv_score'],
                    'std_cv_score': best_result['std_cv_score'],
                    'training_time': best_result['training_time'],
                    'inference_time_per_sample': best_result['inference_time_per_sample']
                },
                'candidate_rankings': []
            }
            
            # Create rankings
            sorted_results = sorted(candidate_results, key=lambda x: x['mean_cv_score'], reverse=True)
            for rank, result in enumerate(sorted_results, 1):
                ranking_entry = {
                    'rank': rank,
                    'name': result['candidate_name'],
                    'mean_score': result['mean_cv_score'],
                    'std_score': result['std_cv_score']
                }
                selection_summary['candidate_rankings'].append(ranking_entry)
            
            # Include detailed results if requested
            if include_detailed_metrics:
                selection_summary['detailed_results'] = candidate_results
            
            self.selection_results = selection_summary
            
            logger.info(f"Model selection completed. Best model: {best_candidate.name} "
                       f"(score: {best_result['mean_cv_score']:.4f})")
            
            return selection_summary
            
        except Exception as e:
            logger.error(f"Model selection failed: {str(e)}")
            raise ModelSelectionError(f"Model selection failed: {str(e)}")
    
    def get_best_model(self) -> Optional[ModelCandidate]:
        """
        Get the best model candidate from selection results.
        
        Returns:
            ModelCandidate: Best performing model candidate, or None if selection not run
        """
        return self.best_model
    
    def train_best_model(self, X: np.ndarray, y: np.ndarray) -> ClassificationModelInterface:
        """
        Train the best model on the full dataset.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            ClassificationModelInterface: Trained best model
            
        Raises:
            ModelSelectionError: If no best model available or training fails
        """
        if self.best_model is None:
            raise ModelSelectionError("No best model available. Run model selection first.")
        
        try:
            logger.info(f"Training best model: {self.best_model.name}")
            
            # Create fresh model instance
            model = self.best_model.create_model()
            
            # Train on full dataset
            model.train(X, y)
            
            logger.info(f"Best model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to train best model: {str(e)}")
            raise ModelSelectionError(f"Failed to train best model: {str(e)}")
    
    def compare_models(self, 
                      metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare all evaluated models across multiple metrics.
        
        Args:
            metric_names: List of metrics to compare (uses selection metric if None)
            
        Returns:
            Dictionary containing model comparison results
        """
        if not self.selection_results:
            raise ModelSelectionError("No selection results available. Run model selection first.")
        
        if metric_names is None:
            metric_names = [self.selection_metric]
        
        try:
            comparison_results = {
                'metrics_compared': metric_names,
                'model_comparisons': [],
                'summary_statistics': {}
            }
            
            # For now, we only have the selection metric results
            # In a full implementation, we would re-evaluate with different metrics
            for candidate in self.candidates:
                if candidate.cv_scores is not None:
                    model_comparison = {
                        'name': candidate.name,
                        'model_class': candidate.model_class.__name__,
                        self.selection_metric: {
                            'mean': float(np.mean(candidate.cv_scores)),
                            'std': float(np.std(candidate.cv_scores)),
                            'scores': candidate.cv_scores.tolist()
                        },
                        'training_time': candidate.training_time,
                        'inference_time': candidate.inference_time
                    }
                    comparison_results['model_comparisons'].append(model_comparison)
            
            # Calculate summary statistics
            all_scores = [np.mean(c.cv_scores) for c in self.candidates if c.cv_scores is not None]
            if all_scores:
                comparison_results['summary_statistics'] = {
                    'best_score': float(np.max(all_scores)),
                    'worst_score': float(np.min(all_scores)),
                    'mean_score': float(np.mean(all_scores)),
                    'score_range': float(np.max(all_scores) - np.min(all_scores))
                }
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            raise ModelSelectionError(f"Model comparison failed: {str(e)}")
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model selection process and results.
        
        Returns:
            Dictionary containing selection summary
        """
        if not self.selection_results:
            return {'status': 'not_run', 'message': 'Model selection has not been run yet'}
        
        return {
            'status': 'completed',
            'selection_metric': self.selection_metric,
            'cv_folds': self.cv_folds,
            'n_candidates_evaluated': len(self.candidates),
            'best_model': self.selection_results['best_candidate'],
            'top_3_models': self.selection_results['candidate_rankings'][:3]
        }


def create_default_model_selector(random_state: int = 42) -> ModelSelector:
    """
    Create a model selector with default configuration and Random Forest candidates.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        ModelSelector: Configured model selector with RF candidates
    """
    selector = ModelSelector(
        selection_metric='f1_weighted',
        cv_folds=5,
        random_state=random_state
    )
    
    # Add default Random Forest candidates
    selector.add_random_forest_candidates()
    
    return selector


def quick_model_selection(X: np.ndarray, 
                         y: np.ndarray,
                         random_state: int = 42) -> Tuple[ClassificationModelInterface, Dict[str, Any]]:
    """
    Perform quick model selection with default candidates and return best trained model.
    
    Args:
        X: Feature matrix
        y: Target labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (best_trained_model, selection_results)
    """
    logger.info("Starting quick model selection")
    
    # Create selector with default candidates
    selector = create_default_model_selector(random_state)
    
    # Run selection
    selection_results = selector.run_model_selection(X, y)
    
    # Train best model
    best_model = selector.train_best_model(X, y)
    
    logger.info(f"Quick model selection completed. Best model: {selector.best_model.name}")
    
    return best_model, selection_results
