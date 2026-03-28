"""
Model Evaluation and Testing Utilities

This module provides comprehensive evaluation utilities for AI voice detection models,
including metrics calculation, confusion matrix analysis, performance visualization,
and language-specific performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from collections import defaultdict
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

from ..models.data_models import FeatureVector, AudioData
from ..models.classifier import ClassificationModelInterface
from ..models.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)


class ModelEvaluationError(Exception):
    """Custom exception for model evaluation errors."""
    pass


class PerformanceMetrics:
    """
    Container for model performance metrics with calculation and formatting utilities.
    """
    
    def __init__(self):
        """Initialize empty performance metrics container."""
        self.metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: Names of classes (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        if class_names is None:
            class_names = ['HUMAN', 'AI_GENERATED']
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro')
            recall_macro = recall_score(y_true, y_pred, average='macro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            
            precision_weighted = precision_score(y_true, y_pred, average='weighted')
            recall_weighted = recall_score(y_true, y_pred, average='weighted')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            self.confusion_matrix = cm
            
            # Classification report
            self.classification_report = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
            
            # Store basic metrics
            self.metrics = {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'per_class_metrics': {
                    class_names[i]: {
                        'precision': float(precision_per_class[i]),
                        'recall': float(recall_per_class[i]),
                        'f1_score': float(f1_per_class[i])
                    }
                    for i in range(len(class_names))
                },
                'confusion_matrix': cm.tolist(),
                'classification_report': self.classification_report
            }
            
            # Add probability-based metrics if available
            if y_proba is not None and len(np.unique(y_true)) == 2:
                # ROC AUC
                auc_score = roc_auc_score(y_true, y_proba[:, 1])
                self.metrics['roc_auc'] = float(auc_score)
                
                # Average Precision (PR AUC)
                avg_precision = average_precision_score(y_true, y_proba[:, 1])
                self.metrics['average_precision'] = float(avg_precision)
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba[:, 1])
                self.metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve data
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba[:, 1])
                self.metrics['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
            
            logger.info(f"Calculated performance metrics: Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise ModelEvaluationError(f"Failed to calculate metrics: {str(e)}")
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of key performance metrics.
        
        Returns:
            Dictionary with key metrics
        """
        if not self.metrics:
            return {}
        
        return {
            'accuracy': self.metrics.get('accuracy', 0.0),
            'f1_weighted': self.metrics.get('f1_weighted', 0.0),
            'precision_weighted': self.metrics.get('precision_weighted', 0.0),
            'recall_weighted': self.metrics.get('recall_weighted', 0.0),
            'roc_auc': self.metrics.get('roc_auc', 0.0)
        }
    
    def format_report(self) -> str:
        """
        Format performance metrics as a readable report.
        
        Returns:
            Formatted string report
        """
        if not self.metrics:
            return "No metrics calculated yet."
        
        report_lines = [
            "=== Model Performance Report ===",
            f"Accuracy: {self.metrics['accuracy']:.4f}",
            f"F1 Score (Weighted): {self.metrics['f1_weighted']:.4f}",
            f"Precision (Weighted): {self.metrics['precision_weighted']:.4f}",
            f"Recall (Weighted): {self.metrics['recall_weighted']:.4f}",
            ""
        ]
        
        if 'roc_auc' in self.metrics:
            report_lines.extend([
                f"ROC AUC: {self.metrics['roc_auc']:.4f}",
                f"Average Precision: {self.metrics['average_precision']:.4f}",
                ""
            ])
        
        report_lines.append("Per-Class Metrics:")
        for class_name, class_metrics in self.metrics['per_class_metrics'].items():
            report_lines.append(f"  {class_name}:")
            report_lines.append(f"    Precision: {class_metrics['precision']:.4f}")
            report_lines.append(f"    Recall: {class_metrics['recall']:.4f}")
            report_lines.append(f"    F1 Score: {class_metrics['f1_score']:.4f}")
        
        return "\n".join(report_lines)


class LanguageSpecificEvaluator:
    """
    Evaluator for analyzing model performance across different languages.
    """
    
    def __init__(self, supported_languages: Optional[List[str]] = None):
        """
        Initialize language-specific evaluator.
        
        Args:
            supported_languages: List of supported language codes
        """
        if supported_languages is None:
            supported_languages = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
        
        self.supported_languages = supported_languages
        self.language_metrics = {}
        
        logger.info(f"LanguageSpecificEvaluator initialized for languages: {supported_languages}")
    
    def evaluate_by_language(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray],
                           language_labels: List[str],
                           file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model performance separately for each language.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            language_labels: Language label for each sample
            file_paths: Optional file paths for additional analysis
            
        Returns:
            Dictionary containing language-specific performance metrics
        """
        try:
            # Group samples by language
            language_groups = defaultdict(list)
            for i, lang in enumerate(language_labels):
                language_groups[lang.lower()].append(i)
            
            # Calculate metrics for each language
            language_results = {}
            
            for language, indices in language_groups.items():
                if len(indices) < 2:  # Need at least 2 samples for meaningful metrics
                    logger.warning(f"Insufficient samples for language '{language}': {len(indices)}")
                    continue
                
                # Extract samples for this language
                lang_y_true = y_true[indices]
                lang_y_pred = y_pred[indices]
                lang_y_proba = y_proba[indices] if y_proba is not None else None
                
                # Calculate metrics
                metrics = PerformanceMetrics()
                lang_metrics = metrics.calculate_metrics(
                    lang_y_true, lang_y_pred, lang_y_proba
                )
                
                # Add language-specific information
                lang_metrics['language'] = language
                lang_metrics['n_samples'] = len(indices)
                lang_metrics['class_distribution'] = {
                    'HUMAN': int(np.sum(lang_y_true == 0)),
                    'AI_GENERATED': int(np.sum(lang_y_true == 1))
                }
                
                language_results[language] = lang_metrics
                
                logger.info(f"Language '{language}': {len(indices)} samples, "
                           f"Accuracy={lang_metrics['accuracy']:.4f}, "
                           f"F1={lang_metrics['f1_weighted']:.4f}")
            
            # Calculate overall statistics
            overall_stats = self._calculate_language_statistics(language_results)
            
            self.language_metrics = {
                'by_language': language_results,
                'overall_statistics': overall_stats,
                'supported_languages': self.supported_languages,
                'languages_evaluated': list(language_results.keys())
            }
            
            return self.language_metrics
            
        except Exception as e:
            logger.error(f"Language-specific evaluation failed: {str(e)}")
            raise ModelEvaluationError(f"Failed to evaluate by language: {str(e)}")
    
    def _calculate_language_statistics(self, language_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics across languages."""
        if not language_results:
            return {}
        
        # Collect metrics across languages
        accuracies = [result['accuracy'] for result in language_results.values()]
        f1_scores = [result['f1_weighted'] for result in language_results.values()]
        sample_counts = [result['n_samples'] for result in language_results.values()]
        
        return {
            'n_languages_evaluated': len(language_results),
            'total_samples': sum(sample_counts),
            'accuracy_stats': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'range': float(np.max(accuracies) - np.min(accuracies))
            },
            'f1_stats': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'range': float(np.max(f1_scores) - np.min(f1_scores))
            },
            'sample_distribution': dict(zip(language_results.keys(), sample_counts))
        }
    
    def compare_languages(self, metric: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Compare performance across languages for a specific metric.
        
        Args:
            metric: Metric to compare across languages
            
        Returns:
            Dictionary containing language comparison results
        """
        if not self.language_metrics:
            raise ModelEvaluationError("No language metrics available. Run evaluate_by_language first.")
        
        try:
            language_scores = {}
            for lang, metrics in self.language_metrics['by_language'].items():
                if metric in metrics:
                    language_scores[lang] = metrics[metric]
            
            if not language_scores:
                raise ModelEvaluationError(f"Metric '{metric}' not found in language results")
            
            # Sort by performance
            sorted_languages = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
            
            comparison_results = {
                'metric': metric,
                'language_rankings': [
                    {'rank': i+1, 'language': lang, 'score': score}
                    for i, (lang, score) in enumerate(sorted_languages)
                ],
                'best_language': sorted_languages[0][0],
                'worst_language': sorted_languages[-1][0],
                'performance_gap': sorted_languages[0][1] - sorted_languages[-1][1],
                'all_scores': language_scores
            }
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Language comparison failed: {str(e)}")
            raise ModelEvaluationError(f"Failed to compare languages: {str(e)}")
    
    def get_language_report(self) -> str:
        """
        Generate a formatted report of language-specific performance.
        
        Returns:
            Formatted string report
        """
        if not self.language_metrics:
            return "No language-specific metrics available."
        
        report_lines = [
            "=== Language-Specific Performance Report ===",
            f"Languages Evaluated: {len(self.language_metrics['languages_evaluated'])}",
            f"Total Samples: {self.language_metrics['overall_statistics']['total_samples']}",
            ""
        ]
        
        # Overall statistics
        acc_stats = self.language_metrics['overall_statistics']['accuracy_stats']
        f1_stats = self.language_metrics['overall_statistics']['f1_stats']
        
        report_lines.extend([
            "Overall Statistics:",
            f"  Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f} (range: {acc_stats['range']:.4f})",
            f"  F1 Score: {f1_stats['mean']:.4f} ± {f1_stats['std']:.4f} (range: {f1_stats['range']:.4f})",
            ""
        ])
        
        # Per-language results
        report_lines.append("Per-Language Results:")
        for lang, metrics in self.language_metrics['by_language'].items():
            report_lines.extend([
                f"  {lang.title()}:",
                f"    Samples: {metrics['n_samples']}",
                f"    Accuracy: {metrics['accuracy']:.4f}",
                f"    F1 Score: {metrics['f1_weighted']:.4f}",
                f"    Precision: {metrics['precision_weighted']:.4f}",
                f"    Recall: {metrics['recall_weighted']:.4f}",
                ""
            ])
        
        return "\n".join(report_lines)


class ModelEvaluator:
    """
    Comprehensive model evaluator that combines performance metrics,
    visualization, and language-specific analysis.
    """
    
    def __init__(self, 
                 model: ClassificationModelInterface,
                 calibrator: Optional[ConfidenceCalibrator] = None):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained classification model to evaluate
            calibrator: Optional confidence calibrator
        """
        self.model = model
        self.calibrator = calibrator
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.language_evaluator = LanguageSpecificEvaluator()
        
        logger.info(f"ModelEvaluator initialized for model: {type(model).__name__}")
    
    def evaluate_model(self, 
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      features_test: Optional[List[FeatureVector]] = None,
                      language_labels: Optional[List[str]] = None,
                      file_paths: Optional[List[str]] = None,
                      save_plots: bool = True,
                      plots_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            features_test: Optional test feature vectors
            language_labels: Optional language labels for each sample
            file_paths: Optional file paths for each sample
            save_plots: Whether to save evaluation plots
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        logger.info(f"Starting comprehensive model evaluation on {len(X_test)} samples")
        
        try:
            # Get model predictions
            y_pred = self.model.model.predict(X_test)
            y_proba = self.model.model.predict_proba(X_test)
            
            # Get confidence scores
            raw_confidences = np.max(y_proba, axis=1)
            calibrated_confidences = raw_confidences
            
            if self.calibrator and self.calibrator.is_fitted:
                calibrated_confidences = self.calibrator.calibrate_batch(
                    raw_confidences, features_test
                )
            
            # Calculate overall performance metrics
            metrics = PerformanceMetrics()
            overall_metrics = metrics.calculate_metrics(y_test, y_pred, y_proba)
            
            # Language-specific evaluation if language labels provided
            language_metrics = None
            if language_labels:
                language_metrics = self.language_evaluator.evaluate_by_language(
                    y_test, y_pred, y_proba, language_labels, file_paths
                )
            
            # Confidence calibration analysis
            calibration_analysis = self._analyze_confidence_calibration(
                y_test, y_proba, raw_confidences, calibrated_confidences
            )
            
            # Error analysis
            error_analysis = self._analyze_prediction_errors(
                y_test, y_pred, y_proba, file_paths
            )
            
            # Compile evaluation results
            self.evaluation_results = {
                'overall_metrics': overall_metrics,
                'language_metrics': language_metrics,
                'calibration_analysis': calibration_analysis,
                'error_analysis': error_analysis,
                'test_size': len(X_test),
                'model_info': {
                    'model_type': type(self.model).__name__,
                    'has_calibrator': self.calibrator is not None and self.calibrator.is_fitted
                }
            }
            
            # Generate plots if requested
            if save_plots:
                plots_info = self._generate_evaluation_plots(
                    y_test, y_pred, y_proba, calibrated_confidences, plots_dir
                )
                self.evaluation_results['plots'] = plots_info
            
            logger.info(f"Model evaluation completed. Overall accuracy: {overall_metrics['accuracy']:.4f}")
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelEvaluationError(f"Failed to evaluate model: {str(e)}")
    
    def _analyze_confidence_calibration(self, 
                                      y_true: np.ndarray,
                                      y_proba: np.ndarray,
                                      raw_confidences: np.ndarray,
                                      calibrated_confidences: np.ndarray) -> Dict[str, Any]:
        """Analyze confidence calibration quality."""
        try:
            # Calibration curve for raw confidences
            fraction_pos_raw, mean_pred_raw = calibration_curve(
                y_true, raw_confidences, n_bins=10
            )
            
            # Calibration curve for calibrated confidences
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_true, calibrated_confidences, n_bins=10
            )
            
            # Calculate calibration error (Expected Calibration Error)
            def calculate_ece(y_true, confidences, n_bins=10):
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = confidences[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                return ece
            
            raw_ece = calculate_ece(y_true, raw_confidences)
            calibrated_ece = calculate_ece(y_true, calibrated_confidences)
            
            return {
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
                },
                'calibration_curves': {
                    'raw': {
                        'fraction_positives': fraction_pos_raw.tolist(),
                        'mean_predicted': mean_pred_raw.tolist()
                    },
                    'calibrated': {
                        'fraction_positives': fraction_pos_cal.tolist(),
                        'mean_predicted': mean_pred_cal.tolist()
                    }
                },
                'expected_calibration_error': {
                    'raw': float(raw_ece),
                    'calibrated': float(calibrated_ece),
                    'improvement': float(raw_ece - calibrated_ece)
                }
            }
            
        except Exception as e:
            logger.warning(f"Calibration analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_prediction_errors(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: np.ndarray,
                                 file_paths: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze prediction errors and misclassifications."""
        try:
            # Find misclassified samples
            misclassified = y_true != y_pred
            n_errors = np.sum(misclassified)
            
            if n_errors == 0:
                return {'n_errors': 0, 'error_rate': 0.0, 'message': 'Perfect classification!'}
            
            # Analyze error types
            false_positives = (y_true == 0) & (y_pred == 1)  # Human classified as AI
            false_negatives = (y_true == 1) & (y_pred == 0)  # AI classified as Human
            
            # Confidence analysis for errors
            error_confidences = np.max(y_proba[misclassified], axis=1)
            correct_confidences = np.max(y_proba[~misclassified], axis=1)
            
            error_analysis = {
                'n_errors': int(n_errors),
                'error_rate': float(n_errors / len(y_true)),
                'error_types': {
                    'false_positives': int(np.sum(false_positives)),
                    'false_negatives': int(np.sum(false_negatives))
                },
                'confidence_analysis': {
                    'error_confidence_mean': float(np.mean(error_confidences)),
                    'error_confidence_std': float(np.std(error_confidences)),
                    'correct_confidence_mean': float(np.mean(correct_confidences)),
                    'correct_confidence_std': float(np.std(correct_confidences)),
                    'confidence_difference': float(np.mean(correct_confidences) - np.mean(error_confidences))
                }
            }
            
            # Add file paths for worst errors if available
            if file_paths:
                error_indices = np.where(misclassified)[0]
                error_confs = error_confidences
                
                # Sort by confidence (highest confidence errors are worst)
                worst_error_indices = error_indices[np.argsort(error_confs)[::-1]][:10]
                
                error_analysis['worst_errors'] = [
                    {
                        'file_path': file_paths[idx],
                        'true_label': 'AI_GENERATED' if y_true[idx] == 1 else 'HUMAN',
                        'predicted_label': 'AI_GENERATED' if y_pred[idx] == 1 else 'HUMAN',
                        'confidence': float(np.max(y_proba[idx]))
                    }
                    for idx in worst_error_indices
                ]
            
            return error_analysis
            
        except Exception as e:
            logger.warning(f"Error analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_evaluation_plots(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: np.ndarray,
                                 calibrated_confidences: np.ndarray,
                                 plots_dir: Optional[Union[str, Path]]) -> Dict[str, str]:
        """Generate and save evaluation plots."""
        if plots_dir is None:
            plots_dir = Path("evaluation_plots")
        else:
            plots_dir = Path(plots_dir)
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots_info = {}
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['HUMAN', 'AI_GENERATED'],
                       yticklabels=['HUMAN', 'AI_GENERATED'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = plots_dir / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_info['confusion_matrix'] = str(cm_path)
            
            # 2. ROC Curve
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                auc = roc_auc_score(y_true, y_proba[:, 1])
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                roc_path = plots_dir / 'roc_curve.png'
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_info['roc_curve'] = str(roc_path)
                
                # 3. Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_proba[:, 1])
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                pr_path = plots_dir / 'precision_recall_curve.png'
                plt.savefig(pr_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_info['precision_recall_curve'] = str(pr_path)
            
            # 4. Confidence Distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            raw_confidences = np.max(y_proba, axis=1)
            plt.hist(raw_confidences[y_true == 0], alpha=0.7, label='HUMAN', bins=20)
            plt.hist(raw_confidences[y_true == 1], alpha=0.7, label='AI_GENERATED', bins=20)
            plt.xlabel('Raw Confidence')
            plt.ylabel('Frequency')
            plt.title('Raw Confidence Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(calibrated_confidences[y_true == 0], alpha=0.7, label='HUMAN', bins=20)
            plt.hist(calibrated_confidences[y_true == 1], alpha=0.7, label='AI_GENERATED', bins=20)
            plt.xlabel('Calibrated Confidence')
            plt.ylabel('Frequency')
            plt.title('Calibrated Confidence Distribution')
            plt.legend()
            
            plt.tight_layout()
            conf_path = plots_dir / 'confidence_distributions.png'
            plt.savefig(conf_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_info['confidence_distributions'] = str(conf_path)
            
            # 5. Feature Importance (if available)
            if hasattr(self.model, 'get_feature_importance'):
                try:
                    # Create a dummy feature vector for importance analysis
                    dummy_features = FeatureVector(
                        mfcc_features=np.zeros((13, 100)),
                        spectral_features=None,  # Will be handled by the method
                        pitch_variance=0.0,
                        energy_variance=0.0,
                        temporal_features=None  # Will be handled by the method
                    )
                    
                    importance_dict = self.model.get_feature_importance(dummy_features)
                    
                    # Plot top 20 features
                    top_features = dict(list(importance_dict.items())[:20])
                    
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
                    
                except Exception as e:
                    logger.warning(f"Feature importance plot failed: {str(e)}")
            
            logger.info(f"Evaluation plots saved to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {str(e)}")
        
        return plots_info
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Formatted string report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_model first."
        
        report_lines = [
            "=" * 60,
            "COMPREHENSIVE MODEL EVALUATION REPORT",
            "=" * 60,
            ""
        ]
        
        # Overall metrics
        overall = self.evaluation_results['overall_metrics']
        report_lines.extend([
            "OVERALL PERFORMANCE:",
            f"  Test Samples: {self.evaluation_results['test_size']}",
            f"  Accuracy: {overall['accuracy']:.4f}",
            f"  F1 Score (Weighted): {overall['f1_weighted']:.4f}",
            f"  Precision (Weighted): {overall['precision_weighted']:.4f}",
            f"  Recall (Weighted): {overall['recall_weighted']:.4f}",
        ])
        
        if 'roc_auc' in overall:
            report_lines.append(f"  ROC AUC: {overall['roc_auc']:.4f}")
        
        report_lines.append("")
        
        # Per-class metrics
        report_lines.append("PER-CLASS PERFORMANCE:")
        for class_name, metrics in overall['per_class_metrics'].items():
            report_lines.extend([
                f"  {class_name}:",
                f"    Precision: {metrics['precision']:.4f}",
                f"    Recall: {metrics['recall']:.4f}",
                f"    F1 Score: {metrics['f1_score']:.4f}",
            ])
        report_lines.append("")
        
        # Language-specific results
        if self.evaluation_results['language_metrics']:
            lang_report = self.language_evaluator.get_language_report()
            report_lines.extend([lang_report, ""])
        
        # Error analysis
        if 'error_analysis' in self.evaluation_results:
            error_analysis = self.evaluation_results['error_analysis']
            if 'n_errors' in error_analysis:
                report_lines.extend([
                    "ERROR ANALYSIS:",
                    f"  Total Errors: {error_analysis['n_errors']}",
                    f"  Error Rate: {error_analysis['error_rate']:.4f}",
                    f"  False Positives: {error_analysis['error_types']['false_positives']}",
                    f"  False Negatives: {error_analysis['error_types']['false_negatives']}",
                    ""
                ])
        
        # Confidence calibration
        if 'calibration_analysis' in self.evaluation_results:
            cal_analysis = self.evaluation_results['calibration_analysis']
            if 'expected_calibration_error' in cal_analysis:
                ece = cal_analysis['expected_calibration_error']
                report_lines.extend([
                    "CONFIDENCE CALIBRATION:",
                    f"  Raw ECE: {ece['raw']:.4f}",
                    f"  Calibrated ECE: {ece['calibrated']:.4f}",
                    f"  Improvement: {ece['improvement']:.4f}",
                    ""
                ])
        
        report_lines.extend([
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
    def save_evaluation_results(self, output_path: Union[str, Path]) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save evaluation results
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}")
            raise ModelEvaluationError(f"Failed to save evaluation results: {str(e)}")


def create_model_evaluator(model: ClassificationModelInterface,
                          calibrator: Optional[ConfidenceCalibrator] = None) -> ModelEvaluator:
    """
    Create a model evaluator with the specified model and calibrator.
    
    Args:
        model: Trained classification model
        calibrator: Optional confidence calibrator
        
    Returns:
        ModelEvaluator: Configured model evaluator
    """
    return ModelEvaluator(model, calibrator)