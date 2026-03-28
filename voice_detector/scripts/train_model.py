#!/usr/bin/env python3
"""
Model Training Script for AI Voice Detection System

This script provides a complete training pipeline with hyperparameter tuning,
cross-validation, and model selection for the AI vs human voice classifier.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from voice_detector.models.training import ModelTrainingPipeline
from voice_detector.models.dataset_preparation import TrainingDatasetPreparator
from voice_detector.models.classifier import RandomForestVoiceClassifier
from voice_detector.models.calibration import ConfidenceCalibrator
from voice_detector.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ModelTrainingOrchestrator:
    """
    Orchestrates the complete model training process including data preparation,
    hyperparameter tuning, training, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training orchestrator with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.random_state = config.get('random_state', 42)
        
        # Initialize components
        self.dataset_preparator = TrainingDatasetPreparator(random_state=self.random_state)
        self.training_pipeline = ModelTrainingPipeline(random_state=self.random_state)
        
        # Training results storage
        self.training_results = {}
        self.best_model_info = None
        
        logger.info(f"ModelTrainingOrchestrator initialized with config: {config.get('experiment_name', 'default')}")
    
    def prepare_training_data(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare training data according to configuration.
        
        Args:
            dataset_config: Dataset preparation configuration
            
        Returns:
            Dictionary containing prepared dataset information
        """
        logger.info("Starting data preparation phase")
        
        try:
            # Extract configuration parameters
            dataset_source = dataset_config['source']
            balance_method = dataset_config.get('balance_method', 'undersample')
            augmentation_config = dataset_config.get('augmentation', None)
            test_size = dataset_config.get('test_size', 0.2)
            validation_size = dataset_config.get('validation_size', 0.2)
            
            # Prepare dataset
            preparation_result = self.dataset_preparator.prepare_dataset(
                dataset_source=dataset_source,
                balance_method=balance_method,
                augmentation_config=augmentation_config,
                extract_features=True,
                test_size=test_size,
                validation_size=validation_size
            )
            
            # Store dataset information
            self.training_results['data_preparation'] = {
                'original_samples': preparation_result['original_analysis']['total_samples'],
                'final_samples': preparation_result['final_analysis']['total_samples'],
                'balance_method': balance_method,
                'augmentation_applied': preparation_result['augmentation_applied'],
                'train_size': preparation_result['dataset_splits']['train_size'],
                'validation_size': preparation_result['dataset_splits']['validation_size'],
                'test_size': preparation_result['dataset_splits']['test_size']
            }
            
            logger.info(f"Data preparation completed: {self.training_results['data_preparation']}")
            return preparation_result
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def perform_hyperparameter_tuning(self, 
                                    dataset: Dict[str, Any],
                                    tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search or random search.
        
        Args:
            dataset: Prepared dataset
            tuning_config: Hyperparameter tuning configuration
            
        Returns:
            Dictionary containing tuning results
        """
        logger.info("Starting hyperparameter tuning phase")
        
        try:
            # Extract training data
            train_data = dataset['data']['train']
            features_list = train_data['features']
            labels = train_data['labels']
            
            # Prepare data for training pipeline
            data_info = self.training_pipeline.prepare_training_data(
                features_list=features_list,
                labels=labels,
                test_size=0.0,  # We already have splits
                validation_size=0.0
            )
            
            X_train = data_info['X_train']
            y_train = data_info['y_train']
            
            # Define parameter grid
            param_grid = tuning_config.get('param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            })
            
            cv_folds = tuning_config.get('cv_folds', 5)
            scoring = tuning_config.get('scoring', 'f1_weighted')
            
            # Perform hyperparameter tuning
            tuning_results = self.training_pipeline.hyperparameter_tuning(
                X_train=X_train,
                y_train=y_train,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring
            )
            
            # Store tuning results
            self.training_results['hyperparameter_tuning'] = {
                'best_params': tuning_results['best_params'],
                'best_score': tuning_results['best_score'],
                'cv_folds': cv_folds,
                'scoring_metric': scoring,
                'param_grid_size': len(tuning_results['cv_results']['params'])
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {tuning_results['best_score']:.4f}")
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            
            return tuning_results
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise
    
    def train_final_model(self, 
                         dataset: Dict[str, Any],
                         best_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the final model with best parameters.
        
        Args:
            dataset: Prepared dataset
            best_params: Best hyperparameters from tuning (optional)
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting final model training phase")
        
        try:
            # Extract training and validation data
            train_data = dataset['data']['train']
            val_data = dataset['data']['validation']
            
            train_features = train_data['features']
            train_labels = train_data['labels']
            val_features = val_data['features']
            val_labels = val_data['labels']
            
            # Prepare training data
            train_info = self.training_pipeline.prepare_training_data(
                features_list=train_features,
                labels=train_labels,
                test_size=0.0,
                validation_size=0.0
            )
            
            val_info = self.training_pipeline.prepare_training_data(
                features_list=val_features,
                labels=val_labels,
                test_size=0.0,
                validation_size=0.0
            )
            
            X_train = train_info['X_train']
            y_train = train_info['y_train']
            X_val = val_info['X_train']  # Using 'X_train' because we set test_size=0
            y_val = val_info['y_train']
            
            # Train model with best parameters
            training_results = self.training_pipeline.train_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_params=best_params
            )
            
            # Train confidence calibrator
            calibration_results = self.training_pipeline.train_calibrator(
                X_val=X_val,
                y_val=y_val,
                features_val=val_features,
                method='platt'
            )
            
            # Store training results
            self.training_results['final_training'] = {
                'training_metrics': training_results['training_metrics'],
                'validation_metrics': training_results['validation_metrics'],
                'cv_metrics': training_results['cv_metrics'],
                'calibration_results': calibration_results,
                'model_params': best_params or {},
                'training_size': len(X_train),
                'validation_size': len(X_val)
            }
            
            logger.info(f"Final model training completed")
            logger.info(f"Training metrics: {training_results['training_metrics']}")
            logger.info(f"Validation metrics: {training_results['validation_metrics']}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Final model training failed: {str(e)}")
            raise
    
    def evaluate_final_model(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the final trained model on test data.
        
        Args:
            dataset: Prepared dataset with test split
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting final model evaluation phase")
        
        try:
            # Extract test data
            test_data = dataset['data']['test']
            test_features = test_data['features']
            test_labels = test_data['labels']
            
            # Prepare test data
            test_info = self.training_pipeline.prepare_training_data(
                features_list=test_features,
                labels=test_labels,
                test_size=0.0,
                validation_size=0.0
            )
            
            X_test = test_info['X_train']  # Using 'X_train' because we set test_size=0
            y_test = test_info['y_train']
            
            # Evaluate model
            evaluation_results = self.training_pipeline.evaluate_model(
                X_test=X_test,
                y_test=y_test,
                features_test=test_features,
                save_plots=True,
                plots_dir=self.config.get('output_dir', 'training_output') + '/evaluation_plots'
            )
            
            # Store evaluation results
            self.training_results['final_evaluation'] = evaluation_results
            
            logger.info(f"Final model evaluation completed")
            logger.info(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
            logger.info(f"Test F1 score: {evaluation_results['f1_score']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Final model evaluation failed: {str(e)}")
            raise
    
    def save_training_results(self, output_dir: str) -> None:
        """
        Save all training results and the trained model.
        
        Args:
            output_dir: Directory to save results
        """
        logger.info("Saving training results and model")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save training results
            results_file = output_path / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.training_results, f, indent=2, default=str)
            
            # Save trained model and pipeline
            model_dir = output_path / 'model'
            self.training_pipeline.save_pipeline(model_dir)
            
            # Save configuration
            config_file = output_path / 'training_config.json'
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            # Create model info summary
            model_info = {
                'training_timestamp': datetime.now().isoformat(),
                'model_type': 'RandomForestVoiceClassifier',
                'best_params': self.training_results.get('hyperparameter_tuning', {}).get('best_params', {}),
                'final_metrics': self.training_results.get('final_evaluation', {}),
                'dataset_info': self.training_results.get('data_preparation', {}),
                'model_path': str(model_dir),
                'config_path': str(config_file)
            }
            
            info_file = output_path / 'model_info.json'
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {str(e)}")
            raise
    
    def run_complete_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data preparation to final evaluation.
        
        Returns:
            Dictionary containing all training results
        """
        logger.info("Starting complete training pipeline")
        
        try:
            # Phase 1: Data Preparation
            dataset = self.prepare_training_data(self.config['dataset'])
            
            # Phase 2: Hyperparameter Tuning (if enabled)
            best_params = None
            if self.config.get('hyperparameter_tuning', {}).get('enabled', True):
                tuning_results = self.perform_hyperparameter_tuning(
                    dataset, self.config['hyperparameter_tuning']
                )
                best_params = tuning_results['best_params']
            
            # Phase 3: Final Model Training
            self.train_final_model(dataset, best_params)
            
            # Phase 4: Final Evaluation
            self.evaluate_final_model(dataset)
            
            # Phase 5: Save Results
            output_dir = self.config.get('output_dir', f'training_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            self.save_training_results(output_dir)
            
            # Add summary information
            self.training_results['summary'] = {
                'experiment_name': self.config.get('experiment_name', 'ai_voice_detection_training'),
                'total_training_time': 'completed',
                'output_directory': output_dir,
                'final_test_accuracy': self.training_results['final_evaluation']['accuracy'],
                'final_test_f1': self.training_results['final_evaluation']['f1_score'],
                'model_ready_for_deployment': True
            }
            
            logger.info("Complete training pipeline finished successfully")
            logger.info(f"Final results: Accuracy={self.training_results['final_evaluation']['accuracy']:.4f}, "
                       f"F1={self.training_results['final_evaluation']['f1_score']:.4f}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Complete training pipeline failed: {str(e)}")
            raise


def create_default_training_config() -> Dict[str, Any]:
    """
    Create default training configuration.
    
    Returns:
        Dictionary containing default training configuration
    """
    return {
        'experiment_name': 'ai_voice_detection_training',
        'random_state': 42,
        'dataset': {
            'source': 'data/training_dataset',  # Path to dataset directory or CSV
            'balance_method': 'undersample',  # 'undersample', 'oversample', or 'augment'
            'test_size': 0.2,
            'validation_size': 0.2,
            'augmentation': {
                'enabled': False,
                'add_noise': {'enabled': True, 'noise_factor': 0.005},
                'time_stretch': {'enabled': True, 'stretch_factors': [0.9, 1.1]},
                'pitch_shift': {'enabled': True, 'n_steps': [-1.0, 1.0]},
                'volume_adjustment': {'enabled': True, 'volume_factors': [0.8, 1.2]}
            }
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'cv_folds': 5,
            'scoring': 'f1_weighted',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'output_dir': None  # Will be auto-generated if None
    }


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description='Train AI Voice Detection Model')
    parser.add_argument('--config', type=str, help='Path to training configuration JSON file')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory or CSV file')
    parser.add_argument('--output-dir', type=str, help='Output directory for training results')
    parser.add_argument('--experiment-name', type=str, help='Name for this training experiment')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--balance-method', type=str, choices=['undersample', 'oversample', 'augment'],
                       default='undersample', help='Dataset balancing method')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    try:
        # Load or create configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = create_default_training_config()
        
        # Override config with command line arguments
        if args.dataset:
            config['dataset']['source'] = args.dataset
        
        if args.output_dir:
            config['output_dir'] = args.output_dir
        
        if args.experiment_name:
            config['experiment_name'] = args.experiment_name
        
        if args.no_tuning:
            config['hyperparameter_tuning']['enabled'] = False
        
        if args.balance_method:
            config['dataset']['balance_method'] = args.balance_method
        
        if args.augment:
            config['dataset']['augmentation']['enabled'] = True
        
        # Validate configuration
        if not config['dataset']['source']:
            raise ValueError("Dataset source must be specified")
        
        # Initialize and run training
        logger.info(f"Starting training with configuration: {config['experiment_name']}")
        
        orchestrator = ModelTrainingOrchestrator(config)
        results = orchestrator.run_complete_training()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment: {results['summary']['experiment_name']}")
        print(f"Final Test Accuracy: {results['summary']['final_test_accuracy']:.4f}")
        print(f"Final Test F1 Score: {results['summary']['final_test_f1']:.4f}")
        print(f"Output Directory: {results['summary']['output_directory']}")
        print(f"Model Ready for Deployment: {results['summary']['model_ready_for_deployment']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\nTRAINING FAILED: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
