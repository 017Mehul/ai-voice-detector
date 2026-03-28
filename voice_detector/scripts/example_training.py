#!/usr/bin/env python3
"""
Example Training Script - Demonstrates how to use the training utilities

This script shows how to use the training dataset preparation, model training,
and evaluation utilities for the AI voice detection system.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from voice_detector.models.dataset_preparation import TrainingDatasetPreparator
from voice_detector.models.training import ModelTrainingPipeline
from voice_detector.models.evaluation import ModelEvaluator
from voice_detector.models.model_selection import ModelSelector
from voice_detector.utils.logging_config import setup_logging

# Setup logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def example_dataset_preparation():
    """Example of dataset preparation workflow."""
    logger.info("=== Dataset Preparation Example ===")
    
    # Initialize dataset preparator
    preparator = TrainingDatasetPreparator(random_state=42)
    
    # Example configuration for dataset preparation
    # Note: This assumes you have a dataset directory structure like:
    # data/training_dataset/
    # ├── human/
    # │   ├── sample1.wav
    # │   └── sample2.wav
    # └── ai_generated/
    #     ├── sample1.wav
    #     └── sample2.wav
    
    dataset_config = {
        'source': 'data/training_dataset',  # Replace with your dataset path
        'balance_method': 'undersample',
        'augmentation': {
            'enabled': True,
            'add_noise': {'enabled': True, 'noise_factor': 0.005},
            'time_stretch': {'enabled': True, 'stretch_factors': [0.9, 1.1]},
            'pitch_shift': {'enabled': False},  # Disabled for this example
            'volume_adjustment': {'enabled': True, 'volume_factors': [0.8, 1.2]}
        },
        'test_size': 0.2,
        'validation_size': 0.2
    }
    
    try:
        # Prepare dataset
        preparation_result = preparator.prepare_dataset(
            dataset_source=dataset_config['source'],
            balance_method=dataset_config['balance_method'],
            augmentation_config=dataset_config['augmentation'],
            extract_features=True,
            test_size=dataset_config['test_size'],
            validation_size=dataset_config['validation_size']
        )
        
        logger.info(f"Dataset preparation completed:")
        logger.info(f"  Original samples: {preparation_result['original_analysis']['total_samples']}")
        logger.info(f"  Final samples: {preparation_result['final_analysis']['total_samples']}")
        logger.info(f"  Train: {preparation_result['dataset_splits']['train_size']}")
        logger.info(f"  Validation: {preparation_result['dataset_splits']['validation_size']}")
        logger.info(f"  Test: {preparation_result['dataset_splits']['test_size']}")
        
        return preparation_result
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        logger.info("Note: Make sure you have a dataset directory with 'human' and 'ai_generated' subdirectories")
        return None


def example_model_selection(dataset):
    """Example of model selection workflow."""
    logger.info("=== Model Selection Example ===")
    
    if dataset is None:
        logger.warning("No dataset available for model selection")
        return None
    
    try:
        # Extract training data
        train_data = dataset['data']['train']
        features_list = train_data['features']
        labels = train_data['labels']
        
        # Convert to arrays for model selection
        X = np.array([f.to_model_array() for f in features_list])
        y = np.array([1 if label == "AI_GENERATED" else 0 for label in labels])
        
        # Initialize model selector
        selector = ModelSelector(
            selection_metric='f1_weighted',
            cv_folds=3,  # Reduced for example
            random_state=42
        )
        
        # Add Random Forest candidates with different configurations
        selector.add_random_forest_candidates([
            {'n_estimators': 50, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 20, 'random_state': 42},
            {'n_estimators': 150, 'max_depth': None, 'random_state': 42}
        ])
        
        # Run model selection
        selection_results = selector.run_model_selection(X, y)
        
        logger.info(f"Model selection completed:")
        logger.info(f"  Best model: {selection_results['best_candidate']['name']}")
        logger.info(f"  Best score: {selection_results['best_candidate']['mean_cv_score']:.4f}")
        
        # Train best model
        best_model = selector.train_best_model(X, y)
        
        return best_model, selector
        
    except Exception as e:
        logger.error(f"Model selection failed: {str(e)}")
        return None, None


def example_model_training(dataset):
    """Example of model training workflow."""
    logger.info("=== Model Training Example ===")
    
    if dataset is None:
        logger.warning("No dataset available for model training")
        return None
    
    try:
        # Initialize training pipeline
        training_pipeline = ModelTrainingPipeline(random_state=42)
        
        # Extract training and validation data
        train_data = dataset['data']['train']
        val_data = dataset['data']['validation']
        
        # Prepare training data
        data_info = training_pipeline.prepare_training_data(
            features_list=train_data['features'],
            labels=train_data['labels'],
            test_size=0.0,
            validation_size=0.0
        )
        
        val_info = training_pipeline.prepare_training_data(
            features_list=val_data['features'],
            labels=val_data['labels'],
            test_size=0.0,
            validation_size=0.0
        )
        
        # Train model with simple parameters
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        training_results = training_pipeline.train_model(
            X_train=data_info['X_train'],
            y_train=data_info['y_train'],
            X_val=val_info['X_train'],
            y_val=val_info['y_train'],
            model_params=model_params
        )
        
        logger.info(f"Model training completed:")
        logger.info(f"  Training accuracy: {training_results['training_metrics']['accuracy']:.4f}")
        logger.info(f"  Validation accuracy: {training_results['validation_metrics']['accuracy']:.4f}")
        
        # Train calibrator
        calibration_results = training_pipeline.train_calibrator(
            X_val=val_info['X_train'],
            y_val=val_info['y_train'],
            features_val=val_data['features']
        )
        
        logger.info(f"Confidence calibrator trained")
        
        return training_pipeline
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return None


def example_model_evaluation(training_pipeline, dataset):
    """Example of model evaluation workflow."""
    logger.info("=== Model Evaluation Example ===")
    
    if training_pipeline is None or dataset is None:
        logger.warning("No trained model or dataset available for evaluation")
        return
    
    try:
        # Extract test data
        test_data = dataset['data']['test']
        
        # Prepare test data
        test_info = training_pipeline.prepare_training_data(
            features_list=test_data['features'],
            labels=test_data['labels'],
            test_size=0.0,
            validation_size=0.0
        )
        
        # Create evaluator
        evaluator = ModelEvaluator(
            model=training_pipeline.model,
            calibrator=training_pipeline.calibrator
        )
        
        # Run evaluation
        evaluation_results = evaluator.evaluate_model(
            X_test=test_info['X_train'],
            y_test=test_info['y_train'],
            features_test=test_data['features'],
            save_plots=True,
            plots_dir='example_evaluation_plots'
        )
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  Test accuracy: {evaluation_results['overall_metrics']['accuracy']:.4f}")
        logger.info(f"  Test F1 score: {evaluation_results['overall_metrics']['f1_weighted']:.4f}")
        
        # Generate and print report
        report = evaluator.generate_evaluation_report()
        print("\n" + report)
        
        # Save results
        evaluator.save_evaluation_results('example_evaluation_results.json')
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")


def main():
    """Main example workflow."""
    logger.info("Starting AI Voice Detection Training Example")
    
    # Step 1: Dataset Preparation
    dataset = example_dataset_preparation()
    
    if dataset is None:
        logger.error("Cannot proceed without dataset. Please check your dataset path.")
        return
    
    # Step 2: Model Selection (optional)
    best_model, selector = example_model_selection(dataset)
    
    # Step 3: Model Training
    training_pipeline = example_model_training(dataset)
    
    # Step 4: Model Evaluation
    example_model_evaluation(training_pipeline, dataset)
    
    logger.info("Example workflow completed!")
    
    # Print summary
    print("\n" + "="*60)
    print("EXAMPLE TRAINING WORKFLOW COMPLETED")
    print("="*60)
    print("Files created:")
    print("  - example_evaluation_plots/ (evaluation plots)")
    print("  - example_evaluation_results.json (detailed results)")
    print("\nTo run with your own dataset:")
    print("  1. Create a dataset directory with 'human' and 'ai_generated' subdirectories")
    print("  2. Add audio files (.wav, .mp3) to each subdirectory")
    print("  3. Update the dataset path in this script")
    print("  4. Run the script")
    print("="*60)


if __name__ == '__main__':
    main()
