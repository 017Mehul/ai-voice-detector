# AI Voice Detection Model Training Guide

This guide explains how to use the training utilities to prepare datasets, train models, and evaluate performance for the AI voice detection system.

## Overview

The training system consists of several key components:

1. **Dataset Preparation** - Load, balance, and augment training data
2. **Model Training** - Train classification models with hyperparameter tuning
3. **Model Selection** - Compare different model configurations
4. **Model Evaluation** - Comprehensive performance analysis

## Quick Start

### 1. Prepare Your Dataset

Create a dataset directory with the following structure:

```
data/training_dataset/
â”œâ”€â”€ human/
â”‚   â”œâ”€â”€ sample1.wav
â”‚   â”œâ”€â”€ sample2.wav
â”‚   â””â”€â”€ ...
â””â”€â”€ ai_generated/
    â”œâ”€â”€ sample1.wav
    â”œâ”€â”€ sample2.wav
    â””â”€â”€ ...
```

Supported audio formats: WAV, MP3, FLAC, M4A, OGG

### 2. Run Training Script

```bash
python voice_detector/scripts/train_model.py \
    --dataset data/training_dataset \
    --output-dir training_results \
    --experiment-name my_experiment
```

### 3. View Results

Training results will be saved to the output directory:
- `training_results.json` - Detailed training metrics
- `model/` - Trained model files
- `evaluation_plots/` - Performance visualizations

## Detailed Usage

### Dataset Preparation

```python
from voice_detector.models.dataset_preparation import TrainingDatasetPreparator

# Initialize preparator
preparator = TrainingDatasetPreparator(random_state=42)

# Prepare dataset with augmentation
result = preparator.prepare_dataset(
    dataset_source='data/training_dataset',
    balance_method='undersample',  # or 'oversample', 'augment'
    augmentation_config={
        'enabled': True,
        'add_noise': {'enabled': True, 'noise_factor': 0.005},
        'time_stretch': {'enabled': True, 'stretch_factors': [0.9, 1.1]},
        'pitch_shift': {'enabled': True, 'n_steps': [-1.0, 1.0]},
        'volume_adjustment': {'enabled': True, 'volume_factors': [0.8, 1.2]}
    },
    extract_features=True,
    test_size=0.2,
    validation_size=0.2
)
```

### Model Training with Hyperparameter Tuning

```python
from voice_detector.models.training import ModelTrainingPipeline

# Initialize training pipeline
pipeline = ModelTrainingPipeline(random_state=42)

# Prepare training data
data_info = pipeline.prepare_training_data(
    features_list=train_features,
    labels=train_labels
)

# Hyperparameter tuning
tuning_results = pipeline.hyperparameter_tuning(
    X_train=data_info['X_train'],
    y_train=data_info['y_train'],
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
)

# Train final model with best parameters
training_results = pipeline.train_model(
    X_train=data_info['X_train'],
    y_train=data_info['y_train'],
    model_params=tuning_results['best_params']
)
```

### Model Selection

```python
from voice_detector.models.model_selection import ModelSelector

# Initialize selector
selector = ModelSelector(
    selection_metric='f1_weighted',
    cv_folds=5,
    random_state=42
)

# Add model candidates
selector.add_random_forest_candidates([
    {'n_estimators': 100, 'max_depth': 20, 'random_state': 42},
    {'n_estimators': 200, 'max_depth': 30, 'random_state': 42}
])

# Run selection
selection_results = selector.run_model_selection(X_train, y_train)

# Train best model
best_model = selector.train_best_model(X_train, y_train)
```

### Model Evaluation

```python
from voice_detector.models.evaluation import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    model=trained_model,
    calibrator=confidence_calibrator
)

# Run comprehensive evaluation
evaluation_results = evaluator.evaluate_model(
    X_test=X_test,
    y_test=y_test,
    features_test=test_features,
    language_labels=language_labels,  # Optional
    save_plots=True,
    plots_dir='evaluation_plots'
)

# Generate report
report = evaluator.generate_evaluation_report()
print(report)
```

## Configuration Options

### Dataset Configuration

```json
{
  "dataset": {
    "source": "data/training_dataset",
    "balance_method": "undersample",
    "test_size": 0.2,
    "validation_size": 0.2,
    "augmentation": {
      "enabled": true,
      "add_noise": {
        "enabled": true,
        "noise_factor": 0.005
      },
      "time_stretch": {
        "enabled": true,
        "stretch_factors": [0.9, 1.1]
      },
      "pitch_shift": {
        "enabled": true,
        "n_steps": [-1.0, 1.0]
      },
      "volume_adjustment": {
        "enabled": true,
        "volume_factors": [0.8, 1.2]
      }
    }
  }
}
```

### Training Configuration

```json
{
  "hyperparameter_tuning": {
    "enabled": true,
    "cv_folds": 5,
    "scoring": "f1_weighted",
    "param_grid": {
      "n_estimators": [50, 100, 200],
      "max_depth": [null, 10, 20, 30],
      "min_samples_split": [2, 5, 10],
      "min_samples_leaf": [1, 2, 4],
      "max_features": ["sqrt", "log2", null]
    }
  }
}
```

## Command Line Interface

### Basic Training

```bash
python voice_detector/scripts/train_model.py \
    --dataset data/training_dataset \
    --output-dir results
```

### Advanced Training

```bash
python voice_detector/scripts/train_model.py \
    --config training_config.json \
    --dataset data/training_dataset \
    --output-dir results \
    --experiment-name baseline_model \
    --balance-method undersample \
    --augment \
    --verbose
```

### Skip Hyperparameter Tuning

```bash
python voice_detector/scripts/train_model.py \
    --dataset data/training_dataset \
    --no-tuning \
    --output-dir quick_results
```

## Output Files

After training, you'll find these files in the output directory:

- `training_results.json` - Complete training metrics and history
- `training_config.json` - Configuration used for training
- `model_info.json` - Model summary and metadata
- `model/` - Directory containing:
  - `model.joblib` - Trained classification model
  - `calibrator.joblib` - Confidence calibrator
  - `metadata.json` - Model metadata
- `evaluation_plots/` - Directory containing:
  - `confusion_matrix.png` - Confusion matrix visualization
  - `roc_curve.png` - ROC curve
  - `precision_recall_curve.png` - Precision-recall curve
  - `confidence_distributions.png` - Confidence score distributions
  - `feature_importance.png` - Feature importance plot

## Performance Metrics

The system calculates comprehensive performance metrics:

### Overall Metrics
- **Accuracy** - Overall classification accuracy
- **Precision** - Weighted and per-class precision
- **Recall** - Weighted and per-class recall
- **F1 Score** - Weighted and per-class F1 scores
- **ROC AUC** - Area under ROC curve
- **Average Precision** - Area under precision-recall curve

### Language-Specific Metrics
- Performance breakdown by language (Tamil, English, Hindi, Malayalam, Telugu)
- Cross-language performance comparison
- Language bias analysis

### Confidence Calibration
- Expected Calibration Error (ECE)
- Calibration curves
- Confidence distribution analysis

## Best Practices

### Dataset Preparation
1. **Balance your dataset** - Use undersample/oversample/augment as needed
2. **Quality over quantity** - Ensure audio files are clear and properly labeled
3. **Diverse samples** - Include various speakers, recording conditions, languages
4. **Minimum duration** - Audio samples should be at least 2 seconds long

### Model Training
1. **Use cross-validation** - Always validate with k-fold CV
2. **Hyperparameter tuning** - Don't skip this step for best performance
3. **Monitor overfitting** - Check validation vs training performance
4. **Feature importance** - Analyze which features matter most

### Evaluation
1. **Hold-out test set** - Never use test data during training
2. **Multiple metrics** - Don't rely on accuracy alone
3. **Error analysis** - Examine misclassified samples
4. **Confidence calibration** - Ensure confidence scores are meaningful

## Troubleshooting

### Common Issues

**"No valid audio samples found"**
- Check audio file formats (WAV, MP3 supported)
- Ensure files are at least 2 seconds long
- Verify directory structure

**"Insufficient samples for training"**
- Need at least 10 samples per class
- Consider data augmentation
- Check class balance

**"Feature extraction failed"**
- Audio files may be corrupted
- Check sample rate compatibility
- Ensure librosa is installed

**"Model training failed"**
- Check for NaN values in features
- Verify sufficient memory
- Try smaller parameter grid

### Performance Issues

**Low accuracy**
- Try different balance methods
- Enable data augmentation
- Tune hyperparameters more extensively
- Check for label errors

**Overfitting**
- Increase min_samples_split/leaf
- Reduce max_depth
- Use more cross-validation folds

**Poor calibration**
- Ensure calibrator is trained on validation set
- Try different calibration methods
- Check confidence score distributions

## Example Workflows

See `voice_detector/scripts/example_training.py` for a complete example workflow demonstrating all components.

## API Reference

For detailed API documentation, see the docstrings in:
- `voice_detector.models.dataset_preparation`
- `voice_detector.models.training`
- `voice_detector.models.model_selection`
- `voice_detector.models.evaluation`
