"""
Synthetic Dataset Generator for AI Voice Detection Training

This module creates synthetic audio samples and features for training
the AI voice detection model when real datasets are not available.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json

from ..models.data_models import AudioData, FeatureVector, SpectralFeatures, TemporalFeatures

logger = logging.getLogger(__name__)


class SyntheticAudioGenerator:
    """
    Generates synthetic audio samples that mimic characteristics of
    human and AI-generated voices for training purposes.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize synthetic audio generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        logger.info(f"SyntheticAudioGenerator initialized with random_state={random_state}")
    
    def generate_human_voice_sample(self, 
                                   duration: float = 3.0,
                                   sample_rate: int = 22050) -> AudioData:
        """
        Generate synthetic audio that mimics human voice characteristics.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            AudioData: Synthetic human voice audio
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Human voice characteristics:
        # - Fundamental frequency around 100-300 Hz
        # - Natural pitch variations
        # - Harmonic structure
        # - Breathing patterns (pauses)
        
        # Base frequency with natural variation
        base_freq = np.random.uniform(120, 250)
        pitch_variation = 0.1 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
        freq = base_freq * (1 + pitch_variation)
        
        # Generate harmonic content (typical of human voice)
        signal = np.zeros(n_samples)
        for harmonic in range(1, 6):  # First 5 harmonics
            amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
            signal += amplitude * np.sin(2 * np.pi * harmonic * freq * t)
        
        # Add formant-like resonances
        formant1 = 0.3 * np.sin(2 * np.pi * 800 * t)
        formant2 = 0.2 * np.sin(2 * np.pi * 1200 * t)
        signal += formant1 + formant2
        
        # Add natural breathing patterns (brief pauses)
        breath_pattern = np.ones(n_samples)
        n_pauses = np.random.randint(1, 4)
        for _ in range(n_pauses):
            pause_start = np.random.randint(0, n_samples - int(0.2 * sample_rate))
            pause_length = np.random.randint(int(0.05 * sample_rate), int(0.15 * sample_rate))
            pause_end = min(pause_start + pause_length, n_samples)
            breath_pattern[pause_start:pause_end] *= 0.1
        
        signal *= breath_pattern
        
        # Add subtle noise (natural voice imperfections)
        noise = 0.02 * np.random.randn(n_samples)
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return AudioData(
            samples=signal.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
    
    def generate_ai_voice_sample(self, 
                                duration: float = 3.0,
                                sample_rate: int = 22050) -> AudioData:
        """
        Generate synthetic audio that mimics AI-generated voice characteristics.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            AudioData: Synthetic AI-generated voice audio
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # AI voice characteristics:
        # - More consistent pitch (less natural variation)
        # - Cleaner harmonic structure
        # - Less natural breathing patterns
        # - Subtle artifacts from synthesis
        
        # More consistent base frequency
        base_freq = np.random.uniform(140, 220)
        pitch_variation = 0.05 * np.sin(2 * np.pi * 0.3 * t) + 0.02 * np.random.randn(n_samples)
        freq = base_freq * (1 + pitch_variation)
        
        # Generate cleaner harmonic content
        signal = np.zeros(n_samples)
        for harmonic in range(1, 8):  # More harmonics, cleaner
            amplitude = 0.9 / harmonic  # Slightly different harmonic decay
            signal += amplitude * np.sin(2 * np.pi * harmonic * freq * t)
        
        # Add more precise formant-like resonances
        formant1 = 0.25 * np.sin(2 * np.pi * 850 * t)
        formant2 = 0.18 * np.sin(2 * np.pi * 1300 * t)
        formant3 = 0.12 * np.sin(2 * np.pi * 2100 * t)
        signal += formant1 + formant2 + formant3
        
        # Less natural breathing (fewer, more regular pauses)
        breath_pattern = np.ones(n_samples)
        n_pauses = np.random.randint(0, 2)
        for _ in range(n_pauses):
            pause_start = np.random.randint(0, n_samples - int(0.1 * sample_rate))
            pause_length = int(0.08 * sample_rate)  # More consistent pause length
            pause_end = min(pause_start + pause_length, n_samples)
            breath_pattern[pause_start:pause_end] *= 0.05
        
        signal *= breath_pattern
        
        # Add synthesis artifacts (subtle digital artifacts)
        artifact_freq = np.random.uniform(8000, 12000)
        artifacts = 0.005 * np.sin(2 * np.pi * artifact_freq * t)
        signal += artifacts
        
        # Add minimal noise (AI voices are typically cleaner)
        noise = 0.01 * np.random.randn(n_samples)
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return AudioData(
            samples=signal.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )


class SyntheticFeatureGenerator:
    """
    Generates synthetic feature vectors that mimic real audio features
    for training when audio processing is not available.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize synthetic feature generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        logger.info(f"SyntheticFeatureGenerator initialized with random_state={random_state}")
    
    def generate_human_features(self) -> FeatureVector:
        """
        Generate feature vector that mimics human voice characteristics.
        
        Returns:
            FeatureVector: Synthetic human voice features
        """
        # MFCC features (13 coefficients)
        # Human voices typically have more variation in MFCCs
        mfcc_features = np.random.normal(0, 1.5, 13).astype(np.float32)
        mfcc_features[0] = np.random.normal(0, 3)  # C0 (energy) varies more
        
        # Spectral features
        spectral_features = SpectralFeatures(
            centroid=np.random.normal(1500, 300, 10).astype(np.float32),  # More variation
            bandwidth=np.random.normal(800, 200, 10).astype(np.float32),
            rolloff=np.random.normal(3000, 500, 10).astype(np.float32),
            zero_crossing_rate=np.random.normal(0.08, 0.02, 10).astype(np.float32)
        )
        
        # Pitch variance (humans have more natural pitch variation)
        pitch_variance = np.random.uniform(0.15, 0.35)
        
        # Energy variance (natural breathing patterns)
        energy_variance = np.random.uniform(0.12, 0.28)
        
        # Temporal features (more irregular for humans)
        temporal_features = TemporalFeatures(
            frame_consistency=np.random.uniform(0.6, 0.8),  # Less consistent
            rhythm_regularity=np.random.uniform(0.5, 0.7),  # Less regular
            pause_patterns=np.random.normal(0.1, 0.05, 5).astype(np.float32)
        )
        
        return FeatureVector(
            mfcc_features=mfcc_features,
            spectral_features=spectral_features,
            pitch_variance=pitch_variance,
            energy_variance=energy_variance,
            temporal_features=temporal_features
        )
    
    def generate_ai_features(self) -> FeatureVector:
        """
        Generate feature vector that mimics AI-generated voice characteristics.
        
        Returns:
            FeatureVector: Synthetic AI voice features
        """
        # MFCC features (13 coefficients)
        # AI voices typically have less variation, more consistent
        mfcc_features = np.random.normal(0, 1.0, 13).astype(np.float32)
        mfcc_features[0] = np.random.normal(0, 2)  # C0 less variable
        
        # Spectral features (more consistent)
        spectral_features = SpectralFeatures(
            centroid=np.random.normal(1600, 150, 10).astype(np.float32),  # Less variation
            bandwidth=np.random.normal(750, 100, 10).astype(np.float32),
            rolloff=np.random.normal(3200, 200, 10).astype(np.float32),
            zero_crossing_rate=np.random.normal(0.06, 0.01, 10).astype(np.float32)
        )
        
        # Pitch variance (AI voices more consistent)
        pitch_variance = np.random.uniform(0.05, 0.15)
        
        # Energy variance (more consistent energy)
        energy_variance = np.random.uniform(0.04, 0.12)
        
        # Temporal features (more regular for AI)
        temporal_features = TemporalFeatures(
            frame_consistency=np.random.uniform(0.8, 0.95),  # More consistent
            rhythm_regularity=np.random.uniform(0.75, 0.9),  # More regular
            pause_patterns=np.random.normal(0.05, 0.02, 5).astype(np.float32)
        )
        
        return FeatureVector(
            mfcc_features=mfcc_features,
            spectral_features=spectral_features,
            pitch_variance=pitch_variance,
            energy_variance=energy_variance,
            temporal_features=temporal_features
        )


class SyntheticDatasetCreator:
    """
    Creates complete synthetic datasets for training the AI voice detection model.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize synthetic dataset creator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.audio_generator = SyntheticAudioGenerator(random_state)
        self.feature_generator = SyntheticFeatureGenerator(random_state)
        
        logger.info(f"SyntheticDatasetCreator initialized with random_state={random_state}")
    
    def create_balanced_dataset(self, 
                               n_samples_per_class: int = 100,
                               include_audio: bool = False) -> Tuple[List[FeatureVector], List[str]]:
        """
        Create a balanced synthetic dataset with equal samples of each class.
        
        Args:
            n_samples_per_class: Number of samples per class
            include_audio: Whether to generate audio data (slower)
            
        Returns:
            Tuple of (features_list, labels)
        """
        logger.info(f"Creating balanced synthetic dataset: {n_samples_per_class} samples per class")
        
        features_list = []
        labels = []
        
        # Generate human voice samples
        for i in range(n_samples_per_class):
            if include_audio:
                audio_data = self.audio_generator.generate_human_voice_sample()
                # In a real scenario, we would extract features from audio
                # For now, we'll use synthetic features
                features = self.feature_generator.generate_human_features()
            else:
                features = self.feature_generator.generate_human_features()
            
            features_list.append(features)
            labels.append("HUMAN")
        
        # Generate AI voice samples
        for i in range(n_samples_per_class):
            if include_audio:
                audio_data = self.audio_generator.generate_ai_voice_sample()
                # In a real scenario, we would extract features from audio
                # For now, we'll use synthetic features
                features = self.feature_generator.generate_ai_features()
            else:
                features = self.feature_generator.generate_ai_features()
            
            features_list.append(features)
            labels.append("AI_GENERATED")
        
        # Shuffle the dataset
        combined = list(zip(features_list, labels))
        np.random.shuffle(combined)
        features_list, labels = zip(*combined)
        
        logger.info(f"Created synthetic dataset: {len(features_list)} total samples")
        return list(features_list), list(labels)
    
    def create_training_splits(self, 
                              features_list: List[FeatureVector],
                              labels: List[str],
                              train_ratio: float = 0.6,
                              val_ratio: float = 0.2,
                              test_ratio: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            features_list: List of feature vectors
            labels: List of corresponding labels
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Dictionary containing split datasets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        n_samples = len(features_list)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        # Create splits
        train_features = features_list[:n_train]
        train_labels = labels[:n_train]
        
        val_features = features_list[n_train:n_train + n_val]
        val_labels = labels[n_train:n_train + n_val]
        
        test_features = features_list[n_train + n_val:]
        test_labels = labels[n_train + n_val:]
        
        splits = {
            'train': {
                'features': train_features,
                'labels': train_labels,
                'size': len(train_features)
            },
            'validation': {
                'features': val_features,
                'labels': val_labels,
                'size': len(val_features)
            },
            'test': {
                'features': test_features,
                'labels': test_labels,
                'size': len(test_features)
            }
        }
        
        logger.info(f"Created dataset splits: Train={n_train}, Val={n_val}, Test={n_test}")
        return splits
    
    def save_synthetic_dataset(self, 
                              dataset_splits: Dict[str, Dict[str, Any]],
                              output_dir: str) -> None:
        """
        Save synthetic dataset to disk.
        
        Args:
            dataset_splits: Dataset splits from create_training_splits
            output_dir: Directory to save dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset_splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Convert features to arrays
            features_arrays = [f.to_model_array() for f in split_data['features']]
            features_matrix = np.array(features_arrays)
            
            # Save features and labels
            np.save(split_dir / 'features.npy', features_matrix)
            
            with open(split_dir / 'labels.json', 'w') as f:
                json.dump(split_data['labels'], f)
            
            # Save metadata
            metadata = {
                'n_samples': split_data['size'],
                'n_features': features_matrix.shape[1],
                'class_distribution': {
                    'HUMAN': split_data['labels'].count('HUMAN'),
                    'AI_GENERATED': split_data['labels'].count('AI_GENERATED')
                }
            }
            
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save dataset info
        dataset_info = {
            'created_by': 'SyntheticDatasetCreator',
            'random_state': self.random_state,
            'total_samples': sum(split['size'] for split in dataset_splits.values()),
            'splits': {name: split['size'] for name, split in dataset_splits.items()}
        }
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Saved synthetic dataset to {output_path}")


def create_synthetic_training_dataset(n_samples_per_class: int = 200,
                                     output_dir: str = "data/synthetic_training_dataset",
                                     random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Create a complete synthetic training dataset.
    
    Args:
        n_samples_per_class: Number of samples per class
        output_dir: Directory to save dataset
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing dataset splits
    """
    creator = SyntheticDatasetCreator(random_state)
    
    # Create balanced dataset
    features_list, labels = creator.create_balanced_dataset(
        n_samples_per_class=n_samples_per_class,
        include_audio=False  # Faster generation
    )
    
    # Create splits
    dataset_splits = creator.create_training_splits(features_list, labels)
    
    # Save dataset
    creator.save_synthetic_dataset(dataset_splits, output_dir)
    
    return dataset_splits
