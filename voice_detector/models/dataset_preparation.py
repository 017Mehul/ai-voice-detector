"""
Dataset Preparation Utilities for AI Voice Detection Training

This module provides utilities for loading, preprocessing, and augmenting
training datasets for the AI vs human voice classification model.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import random
from collections import Counter
import librosa
from sklearn.model_selection import train_test_split
import soundfile as sf

from ..models.data_models import AudioData, FeatureVector
from ..audio.processor import AudioProcessor
from ..features.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DatasetPreparationError(Exception):
    """Custom exception for dataset preparation errors."""
    pass


class AudioDataAugmentor:
    """
    Audio data augmentation utilities for robust model training.
    
    Provides various augmentation techniques to increase dataset diversity
    and improve model generalization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize audio data augmentor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        logger.info(f"AudioDataAugmentor initialized with random_state={random_state}")
    
    def add_noise(self, audio_data: AudioData, noise_factor: float = 0.005) -> AudioData:
        """
        Add Gaussian noise to audio signal.
        
        Args:
            audio_data: Original audio data
            noise_factor: Noise intensity factor (0.0 to 0.1 recommended)
            
        Returns:
            AudioData: Augmented audio with added noise
        """
        try:
            noise = np.random.normal(0, noise_factor, audio_data.samples.shape)
            augmented_samples = audio_data.samples + noise
            
            # Ensure samples stay within valid range
            augmented_samples = np.clip(augmented_samples, -1.0, 1.0)
            
            return AudioData(
                samples=augmented_samples,
                sample_rate=audio_data.sample_rate,
                duration=audio_data.duration,
                channels=audio_data.channels
            )
            
        except Exception as e:
            logger.error(f"Noise augmentation failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to add noise: {str(e)}")
    
    def time_stretch(self, audio_data: AudioData, stretch_factor: float = 1.1) -> AudioData:
        """
        Apply time stretching to audio without changing pitch.
        
        Args:
            audio_data: Original audio data
            stretch_factor: Time stretch factor (1.0 = no change, >1.0 = slower, <1.0 = faster)
            
        Returns:
            AudioData: Time-stretched audio
        """
        try:
            stretched_samples = librosa.effects.time_stretch(
                audio_data.samples, 
                rate=stretch_factor
            )
            
            new_duration = len(stretched_samples) / audio_data.sample_rate
            
            return AudioData(
                samples=stretched_samples,
                sample_rate=audio_data.sample_rate,
                duration=new_duration,
                channels=audio_data.channels
            )
            
        except Exception as e:
            logger.error(f"Time stretch augmentation failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to time stretch: {str(e)}")
    
    def pitch_shift(self, audio_data: AudioData, n_steps: float = 2.0) -> AudioData:
        """
        Shift pitch of audio signal.
        
        Args:
            audio_data: Original audio data
            n_steps: Number of semitones to shift (positive = higher, negative = lower)
            
        Returns:
            AudioData: Pitch-shifted audio
        """
        try:
            shifted_samples = librosa.effects.pitch_shift(
                audio_data.samples,
                sr=audio_data.sample_rate,
                n_steps=n_steps
            )
            
            return AudioData(
                samples=shifted_samples,
                sample_rate=audio_data.sample_rate,
                duration=audio_data.duration,
                channels=audio_data.channels
            )
            
        except Exception as e:
            logger.error(f"Pitch shift augmentation failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to pitch shift: {str(e)}")
    
    def volume_adjustment(self, audio_data: AudioData, volume_factor: float = 1.2) -> AudioData:
        """
        Adjust volume of audio signal.
        
        Args:
            audio_data: Original audio data
            volume_factor: Volume multiplication factor (1.0 = no change)
            
        Returns:
            AudioData: Volume-adjusted audio
        """
        try:
            adjusted_samples = audio_data.samples * volume_factor
            
            # Ensure samples stay within valid range
            adjusted_samples = np.clip(adjusted_samples, -1.0, 1.0)
            
            return AudioData(
                samples=adjusted_samples,
                sample_rate=audio_data.sample_rate,
                duration=audio_data.duration,
                channels=audio_data.channels
            )
            
        except Exception as e:
            logger.error(f"Volume adjustment failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to adjust volume: {str(e)}")
    
    def augment_audio_sample(self, 
                           audio_data: AudioData,
                           augmentation_config: Optional[Dict[str, Any]] = None) -> List[AudioData]:
        """
        Apply multiple augmentations to a single audio sample.
        
        Args:
            audio_data: Original audio data
            augmentation_config: Configuration for augmentations
            
        Returns:
            List[AudioData]: List of augmented audio samples (including original)
        """
        if augmentation_config is None:
            augmentation_config = {
                'add_noise': {'enabled': True, 'noise_factor': 0.005},
                'time_stretch': {'enabled': True, 'stretch_factors': [0.9, 1.1]},
                'pitch_shift': {'enabled': True, 'n_steps': [-1.0, 1.0]},
                'volume_adjustment': {'enabled': True, 'volume_factors': [0.8, 1.2]}
            }
        
        augmented_samples = [audio_data]  # Include original
        
        try:
            # Add noise augmentation
            if augmentation_config.get('add_noise', {}).get('enabled', False):
                noise_factor = augmentation_config['add_noise'].get('noise_factor', 0.005)
                noisy_audio = self.add_noise(audio_data, noise_factor)
                augmented_samples.append(noisy_audio)
            
            # Time stretch augmentations
            if augmentation_config.get('time_stretch', {}).get('enabled', False):
                stretch_factors = augmentation_config['time_stretch'].get('stretch_factors', [1.1])
                for factor in stretch_factors:
                    stretched_audio = self.time_stretch(audio_data, factor)
                    augmented_samples.append(stretched_audio)
            
            # Pitch shift augmentations
            if augmentation_config.get('pitch_shift', {}).get('enabled', False):
                n_steps_list = augmentation_config['pitch_shift'].get('n_steps', [1.0])
                for n_steps in n_steps_list:
                    shifted_audio = self.pitch_shift(audio_data, n_steps)
                    augmented_samples.append(shifted_audio)
            
            # Volume adjustment augmentations
            if augmentation_config.get('volume_adjustment', {}).get('enabled', False):
                volume_factors = augmentation_config['volume_adjustment'].get('volume_factors', [1.2])
                for factor in volume_factors:
                    adjusted_audio = self.volume_adjustment(audio_data, factor)
                    augmented_samples.append(adjusted_audio)
            
            logger.debug(f"Generated {len(augmented_samples)} augmented samples from 1 original")
            return augmented_samples
            
        except Exception as e:
            logger.error(f"Audio augmentation failed: {str(e)}")
            # Return at least the original sample
            return [audio_data]


class DatasetLoader:
    """
    Dataset loading utilities for training data preparation.
    
    Handles loading audio files from various sources and formats,
    with support for different dataset structures and metadata.
    """
    
    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize dataset loader.
        
        Args:
            audio_processor: AudioProcessor instance for audio processing
        """
        self.audio_processor = audio_processor or AudioProcessor()
        
        logger.info("DatasetLoader initialized")
    
    def load_from_directory(self, 
                          dataset_dir: Union[str, Path],
                          label_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[AudioData], List[str], List[str]]:
        """
        Load audio files from directory structure.
        
        Expected structure:
        dataset_dir/
        ├── human/
        │   ├── file1.wav
        │   └── file2.wav
        └── ai_generated/
            ├── file1.wav
            └── file2.wav
        
        Args:
            dataset_dir: Root directory containing audio files
            label_mapping: Optional mapping from folder names to labels
            
        Returns:
            Tuple of (audio_data_list, labels, file_paths)
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise DatasetPreparationError(f"Dataset directory not found: {dataset_dir}")
        
        if label_mapping is None:
            label_mapping = {
                'human': 'HUMAN',
                'ai_generated': 'AI_GENERATED',
                'ai': 'AI_GENERATED',
                'synthetic': 'AI_GENERATED',
                'real': 'HUMAN'
            }
        
        audio_data_list = []
        labels = []
        file_paths = []
        
        try:
            # Supported audio formats
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            
            for subfolder in dataset_dir.iterdir():
                if not subfolder.is_dir():
                    continue
                
                # Map folder name to label
                folder_name = subfolder.name.lower()
                if folder_name not in label_mapping:
                    logger.warning(f"Unknown folder '{folder_name}', skipping")
                    continue
                
                label = label_mapping[folder_name]
                
                # Load all audio files in this folder
                for audio_file in subfolder.iterdir():
                    if audio_file.suffix.lower() not in audio_extensions:
                        continue
                    
                    try:
                        # Load audio file
                        samples, sample_rate = librosa.load(
                            str(audio_file), 
                            sr=self.audio_processor.target_sample_rate,
                            mono=True
                        )
                        
                        duration = len(samples) / sample_rate
                        
                        audio_data = AudioData(
                            samples=samples,
                            sample_rate=sample_rate,
                            duration=duration,
                            channels=1
                        )
                        
                        # Validate audio data
                        if audio_data.is_valid() and duration >= self.audio_processor.min_duration:
                            audio_data_list.append(audio_data)
                            labels.append(label)
                            file_paths.append(str(audio_file))
                        else:
                            logger.warning(f"Invalid or too short audio file: {audio_file}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to load {audio_file}: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(audio_data_list)} audio files from {dataset_dir}")
            return audio_data_list, labels, file_paths
            
        except Exception as e:
            logger.error(f"Directory loading failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to load from directory: {str(e)}")
    
    def load_from_csv(self, 
                     csv_path: Union[str, Path],
                     audio_column: str = 'audio_path',
                     label_column: str = 'label',
                     base_path: Optional[Union[str, Path]] = None) -> Tuple[List[AudioData], List[str], List[str]]:
        """
        Load audio files from CSV metadata file.
        
        Args:
            csv_path: Path to CSV file with audio metadata
            audio_column: Column name containing audio file paths
            label_column: Column name containing labels
            base_path: Optional base path for relative audio paths
            
        Returns:
            Tuple of (audio_data_list, labels, file_paths)
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise DatasetPreparationError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            if audio_column not in df.columns:
                raise DatasetPreparationError(f"Audio column '{audio_column}' not found in CSV")
            
            if label_column not in df.columns:
                raise DatasetPreparationError(f"Label column '{label_column}' not found in CSV")
            
            audio_data_list = []
            labels = []
            file_paths = []
            
            base_path = Path(base_path) if base_path else csv_path.parent
            
            for _, row in df.iterrows():
                audio_path = base_path / row[audio_column]
                label = str(row[label_column]).upper()
                
                # Validate label
                if label not in ['HUMAN', 'AI_GENERATED']:
                    logger.warning(f"Invalid label '{label}' for {audio_path}, skipping")
                    continue
                
                try:
                    # Load audio file
                    samples, sample_rate = librosa.load(
                        str(audio_path),
                        sr=self.audio_processor.target_sample_rate,
                        mono=True
                    )
                    
                    duration = len(samples) / sample_rate
                    
                    audio_data = AudioData(
                        samples=samples,
                        sample_rate=sample_rate,
                        duration=duration,
                        channels=1
                    )
                    
                    # Validate audio data
                    if audio_data.is_valid() and duration >= self.audio_processor.min_duration:
                        audio_data_list.append(audio_data)
                        labels.append(label)
                        file_paths.append(str(audio_path))
                    else:
                        logger.warning(f"Invalid or too short audio file: {audio_path}")
                
                except Exception as e:
                    logger.warning(f"Failed to load {audio_path}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(audio_data_list)} audio files from CSV")
            return audio_data_list, labels, file_paths
            
        except Exception as e:
            logger.error(f"CSV loading failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to load from CSV: {str(e)}")


class BalancedDatasetHandler:
    """
    Utilities for handling balanced datasets with equal representation
    of AI-generated and human voice samples.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize balanced dataset handler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        logger.info(f"BalancedDatasetHandler initialized with random_state={random_state}")
    
    def balance_dataset(self, 
                       audio_data_list: List[AudioData],
                       labels: List[str],
                       file_paths: List[str],
                       balance_method: str = 'undersample') -> Tuple[List[AudioData], List[str], List[str]]:
        """
        Balance dataset to have equal representation of both classes.
        
        Args:
            audio_data_list: List of audio data samples
            labels: List of corresponding labels
            file_paths: List of corresponding file paths
            balance_method: Method for balancing ('undersample', 'oversample', or 'augment')
            
        Returns:
            Tuple of balanced (audio_data_list, labels, file_paths)
        """
        if len(audio_data_list) != len(labels) or len(labels) != len(file_paths):
            raise DatasetPreparationError("Audio data, labels, and file paths must have same length")
        
        try:
            # Count samples per class
            label_counts = Counter(labels)
            logger.info(f"Original class distribution: {dict(label_counts)}")
            
            if len(label_counts) != 2:
                raise DatasetPreparationError(f"Expected 2 classes, found {len(label_counts)}")
            
            # Separate samples by class
            human_indices = [i for i, label in enumerate(labels) if label == 'HUMAN']
            ai_indices = [i for i, label in enumerate(labels) if label == 'AI_GENERATED']
            
            if balance_method == 'undersample':
                return self._undersample_balance(
                    audio_data_list, labels, file_paths, human_indices, ai_indices
                )
            elif balance_method == 'oversample':
                return self._oversample_balance(
                    audio_data_list, labels, file_paths, human_indices, ai_indices
                )
            elif balance_method == 'augment':
                return self._augment_balance(
                    audio_data_list, labels, file_paths, human_indices, ai_indices
                )
            else:
                raise DatasetPreparationError(f"Unknown balance method: {balance_method}")
            
        except Exception as e:
            logger.error(f"Dataset balancing failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to balance dataset: {str(e)}")
    
    def _undersample_balance(self, 
                           audio_data_list: List[AudioData],
                           labels: List[str],
                           file_paths: List[str],
                           human_indices: List[int],
                           ai_indices: List[int]) -> Tuple[List[AudioData], List[str], List[str]]:
        """Balance by undersampling the majority class."""
        min_count = min(len(human_indices), len(ai_indices))
        
        # Randomly sample from each class
        selected_human = np.random.choice(human_indices, min_count, replace=False)
        selected_ai = np.random.choice(ai_indices, min_count, replace=False)
        
        # Combine and shuffle
        selected_indices = np.concatenate([selected_human, selected_ai])
        np.random.shuffle(selected_indices)
        
        balanced_audio = [audio_data_list[i] for i in selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        balanced_paths = [file_paths[i] for i in selected_indices]
        
        logger.info(f"Undersampled to {len(balanced_audio)} samples ({min_count} per class)")
        return balanced_audio, balanced_labels, balanced_paths
    
    def _oversample_balance(self, 
                          audio_data_list: List[AudioData],
                          labels: List[str],
                          file_paths: List[str],
                          human_indices: List[int],
                          ai_indices: List[int]) -> Tuple[List[AudioData], List[str], List[str]]:
        """Balance by oversampling the minority class."""
        max_count = max(len(human_indices), len(ai_indices))
        
        # Oversample minority class with replacement
        if len(human_indices) < max_count:
            selected_human = np.random.choice(human_indices, max_count, replace=True)
            selected_ai = np.array(ai_indices)
        else:
            selected_human = np.array(human_indices)
            selected_ai = np.random.choice(ai_indices, max_count, replace=True)
        
        # Combine and shuffle
        selected_indices = np.concatenate([selected_human, selected_ai])
        np.random.shuffle(selected_indices)
        
        balanced_audio = [audio_data_list[i] for i in selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        balanced_paths = [file_paths[i] for i in selected_indices]
        
        logger.info(f"Oversampled to {len(balanced_audio)} samples ({max_count} per class)")
        return balanced_audio, balanced_labels, balanced_paths
    
    def _augment_balance(self, 
                       audio_data_list: List[AudioData],
                       labels: List[str],
                       file_paths: List[str],
                       human_indices: List[int],
                       ai_indices: List[int]) -> Tuple[List[AudioData], List[str], List[str]]:
        """Balance by augmenting the minority class."""
        max_count = max(len(human_indices), len(ai_indices))
        augmentor = AudioDataAugmentor(self.random_state)
        
        balanced_audio = []
        balanced_labels = []
        balanced_paths = []
        
        # Add all original samples
        for i in range(len(audio_data_list)):
            balanced_audio.append(audio_data_list[i])
            balanced_labels.append(labels[i])
            balanced_paths.append(file_paths[i])
        
        # Determine which class needs augmentation
        if len(human_indices) < max_count:
            minority_indices = human_indices
            minority_label = 'HUMAN'
            needed_samples = max_count - len(human_indices)
        else:
            minority_indices = ai_indices
            minority_label = 'AI_GENERATED'
            needed_samples = max_count - len(ai_indices)
        
        # Generate augmented samples
        samples_generated = 0
        while samples_generated < needed_samples:
            # Randomly select a sample from minority class
            source_idx = np.random.choice(minority_indices)
            source_audio = audio_data_list[source_idx]
            source_path = file_paths[source_idx]
            
            # Generate augmented versions
            augmented_samples = augmentor.augment_audio_sample(source_audio)
            
            # Add augmented samples (skip the original which is already included)
            for aug_audio in augmented_samples[1:]:  # Skip first (original)
                if samples_generated >= needed_samples:
                    break
                
                balanced_audio.append(aug_audio)
                balanced_labels.append(minority_label)
                balanced_paths.append(f"{source_path}_aug_{samples_generated}")
                samples_generated += 1
        
        # Shuffle the final dataset
        combined = list(zip(balanced_audio, balanced_labels, balanced_paths))
        random.shuffle(combined)
        balanced_audio, balanced_labels, balanced_paths = zip(*combined)
        
        logger.info(f"Augmented to {len(balanced_audio)} samples with {samples_generated} augmented samples")
        return list(balanced_audio), list(balanced_labels), list(balanced_paths)
    
    def analyze_dataset_balance(self, labels: List[str]) -> Dict[str, Any]:
        """
        Analyze the balance of a dataset.
        
        Args:
            labels: List of dataset labels
            
        Returns:
            Dictionary containing balance analysis
        """
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        analysis = {
            'total_samples': total_samples,
            'class_counts': dict(label_counts),
            'class_proportions': {
                label: count / total_samples 
                for label, count in label_counts.items()
            },
            'is_balanced': len(set(label_counts.values())) == 1,
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()) if label_counts else 1.0
        }
        
        return analysis


class TrainingDatasetPreparator:
    """
    Main class for preparing training datasets with all necessary preprocessing,
    augmentation, and balancing operations.
    """
    
    def __init__(self, 
                 audio_processor: Optional[AudioProcessor] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 random_state: int = 42):
        """
        Initialize training dataset preparator.
        
        Args:
            audio_processor: AudioProcessor for audio preprocessing
            feature_extractor: FeatureExtractor for feature extraction
            random_state: Random seed for reproducibility
        """
        self.audio_processor = audio_processor or AudioProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.random_state = random_state
        
        self.loader = DatasetLoader(self.audio_processor)
        self.augmentor = AudioDataAugmentor(random_state)
        self.balancer = BalancedDatasetHandler(random_state)
        
        logger.info(f"TrainingDatasetPreparator initialized with random_state={random_state}")
    
    def prepare_dataset(self, 
                       dataset_source: Union[str, Path, Dict[str, Any]],
                       output_dir: Optional[Union[str, Path]] = None,
                       balance_method: str = 'undersample',
                       augmentation_config: Optional[Dict[str, Any]] = None,
                       extract_features: bool = True,
                       test_size: float = 0.2,
                       validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Complete dataset preparation pipeline.
        
        Args:
            dataset_source: Path to dataset directory, CSV file, or configuration dict
            output_dir: Optional directory to save prepared dataset
            balance_method: Method for balancing dataset
            augmentation_config: Configuration for data augmentation
            extract_features: Whether to extract features from audio
            test_size: Proportion for test set
            validation_size: Proportion of training data for validation
            
        Returns:
            Dictionary containing prepared dataset information
        """
        try:
            logger.info("Starting complete dataset preparation pipeline")
            
            # Step 1: Load raw audio data
            if isinstance(dataset_source, (str, Path)):
                dataset_path = Path(dataset_source)
                if dataset_path.is_dir():
                    audio_data_list, labels, file_paths = self.loader.load_from_directory(dataset_path)
                elif dataset_path.suffix.lower() == '.csv':
                    audio_data_list, labels, file_paths = self.loader.load_from_csv(dataset_path)
                else:
                    raise DatasetPreparationError(f"Unsupported dataset source: {dataset_source}")
            else:
                audio_data_list, labels, file_paths = self._load_dataset_from_config(dataset_source)
            
            if len(audio_data_list) == 0:
                raise DatasetPreparationError("No valid audio samples found in dataset")
            
            # Step 2: Analyze original dataset
            original_analysis = self.balancer.analyze_dataset_balance(labels)
            logger.info(f"Original dataset analysis: {original_analysis}")
            
            # Step 3: Balance dataset
            balanced_audio, balanced_labels, balanced_paths = self.balancer.balance_dataset(
                audio_data_list, labels, file_paths, balance_method
            )
            
            # Step 4: Apply augmentation if requested
            if augmentation_config and augmentation_config.get('enabled', False):
                augmented_audio, augmented_labels, augmented_paths = self._apply_augmentation(
                    balanced_audio, balanced_labels, balanced_paths, augmentation_config
                )
            else:
                augmented_audio, augmented_labels, augmented_paths = balanced_audio, balanced_labels, balanced_paths
            
            # Step 5: Extract features if requested
            if extract_features:
                features_list = self._extract_features_batch(augmented_audio)
            else:
                features_list = None
            
            # Step 6: Split into train/validation/test sets
            dataset_splits = self._create_dataset_splits(
                augmented_audio, augmented_labels, augmented_paths, features_list,
                test_size, validation_size
            )
            
            # Step 7: Save prepared dataset if output directory specified
            if output_dir:
                self._save_prepared_dataset(dataset_splits, output_dir)
            
            # Step 8: Generate final analysis
            final_analysis = self.balancer.analyze_dataset_balance(augmented_labels)
            
            preparation_info = {
                'original_analysis': original_analysis,
                'final_analysis': final_analysis,
                'dataset_splits': {
                    'train_size': len(dataset_splits['train']['labels']),
                    'validation_size': len(dataset_splits['validation']['labels']),
                    'test_size': len(dataset_splits['test']['labels'])
                },
                'balance_method': balance_method,
                'augmentation_applied': augmentation_config is not None and augmentation_config.get('enabled', False),
                'features_extracted': extract_features,
                'output_dir': str(output_dir) if output_dir else None
            }
            
            logger.info(f"Dataset preparation completed successfully: {preparation_info}")
            return {**preparation_info, 'data': dataset_splits}
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise DatasetPreparationError(f"Failed to prepare dataset: {str(e)}")

    def _load_dataset_from_config(
        self,
        dataset_config: Dict[str, Any]
    ) -> Tuple[List[AudioData], List[str], List[str]]:
        """Load a dataset from a configuration dictionary."""
        if not isinstance(dataset_config, dict):
            raise DatasetPreparationError("Dataset configuration must be a dictionary")

        source_value = dataset_config.get('path') or dataset_config.get('source')
        if not source_value:
            raise DatasetPreparationError("Dataset configuration must include 'path' or 'source'")

        source_type = str(dataset_config.get('type', '')).strip().lower()
        source_path = Path(source_value)

        if source_type in ('', 'auto'):
            if source_path.is_dir():
                source_type = 'directory'
            elif source_path.suffix.lower() == '.csv':
                source_type = 'csv'
            else:
                raise DatasetPreparationError(f"Unable to infer dataset source type for: {source_value}")

        if source_type in ('directory', 'dir', 'folder'):
            label_mapping = dataset_config.get('label_mapping') or dataset_config.get('label_map')
            return self.loader.load_from_directory(source_path, label_mapping=label_mapping)

        if source_type == 'csv':
            return self.loader.load_from_csv(
                source_path,
                audio_column=dataset_config.get('audio_column', 'audio_path'),
                label_column=dataset_config.get('label_column', 'label'),
                base_path=dataset_config.get('base_path')
            )

        raise DatasetPreparationError(f"Unsupported dataset configuration type: {source_type}")
    
    def _apply_augmentation(self, 
                          audio_data_list: List[AudioData],
                          labels: List[str],
                          file_paths: List[str],
                          augmentation_config: Dict[str, Any]) -> Tuple[List[AudioData], List[str], List[str]]:
        """Apply data augmentation to the dataset."""
        augmented_audio = []
        augmented_labels = []
        augmented_paths = []
        
        for i, (audio_data, label, path) in enumerate(zip(audio_data_list, labels, file_paths)):
            # Add original sample
            augmented_audio.append(audio_data)
            augmented_labels.append(label)
            augmented_paths.append(path)
            
            # Add augmented samples
            aug_samples = self.augmentor.augment_audio_sample(audio_data, augmentation_config)
            for j, aug_audio in enumerate(aug_samples[1:]):  # Skip original
                augmented_audio.append(aug_audio)
                augmented_labels.append(label)
                augmented_paths.append(f"{path}_aug_{j}")
        
        logger.info(f"Applied augmentation: {len(audio_data_list)} -> {len(augmented_audio)} samples")
        return augmented_audio, augmented_labels, augmented_paths
    
    def _extract_features_batch(self, audio_data_list: List[AudioData]) -> List[FeatureVector]:
        """Extract features from a batch of audio samples."""
        features_list = []
        
        for i, audio_data in enumerate(audio_data_list):
            try:
                features = self.feature_extractor.extract_all_features(audio_data)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Feature extraction failed for sample {i}: {str(e)}")
                # Skip this sample or use a dummy feature vector
                continue
        
        logger.info(f"Extracted features from {len(features_list)} audio samples")
        return features_list
    
    def _create_dataset_splits(self, 
                             audio_data_list: List[AudioData],
                             labels: List[str],
                             file_paths: List[str],
                             features_list: Optional[List[FeatureVector]],
                             test_size: float,
                             validation_size: float) -> Dict[str, Dict[str, Any]]:
        """Create train/validation/test splits."""
        if not 0.0 <= test_size < 1.0:
            raise DatasetPreparationError("test_size must be in the range [0.0, 1.0)")
        if not 0.0 <= validation_size < 1.0:
            raise DatasetPreparationError("validation_size must be in the range [0.0, 1.0)")

        has_features = features_list is not None and len(features_list) == len(audio_data_list)

        def empty_split() -> Tuple[List[Any], List[str], List[str], Optional[List[FeatureVector]]]:
            return [], [], [], [] if has_features else None

        def stratify_labels(current_labels: List[str]) -> Optional[List[str]]:
            label_counts = Counter(current_labels)
            if len(label_counts) < 2 or min(label_counts.values()) < 2:
                return None
            return current_labels

        data_temp = list(audio_data_list)
        labels_temp = list(labels)
        paths_temp = list(file_paths)
        features_temp = list(features_list) if has_features else None

        if test_size > 0.0:
            split_args: List[Any] = [data_temp, labels_temp, paths_temp]
            if has_features:
                split_args.append(features_temp)

            split_result = train_test_split(
                *split_args,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_labels(labels_temp)
            )

            if has_features:
                (
                    data_temp, data_test,
                    labels_temp, labels_test,
                    paths_temp, paths_test,
                    features_temp, features_test
                ) = split_result
            else:
                (
                    data_temp, data_test,
                    labels_temp, labels_test,
                    paths_temp, paths_test
                ) = split_result
                features_test = None
        else:
            data_test, labels_test, paths_test, features_test = empty_split()

        if validation_size > 0.0 and len(data_temp) > 0:
            split_args = [data_temp, labels_temp, paths_temp]
            if has_features:
                split_args.append(features_temp)

            split_result = train_test_split(
                *split_args,
                test_size=validation_size,
                random_state=self.random_state,
                stratify=stratify_labels(labels_temp)
            )

            if has_features:
                (
                    data_train, data_val,
                    labels_train, labels_val,
                    paths_train, paths_val,
                    features_train, features_val
                ) = split_result
            else:
                (
                    data_train, data_val,
                    labels_train, labels_val,
                    paths_train, paths_val
                ) = split_result
                features_train, features_val = None, None
        else:
            data_train = data_temp
            labels_train = labels_temp
            paths_train = paths_temp
            features_train = features_temp
            data_val, labels_val, paths_val, features_val = empty_split()
        
        return {
            'train': {
                'audio_data': data_train,
                'labels': labels_train,
                'file_paths': paths_train,
                'features': features_train
            },
            'validation': {
                'audio_data': data_val,
                'labels': labels_val,
                'file_paths': paths_val,
                'features': features_val
            },
            'test': {
                'audio_data': data_test,
                'labels': labels_test,
                'file_paths': paths_test,
                'features': features_test
            }
        }
    
    def _save_prepared_dataset(self, dataset_splits: Dict[str, Dict[str, Any]], output_dir: Union[str, Path]) -> None:
        """Save prepared dataset to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in dataset_splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save metadata
            metadata = {
                'labels': split_data['labels'],
                'file_paths': split_data['file_paths'],
                'n_samples': len(split_data['labels'])
            }
            
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save features if available
            if split_data['features']:
                features_arrays = [f.to_model_array() for f in split_data['features']]
                np.save(split_dir / 'features.npy', np.array(features_arrays))
            
            logger.info(f"Saved {split_name} split to {split_dir}")


def create_dataset_preparator(random_state: int = 42) -> TrainingDatasetPreparator:
    """
    Create a training dataset preparator with default configuration.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        TrainingDatasetPreparator: Configured dataset preparator
    """
    return TrainingDatasetPreparator(random_state=random_state)
