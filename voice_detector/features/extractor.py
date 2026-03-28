"""
Feature Extraction Layer - Extracts acoustic features for ML analysis

This module implements the FeatureExtractor component that extracts various
acoustic features from audio data for AI vs human voice classification.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
import logging
from ..models.data_models import AudioData, SpectralFeatures, TemporalFeatures, FeatureVector

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts acoustic features that distinguish AI-generated from human voices.
    
    This class implements feature extraction methods for:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Spectral features (centroid, bandwidth, rolloff, zero-crossing rate)
    - Pitch and energy variance analysis
    - Temporal consistency analysis
    """
    
    def __init__(self, 
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 enable_caching: bool = True,
                 optimize_for_speed: bool = True):
        """
        Initialize feature extractor with configuration parameters.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
            enable_caching: Enable caching of intermediate computations
            optimize_for_speed: Enable speed optimizations
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.enable_caching = enable_caching
        self.optimize_for_speed = optimize_for_speed
        
        # Performance optimization: Pre-compute mel filter bank
        if optimize_for_speed:
            self._mel_filters = None
            self._stft_cache = {}
            self._cache_max_size = 10  # Limit cache size to prevent memory issues
        
        logger.info(f"FeatureExtractor initialized with n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}, optimized={optimize_for_speed}")
    
    def extract_mfcc_features(self, audio_data: AudioData) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients from audio.
        
        MFCCs are widely used in speech processing and capture the spectral
        characteristics of audio in a compact representation.
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            np.ndarray: MFCC feature matrix of shape (n_mfcc, n_frames)
            
        Raises:
            ValueError: If audio data is invalid or MFCC extraction fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for MFCC extraction")
        
        try:
            # Performance optimization: Use cached STFT if available
            stft_data = None
            if self.optimize_for_speed and self.enable_caching:
                cache_key = f"stft_{hash(audio_data.samples.tobytes())}"
                if cache_key in self._stft_cache:
                    stft_data = self._stft_cache[cache_key]
                    logger.debug("Using cached STFT data")
            
            # Extract MFCCs using librosa with optimizations
            if self.optimize_for_speed:
                # Use faster parameters for real-time performance
                mfccs = librosa.feature.mfcc(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    fmax=8000,  # Limit frequency range for speech
                    S=stft_data  # Use cached STFT if available
                )
            else:
                # Standard extraction
                mfccs = librosa.feature.mfcc(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
            
            # Cache STFT for reuse if enabled
            if self.optimize_for_speed and self.enable_caching and stft_data is None:
                # Compute and cache STFT for potential reuse
                stft_data = np.abs(librosa.stft(
                    audio_data.samples, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length
                ))**2
                cache_key = f"stft_{hash(audio_data.samples.tobytes())}"
                
                # Manage cache size
                if len(self._stft_cache) >= self._cache_max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._stft_cache))
                    del self._stft_cache[oldest_key]
                
                self._stft_cache[cache_key] = stft_data
            
            # Normalize MFCCs to have zero mean and unit variance
            mfccs_normalized = self._normalize_features(mfccs)
            
            # Validate output dimensionality
            if mfccs_normalized.shape[0] != self.n_mfcc:
                raise ValueError(f"MFCC extraction produced unexpected shape: {mfccs_normalized.shape}")
            
            logger.debug(f"Extracted MFCC features with shape: {mfccs_normalized.shape}")
            return mfccs_normalized
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract MFCC features: {str(e)}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            features: Feature matrix to normalize
            
        Returns:
            np.ndarray: Normalized feature matrix
        """
        # Handle edge case where features might be constant
        std = np.std(features, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        normalized = (features - np.mean(features, axis=1, keepdims=True)) / std
        
        # Ensure no NaN or infinite values
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
    
    def _validate_feature_ranges(self, features: np.ndarray, feature_name: str) -> bool:
        """
        Validate that extracted features are within reasonable ranges.
        
        Args:
            features: Feature array to validate
            feature_name: Name of the feature for logging
            
        Returns:
            bool: True if features are valid
        """
        if not isinstance(features, np.ndarray) or features.size == 0:
            logger.warning(f"{feature_name} features are empty or invalid")
            return False
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning(f"{feature_name} features contain NaN or infinite values")
            return False
        
        return True
    def extract_spectral_features(self, audio_data: AudioData) -> SpectralFeatures:
        """
        Extract spectral features including centroid, bandwidth, and rolloff.
        
        Spectral features capture the frequency distribution characteristics
        of the audio signal, which can differ between AI-generated and human voices.
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            SpectralFeatures: Object containing all spectral feature arrays
            
        Raises:
            ValueError: If audio data is invalid or spectral extraction fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for spectral feature extraction")
        
        try:
            # Performance optimization: Compute STFT once and reuse
            if self.optimize_for_speed:
                # Use cached STFT if available
                cache_key = f"stft_{hash(audio_data.samples.tobytes())}"
                if self.enable_caching and cache_key in self._stft_cache:
                    S = self._stft_cache[cache_key]
                else:
                    S = np.abs(librosa.stft(
                        audio_data.samples, 
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length
                    ))**2
                    
                    # Cache for reuse
                    if self.enable_caching:
                        if len(self._stft_cache) >= self._cache_max_size:
                            oldest_key = next(iter(self._stft_cache))
                            del self._stft_cache[oldest_key]
                        self._stft_cache[cache_key] = S
                
                # Extract all spectral features using the same STFT
                centroid = librosa.feature.spectral_centroid(
                    S=S, sr=audio_data.sample_rate, hop_length=self.hop_length
                )[0]
                
                bandwidth = librosa.feature.spectral_bandwidth(
                    S=S, sr=audio_data.sample_rate, hop_length=self.hop_length
                )[0]
                
                rolloff = librosa.feature.spectral_rolloff(
                    S=S, sr=audio_data.sample_rate, hop_length=self.hop_length,
                    roll_percent=0.85
                )[0]
                
                # Zero crossing rate doesn't use STFT
                zero_crossing_rate = librosa.feature.zero_crossing_rate(
                    y=audio_data.samples, hop_length=self.hop_length
                )[0]
                
            else:
                # Standard extraction (compute STFT multiple times)
                centroid = librosa.feature.spectral_centroid(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length
                )[0]
                
                bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length
                )[0]
                
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length,
                    roll_percent=0.85
                )[0]
                
                zero_crossing_rate = librosa.feature.zero_crossing_rate(
                    y=audio_data.samples,
                    hop_length=self.hop_length
                )[0]
            
            # Validate all features
            features = [centroid, bandwidth, rolloff, zero_crossing_rate]
            feature_names = ['centroid', 'bandwidth', 'rolloff', 'zero_crossing_rate']
            
            for feature, name in zip(features, feature_names):
                if not self._validate_feature_ranges(feature, f"spectral_{name}"):
                    raise ValueError(f"Invalid {name} feature values")
            
            # Ensure all features have the same length
            lengths = [len(f) for f in features]
            if len(set(lengths)) > 1:
                raise ValueError(f"Spectral features have inconsistent lengths: {lengths}")
            
            spectral_features = SpectralFeatures(
                centroid=centroid,
                bandwidth=bandwidth,
                rolloff=rolloff,
                zero_crossing_rate=zero_crossing_rate
            )
            
            # Validate the complete spectral features object
            if not spectral_features.validate():
                raise ValueError("Spectral features validation failed")
            
            logger.debug(f"Extracted spectral features with {len(centroid)} frames")
            return spectral_features
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract spectral features: {str(e)}")
    def analyze_pitch_variance(self, audio_data: AudioData) -> float:
        """
        Calculate pitch variation patterns in the audio signal.
        
        Pitch variance can help distinguish between AI-generated voices
        (which may have more consistent pitch) and human voices (which
        naturally vary in pitch).
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            float: Pitch variance measure (non-negative)
            
        Raises:
            ValueError: If audio data is invalid or pitch analysis fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for pitch variance analysis")
        
        try:
            if self.optimize_for_speed:
                # Much faster pitch estimation using a simplified approach
                # This trades some accuracy for significant speed improvement
                pitch_values = []
                
                # Use larger hop size for speed
                fast_hop_length = self.hop_length * 4  # 4x faster
                frame_length = min(1024, self.n_fft // 2)  # Smaller frame for speed
                
                # Process fewer frames for speed
                for i in range(0, len(audio_data.samples) - frame_length, fast_hop_length):
                    frame = audio_data.samples[i:i + frame_length]
                    
                    # Skip frames with very low energy (faster than autocorrelation)
                    frame_energy = np.sum(frame**2)
                    if frame_energy < 0.001:
                        continue
                    
                    # Very simple zero-crossing based pitch estimation (much faster)
                    zero_crossings = np.where(np.diff(np.signbit(frame)))[0]
                    if len(zero_crossings) > 4:  # Need some zero crossings
                        # Estimate period from zero crossings
                        avg_period = len(frame) / (len(zero_crossings) / 2)
                        if avg_period > 0:
                            pitch = audio_data.sample_rate / avg_period
                            if 80 <= pitch <= 800:  # Valid pitch range for speech
                                pitch_values.append(pitch)
                    
                    # Limit processing for speed (max 50 frames)
                    if len(pitch_values) >= 50:
                        break
                
            else:
                # Standard pitch extraction using librosa's piptrack
                pitches, magnitudes = librosa.piptrack(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length,
                    threshold=0.1
                )
                
                # Extract the most prominent pitch at each frame
                pitch_values = []
                for frame_idx in range(pitches.shape[1]):
                    frame_pitches = pitches[:, frame_idx]
                    frame_magnitudes = magnitudes[:, frame_idx]
                    
                    if np.any(frame_magnitudes > 0):
                        max_mag_idx = np.argmax(frame_magnitudes)
                        pitch = frame_pitches[max_mag_idx]
                        if pitch > 0:
                            pitch_values.append(pitch)
            
            # Calculate pitch variance
            if len(pitch_values) < 2:
                pitch_variance = 0.0
            else:
                pitch_array = np.array(pitch_values)
                mean_pitch = np.mean(pitch_array)
                if mean_pitch > 0:
                    pitch_variance = np.std(pitch_array) / mean_pitch
                else:
                    pitch_variance = 0.0
            
            # Ensure non-negative result
            pitch_variance = max(0.0, float(pitch_variance))
            
            logger.debug(f"Calculated pitch variance: {pitch_variance} from {len(pitch_values)} pitch values")
            return pitch_variance
            
        except Exception as e:
            logger.error(f"Pitch variance analysis failed: {str(e)}")
            raise ValueError(f"Failed to analyze pitch variance: {str(e)}")
    
    def analyze_energy_variance(self, audio_data: AudioData) -> float:
        """
        Calculate energy variation patterns in the audio signal.
        
        Energy variance measures how much the signal energy fluctuates,
        which can be different between AI-generated and human voices.
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            float: Energy variance measure (non-negative)
            
        Raises:
            ValueError: If audio data is invalid or energy analysis fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for energy variance analysis")
        
        try:
            # Calculate RMS energy for each frame
            rms_energy = librosa.feature.rms(
                y=audio_data.samples,
                hop_length=self.hop_length,
                frame_length=self.n_fft
            )[0]  # Remove extra dimension
            
            # Calculate energy variance
            if len(rms_energy) < 2:
                energy_variance = 0.0
            else:
                # Use coefficient of variation for normalized variance
                mean_energy = np.mean(rms_energy)
                if mean_energy > 0:
                    energy_variance = np.std(rms_energy) / mean_energy
                else:
                    energy_variance = 0.0
            
            # Ensure non-negative result
            energy_variance = max(0.0, float(energy_variance))
            
            logger.debug(f"Calculated energy variance: {energy_variance} from {len(rms_energy)} energy frames")
            return energy_variance
            
        except Exception as e:
            logger.error(f"Energy variance analysis failed: {str(e)}")
            raise ValueError(f"Failed to analyze energy variance: {str(e)}")
    def analyze_temporal_consistency(self, audio_data: AudioData) -> TemporalFeatures:
        """
        Analyze temporal consistency patterns in the audio signal.
        
        Temporal analysis examines frame-to-frame variations, rhythm regularity,
        and pause patterns that may differ between AI-generated and human voices.
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            TemporalFeatures: Object containing temporal analysis results
            
        Raises:
            ValueError: If audio data is invalid or temporal analysis fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for temporal consistency analysis")
        
        try:
            # Calculate frame-to-frame consistency using spectral features
            frame_consistency = self._calculate_frame_consistency(audio_data)
            
            # Calculate rhythm regularity using onset detection
            rhythm_regularity = self._calculate_rhythm_regularity(audio_data)
            
            # Analyze pause patterns using energy-based segmentation
            pause_patterns = self._analyze_pause_patterns(audio_data)
            
            temporal_features = TemporalFeatures(
                frame_consistency=frame_consistency,
                rhythm_regularity=rhythm_regularity,
                pause_patterns=pause_patterns
            )
            
            # Validate the temporal features
            if not temporal_features.validate():
                raise ValueError("Temporal features validation failed")
            
            logger.debug(f"Extracted temporal features: consistency={frame_consistency:.3f}, "
                        f"rhythm={rhythm_regularity:.3f}, pauses={len(pause_patterns)}")
            return temporal_features
            
        except Exception as e:
            logger.error(f"Temporal consistency analysis failed: {str(e)}")
            raise ValueError(f"Failed to analyze temporal consistency: {str(e)}")
    
    def _calculate_frame_consistency(self, audio_data: AudioData) -> float:
        """
        Calculate frame-to-frame consistency using spectral centroid variations.
        
        Args:
            audio_data: AudioData object
            
        Returns:
            float: Frame consistency score between 0.0 and 1.0
        """
        try:
            if self.optimize_for_speed:
                # Faster approach: use cached STFT if available, or compute with larger hop
                cache_key = f"stft_{hash(audio_data.samples.tobytes())}"
                if self.enable_caching and cache_key in self._stft_cache:
                    S = self._stft_cache[cache_key]
                    centroid = librosa.feature.spectral_centroid(
                        S=S, sr=audio_data.sample_rate, hop_length=self.hop_length
                    )[0]
                else:
                    # Use larger hop length for speed
                    fast_hop = self.hop_length * 2
                    centroid = librosa.feature.spectral_centroid(
                        y=audio_data.samples,
                        sr=audio_data.sample_rate,
                        hop_length=fast_hop
                    )[0]
            else:
                # Standard approach
                centroid = librosa.feature.spectral_centroid(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length
                )[0]
            
            if len(centroid) < 2:
                return 1.0  # Perfect consistency for single frame
            
            # Calculate frame-to-frame differences
            frame_diffs = np.abs(np.diff(centroid))
            
            # Normalize by mean centroid value to get relative changes
            mean_centroid = np.mean(centroid)
            if mean_centroid > 0:
                normalized_diffs = frame_diffs / mean_centroid
            else:
                normalized_diffs = frame_diffs
            
            # Convert to consistency score (lower differences = higher consistency)
            mean_diff = np.mean(normalized_diffs)
            consistency = 1.0 / (1.0 + mean_diff)  # Maps [0, inf) to (0, 1]
            
            return float(np.clip(consistency, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Frame consistency calculation failed: {str(e)}")
            return 0.5  # Default moderate consistency
    
    def _calculate_rhythm_regularity(self, audio_data: AudioData) -> float:
        """
        Calculate rhythm regularity using onset detection and timing analysis.
        
        Args:
            audio_data: AudioData object
            
        Returns:
            float: Rhythm regularity score between 0.0 and 1.0
        """
        try:
            if self.optimize_for_speed:
                # Faster onset detection with reduced precision
                onset_frames = librosa.onset.onset_detect(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length * 2,  # Larger hop for speed
                    units='frames',
                    pre_max=10,  # Reduced from default
                    post_max=10,  # Reduced from default
                    pre_avg=5,   # Reduced from default
                    post_avg=5,  # Reduced from default
                    delta=0.1,   # Higher threshold for speed
                    wait=10      # Reduced from default
                )
            else:
                # Standard onset detection
                onset_frames = librosa.onset.onset_detect(
                    y=audio_data.samples,
                    sr=audio_data.sample_rate,
                    hop_length=self.hop_length,
                    units='frames'
                )
            
            if len(onset_frames) < 3:
                return 0.5  # Default moderate regularity for few onsets
            
            # Convert frames to time
            hop_length = self.hop_length * 2 if self.optimize_for_speed else self.hop_length
            onset_times = librosa.frames_to_time(
                onset_frames,
                sr=audio_data.sample_rate,
                hop_length=hop_length
            )
            
            # Calculate inter-onset intervals
            intervals = np.diff(onset_times)
            
            if len(intervals) < 2:
                return 0.5  # Default for insufficient intervals
            
            # Calculate regularity as inverse of coefficient of variation
            mean_interval = np.mean(intervals)
            if mean_interval > 0:
                cv = np.std(intervals) / mean_interval
                regularity = 1.0 / (1.0 + cv)  # Maps [0, inf) to (0, 1]
            else:
                regularity = 0.5
            
            return float(np.clip(regularity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Rhythm regularity calculation failed: {str(e)}")
            return 0.5  # Default moderate regularity
    
    def _analyze_pause_patterns(self, audio_data: AudioData) -> np.ndarray:
        """
        Analyze pause patterns using energy-based segmentation.
        
        Args:
            audio_data: AudioData object
            
        Returns:
            np.ndarray: Array of pause durations in seconds
        """
        try:
            # Calculate RMS energy
            rms_energy = librosa.feature.rms(
                y=audio_data.samples,
                hop_length=self.hop_length,
                frame_length=self.n_fft
            )[0]
            
            # Define energy threshold for pause detection (10% of max energy)
            energy_threshold = 0.1 * np.max(rms_energy)
            
            # Find frames below threshold (potential pauses)
            pause_frames = rms_energy < energy_threshold
            
            # Convert to time domain
            frame_times = librosa.frames_to_time(
                np.arange(len(pause_frames)),
                sr=audio_data.sample_rate,
                hop_length=self.hop_length
            )
            
            # Find continuous pause segments
            pause_segments = []
            in_pause = False
            pause_start = 0
            
            for i, is_pause in enumerate(pause_frames):
                if is_pause and not in_pause:
                    # Start of pause
                    in_pause = True
                    pause_start = frame_times[i]
                elif not is_pause and in_pause:
                    # End of pause
                    in_pause = False
                    pause_duration = frame_times[i] - pause_start
                    if pause_duration > 0.1:  # Only count pauses longer than 100ms
                        pause_segments.append(pause_duration)
            
            # Handle case where audio ends in a pause
            if in_pause:
                pause_duration = frame_times[-1] - pause_start
                if pause_duration > 0.1:
                    pause_segments.append(pause_duration)
            
            # Return as numpy array, with at least one element
            if len(pause_segments) == 0:
                pause_segments = [0.0]  # No significant pauses found
            
            return np.array(pause_segments)
            
        except Exception as e:
            logger.warning(f"Pause pattern analysis failed: {str(e)}")
            return np.array([0.0])  # Default no pauses
    def combine_features(self, 
                        mfcc_features: np.ndarray,
                        spectral_features: SpectralFeatures,
                        pitch_variance: float,
                        energy_variance: float,
                        temporal_features: TemporalFeatures) -> FeatureVector:
        """
        Combine all extracted features into a unified feature vector.
        
        This method creates a consistent feature vector format that combines
        all feature types with proper scaling and normalization for ML model input.
        
        Args:
            mfcc_features: MFCC coefficient matrix
            spectral_features: Spectral analysis features
            pitch_variance: Pitch variation measure
            energy_variance: Energy variation measure
            temporal_features: Temporal consistency features
            
        Returns:
            FeatureVector: Combined and validated feature vector
            
        Raises:
            ValueError: If feature combination or validation fails
        """
        try:
            # Validate all input features
            if not self._validate_feature_ranges(mfcc_features, "MFCC"):
                raise ValueError("Invalid MFCC features provided")
            
            if not spectral_features.validate():
                raise ValueError("Invalid spectral features provided")
            
            if not isinstance(pitch_variance, (int, float)) or pitch_variance < 0:
                raise ValueError(f"Invalid pitch variance: {pitch_variance}")
            
            if not isinstance(energy_variance, (int, float)) or energy_variance < 0:
                raise ValueError(f"Invalid energy variance: {energy_variance}")
            
            if not temporal_features.validate():
                raise ValueError("Invalid temporal features provided")
            
            # Create the combined feature vector
            feature_vector = FeatureVector(
                mfcc_features=mfcc_features,
                spectral_features=spectral_features,
                pitch_variance=float(pitch_variance),
                energy_variance=float(energy_variance),
                temporal_features=temporal_features
            )
            
            # Validate the complete feature vector
            if not feature_vector.validate():
                raise ValueError("Combined feature vector validation failed")
            
            logger.debug(f"Created feature vector with {feature_vector.get_feature_info()['total_features']} total features")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature combination failed: {str(e)}")
            raise ValueError(f"Failed to combine features: {str(e)}")
    
    def extract_all_features(self, audio_data: AudioData) -> FeatureVector:
        """
        Extract all features from audio data in a single call.
        
        This is the main method for complete feature extraction that processes
        audio through all feature extraction stages and returns a unified
        feature vector ready for ML model input.
        
        Args:
            audio_data: AudioData object containing audio samples and metadata
            
        Returns:
            FeatureVector: Complete feature vector with all extracted features
            
        Raises:
            ValueError: If any feature extraction step fails
        """
        if not audio_data.is_valid():
            raise ValueError("Invalid audio data provided for feature extraction")
        
        logger.info(f"Starting complete feature extraction for {audio_data.duration:.2f}s audio")
        
        try:
            # Extract MFCC features
            mfcc_features = self.extract_mfcc_features(audio_data)
            
            # Extract spectral features
            spectral_features = self.extract_spectral_features(audio_data)
            
            # Analyze pitch variance
            pitch_variance = self.analyze_pitch_variance(audio_data)
            
            # Analyze energy variance
            energy_variance = self.analyze_energy_variance(audio_data)
            
            # Analyze temporal consistency
            temporal_features = self.analyze_temporal_consistency(audio_data)
            
            # Combine all features
            feature_vector = self.combine_features(
                mfcc_features=mfcc_features,
                spectral_features=spectral_features,
                pitch_variance=pitch_variance,
                energy_variance=energy_variance,
                temporal_features=temporal_features
            )
            
            logger.info(f"Successfully extracted all features: {feature_vector.get_feature_info()}")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Complete feature extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract all features: {str(e)}")
    
    def get_feature_dimensions(self) -> dict:
        """
        Get information about expected feature dimensions.
        
        Returns:
            dict: Information about feature vector structure and dimensions
        """
        return {
            "mfcc_coefficients": self.n_mfcc,
            "spectral_features": 4,  # centroid, bandwidth, rolloff, zcr
            "variance_features": 2,  # pitch_variance, energy_variance
            "temporal_features": 3,  # frame_consistency, rhythm_regularity, pause_patterns_mean
            "processing_params": {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels
            }
        }
    
    def clear_cache(self) -> None:
        """
        Clear internal caches to free memory.
        
        This method should be called periodically during sustained load
        to prevent memory accumulation.
        """
        if hasattr(self, '_stft_cache'):
            self._stft_cache.clear()
            logger.debug("Feature extractor cache cleared")
    
    def get_cache_info(self) -> dict:
        """
        Get information about cache usage.
        
        Returns:
            dict: Cache statistics
        """
        if not hasattr(self, '_stft_cache'):
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": self.enable_caching,
            "cache_size": len(self._stft_cache),
            "cache_max_size": self._cache_max_size,
            "optimize_for_speed": self.optimize_for_speed
        }