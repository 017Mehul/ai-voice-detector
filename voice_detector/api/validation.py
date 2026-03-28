"""
Input Validation and Sanitization for API Requests

This module provides comprehensive validation and sanitization functions
for API requests, including Base64 validation, audio size checks, and
security measures for input handling.
"""

import base64
import re
import binascii
from typing import Optional
import io
import hashlib
import time
import os
import threading

from ..utils.logging_config import get_logger
from ..models.data_models import DetectionRequest, ValidationResult
from ..utils.config import get_config


logger = get_logger(__name__)

# Configuration constants
config = get_config()
MAX_AUDIO_SIZE_BYTES = config.audio.max_size_mb * 1024 * 1024  # Convert MB to bytes
MIN_AUDIO_DURATION_SECONDS = config.audio.min_duration_seconds
MAX_BASE64_SIZE = int(MAX_AUDIO_SIZE_BYTES * 1.37)  # Base64 is ~37% larger than binary
MAX_REQUEST_SIZE = config.security.max_request_size

# Security constants
MAX_VALIDATION_TIME_SECONDS = 5.0  # Prevent DoS through slow validation
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>',
    r'javascript:',
    r'data:text/html',
    r'<?php',
    r'<%',
    r'{{',
    r'${',
    r'eval\s*\(',
    r'exec\s*\(',
    r'system\s*\(',
    r'shell_exec\s*\(',
    r'passthru\s*\(',
    r'file_get_contents\s*\(',
    r'fopen\s*\(',
    r'curl_exec\s*\(',
    r'wget\s+',
    r'curl\s+',
]

_duplicate_request_lock = threading.Lock()
_recent_request_fingerprints: dict[str, dict[str, float]] = {}
_last_duplicate_cleanup = 0.0
_DUPLICATE_TRACKING_CLEANUP_INTERVAL_SECONDS = 300


def validate_detection_request(request: DetectionRequest) -> ValidationResult:
    """
    Comprehensive validation of detection request.
    
    Args:
        request: DetectionRequest to validate
        
    Returns:
        ValidationResult with validation status and error details
    """
    try:
        # Basic request validation
        if not request.validate():
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_REQUEST",
                error_message="Request validation failed. Please check the audio_base64 field."
            )
        
        # Validate Base64 format more thoroughly
        base64_validation = validate_base64_format(request.audio_base64)
        if not base64_validation.is_valid:
            return base64_validation
        
        # Validate decoded audio header (basic MP3 check)
        audio_validation = validate_audio_header(request.audio_base64)
        if not audio_validation.is_valid:
            return audio_validation
        
        logger.debug("Detection request validation successful")
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during request validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during request validation"
        )


def validate_base64_format(base64_string: str) -> ValidationResult:
    """
    Validate Base64 string format and decodability.
    
    Args:
        base64_string: Base64 string to validate
        
    Returns:
        ValidationResult with validation status
    """
    try:
        if not base64_string or not isinstance(base64_string, str):
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message="audio_base64 must be a non-empty string"
            )
        
        # Remove whitespace and check length
        cleaned = re.sub(r'\s+', '', base64_string)
        if len(cleaned) == 0:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message="audio_base64 is empty after removing whitespace"
            )
        
        # Check Base64 character set
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', cleaned):
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message="audio_base64 contains invalid Base64 characters"
            )
        
        # Check Base64 length (must be multiple of 4)
        if len(cleaned) % 4 != 0:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message="audio_base64 length is not a multiple of 4"
            )
        
        # Try to decode to verify it's valid Base64
        try:
            decoded = base64.b64decode(cleaned, validate=True)
            if len(decoded) == 0:
                return ValidationResult(
                    is_valid=False,
                    error_code="INVALID_BASE64",
                    error_message="audio_base64 decodes to empty data"
                )
        except (binascii.Error, ValueError) as e:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message=f"audio_base64 is not valid Base64 data: {str(e)}"
            )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during Base64 validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during Base64 validation"
        )


def validate_audio_size(base64_string: str) -> ValidationResult:
    """
    Validate audio size constraints.
    
    Args:
        base64_string: Base64-encoded audio data
        
    Returns:
        ValidationResult with size validation status
    """
    try:
        # Check Base64 string size
        if len(base64_string) > MAX_BASE64_SIZE:
            return ValidationResult(
                is_valid=False,
                error_code="PAYLOAD_TOO_LARGE",
                error_message=f"Audio file size exceeds maximum allowed limit of {MAX_AUDIO_SIZE_BYTES // (1024*1024)}MB"
            )
        
        # Decode and check actual binary size
        try:
            cleaned = re.sub(r'\s+', '', base64_string)
            decoded = base64.b64decode(cleaned, validate=True)
            
            if len(decoded) > MAX_AUDIO_SIZE_BYTES:
                return ValidationResult(
                    is_valid=False,
                    error_code="PAYLOAD_TOO_LARGE",
                    error_message=f"Decoded audio size exceeds maximum allowed limit of {MAX_AUDIO_SIZE_BYTES // (1024*1024)}MB"
                )
            
            # Check minimum size (should be at least a few KB for valid audio)
            if len(decoded) < 1024:  # 1KB minimum
                return ValidationResult(
                    is_valid=False,
                    error_code="AUDIO_TOO_SMALL",
                    error_message="Audio file is too small to be valid"
                )
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message=f"Cannot decode Base64 data for size validation: {str(e)}"
            )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during size validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during size validation"
        )


def validate_audio_header(base64_string: str) -> ValidationResult:
    """
    Validate audio file header to ensure it is a supported MP3, WAV, or FLAC format.
    
    Args:
        base64_string: Base64-encoded audio data
        
    Returns:
        ValidationResult with format validation status
    """
    try:
        # Decode Base64 to get binary data
        cleaned = re.sub(r'\s+', '', base64_string)
        decoded = base64.b64decode(cleaned, validate=True)
        
        # Check for MP3 header signatures
        if len(decoded) < 4:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_AUDIO_FORMAT",
                error_message="Audio data is too short to contain valid header"
            )
        
        # WAV files are supported by the audio processor and commonly used in tests.
        if decoded.startswith(b'RIFF') and decoded[8:12] == b'WAVE':
            return ValidationResult(is_valid=True)

        # Native FLAC streams start with the "fLaC" marker.
        if decoded.startswith(b'fLaC'):
            return ValidationResult(is_valid=True)

        # Check for MP3 frame header (starts with 0xFF and specific bit patterns)
        # MP3 frame header: 11111111 111xxxxx (first 11 bits are all 1s)
        first_bytes = decoded[:4]
        
        # Look for MP3 sync word in first few bytes
        mp3_found = False
        for i in range(min(100, len(decoded) - 1)):  # Check first 100 bytes
            if decoded[i] == 0xFF and len(decoded) > i + 1:
                second_byte = decoded[i + 1]
                # Check if second byte has the MP3 frame header pattern (111xxxxx)
                if (second_byte & 0xE0) == 0xE0:  # First 3 bits are 1
                    mp3_found = True
                    break
        
        # Also check for ID3 tag (common in MP3 files)
        id3_found = decoded.startswith(b'ID3')
        
        if not (mp3_found or id3_found):
            return ValidationResult(
                is_valid=False,
                error_code="UNSUPPORTED_FORMAT",
                error_message="Audio format is not supported. Please provide MP3, WAV, or FLAC audio data."
            )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during audio header validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during audio format validation"
        )


def sanitize_base64_input(base64_string: str) -> Optional[str]:
    """
    Sanitize Base64 input by removing whitespace and validating format.
    
    Args:
        base64_string: Raw Base64 string from request
        
    Returns:
        Sanitized Base64 string or None if invalid
    """
    try:
        if not base64_string or not isinstance(base64_string, str):
            return None
        
        # Remove all whitespace characters
        sanitized = re.sub(r'\s+', '', base64_string)
        
        # Remove any potential null bytes or control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Validate the sanitized string
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', sanitized):
            return None
        
        # Ensure length is multiple of 4 (pad if necessary)
        while len(sanitized) % 4 != 0:
            sanitized += '='
        
        # Verify it can be decoded
        try:
            base64.b64decode(sanitized, validate=True)
        except Exception:
            return None
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Error during Base64 sanitization: {e}", exc_info=True)
        return None


def validate_request_security(request_data: dict) -> ValidationResult:
    """
    Perform security validation on request data.
    
    Args:
        request_data: Raw request data dictionary
        
    Returns:
        ValidationResult with security validation status
    """
    start_time = time.time()
    
    try:
        # Timeout protection
        def check_timeout():
            if time.time() - start_time > MAX_VALIDATION_TIME_SECONDS:
                raise TimeoutError("Validation timeout exceeded")
        
        check_timeout()
        
        # Check for suspicious patterns in request
        if not isinstance(request_data, dict):
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_REQUEST",
                error_message="Request must be a JSON object"
            )
        
        check_timeout()
        
        # Check for required fields only
        if 'audio_base64' not in request_data:
            return ValidationResult(
                is_valid=False,
                error_code="MISSING_FIELD",
                error_message="Required field 'audio_base64' is missing"
            )
        
        # Check for unexpected fields (security measure)
        allowed_fields = {'audio_base64'}
        extra_fields = set(request_data.keys()) - allowed_fields
        if extra_fields:
            logger.warning(f"Request contains unexpected fields: {extra_fields}")
            # Don't fail the request, but log for monitoring
        
        check_timeout()
        
        # Check for potential injection attempts in Base64 data
        audio_data = request_data.get('audio_base64', '')
        if isinstance(audio_data, str):
            # For base64 data, only check for very obvious injection attempts
            # Base64 can contain any character sequence by coincidence, so we need to be careful
            obvious_injection_patterns = [
                r'<script[^>]*>.*</script>',  # Complete script tags
                r'javascript:\s*[a-zA-Z]',   # javascript: protocol with actual code
                r'data:text/html',           # HTML data URLs
                r'eval\s*\([^)]+\)',        # eval() function calls
                r'exec\s*\([^)]+\)',        # exec() function calls
                r'system\s*\([^)]+\)',      # system() function calls
            ]
            
            for pattern in obvious_injection_patterns:
                check_timeout()
                if re.search(pattern, audio_data, re.IGNORECASE):
                    logger.warning(f"Obvious injection pattern detected in audio_base64: {pattern}")
                    return ValidationResult(
                        is_valid=False,
                        error_code="SUSPICIOUS_INPUT",
                        error_message="Input contains potentially malicious content"
                    )
            
            # Additional check: if the data looks like it might be HTML/XML/code rather than base64
            # Base64 should mostly be alphanumeric with +, /, and = characters
            non_base64_chars = set(audio_data) - set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r\t ')
            if non_base64_chars:
                suspicious_chars = non_base64_chars & set('<>{}[]|\\^`"\'')
                if len(suspicious_chars) > 2:  # More than 2 suspicious characters
                    logger.warning(f"Multiple suspicious characters in audio_base64: {suspicious_chars}")
                    return ValidationResult(
                        is_valid=False,
                        error_code="SUSPICIOUS_INPUT",
                        error_message="Input contains characters not typical of base64 data"
                    )
        
        # Check for excessively long input (DoS protection)
        if len(str(request_data)) > MAX_REQUEST_SIZE:
            return ValidationResult(
                is_valid=False,
                error_code="REQUEST_TOO_LARGE",
                error_message=f"Request size exceeds maximum allowed limit"
            )
        
        check_timeout()
        
        # Check for repeated patterns that might indicate attack
        if isinstance(audio_data, str) and len(audio_data) > 1000:
            # Check for excessive repetition (potential DoS)
            sample = audio_data[:1000]
            if len(set(sample)) < 10:  # Too few unique characters
                return ValidationResult(
                    is_valid=False,
                    error_code="SUSPICIOUS_INPUT",
                    error_message="Input appears to contain excessive repetition"
                )
        
        return ValidationResult(is_valid=True)
        
    except TimeoutError:
        logger.warning("Validation timeout exceeded - potential DoS attempt")
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_TIMEOUT",
            error_message="Request validation timed out"
        )
        
    except Exception as e:
        logger.error(f"Error during security validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during security validation"
        )


def validate_content_length(content: str, max_length: int = MAX_BASE64_SIZE) -> ValidationResult:
    """
    Validate content length to prevent memory exhaustion attacks.
    
    Args:
        content: Content to validate
        max_length: Maximum allowed length
        
    Returns:
        ValidationResult with length validation status
    """
    try:
        if len(content) > max_length:
            return ValidationResult(
                is_valid=False,
                error_code="CONTENT_TOO_LARGE",
                error_message=f"Content length exceeds maximum allowed size of {max_length} characters"
            )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during content length validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during content length validation"
        )


def validate_base64_entropy(base64_string: str) -> ValidationResult:
    """
    Validate Base64 string entropy to detect potential attacks.
    
    Args:
        base64_string: Base64 string to validate
        
    Returns:
        ValidationResult with entropy validation status
    """
    try:
        if len(base64_string) < 100:
            # Skip entropy check for very short strings
            return ValidationResult(is_valid=True)
        
        # Calculate Shannon entropy
        import math
        from collections import Counter
        
        # Sample first 1000 characters for efficiency
        sample = base64_string[:1000]
        counter = Counter(sample)
        length = len(sample)
        
        entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
        
        # Base64 should have reasonable entropy (not too low, not too high)
        # Low entropy might indicate repeated patterns or padding
        # Very high entropy might indicate encrypted/compressed data
        if entropy < 2.0:
            logger.warning(f"Low entropy detected in Base64 data: {entropy}")
            return ValidationResult(
                is_valid=False,
                error_code="LOW_ENTROPY_INPUT",
                error_message="Input appears to have unusually low entropy"
            )
        
        if entropy > 6.0:
            logger.warning(f"Very high entropy detected in Base64 data: {entropy}")
            # Don't fail for high entropy, but log it
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during entropy validation: {e}", exc_info=True)
        # Don't fail validation if entropy calculation fails
        return ValidationResult(is_valid=True)


def sanitize_and_validate_comprehensive(request_data: dict) -> ValidationResult:
    """
    Comprehensive validation and sanitization pipeline.
    
    Args:
        request_data: Raw request data
        
    Returns:
        ValidationResult with comprehensive validation status
    """
    try:
        # Step 1: Security validation
        security_result = validate_request_security(request_data)
        if not security_result.is_valid:
            return security_result
        
        # Step 2: Extract and validate audio_base64
        audio_base64 = request_data.get('audio_base64', '')
        if not isinstance(audio_base64, str):
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_INPUT_TYPE",
                error_message="audio_base64 must be a string"
            )
        
        # Step 3: Content length validation
        length_result = validate_content_length(audio_base64)
        if not length_result.is_valid:
            return length_result
        
        # Step 4: Base64 format validation
        format_result = validate_base64_format(audio_base64)
        if not format_result.is_valid:
            return format_result
        
        # Step 5: Size validation
        size_result = validate_audio_size(audio_base64)
        if not size_result.is_valid:
            return size_result
        
        # Step 6: Entropy validation
        entropy_result = validate_base64_entropy(audio_base64)
        if not entropy_result.is_valid:
            return entropy_result
        
        # Step 7: Audio header validation
        header_result = validate_audio_header(audio_base64)
        if not header_result.is_valid:
            return header_result
        
        logger.debug("Comprehensive validation completed successfully")
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during comprehensive validation: {e}", exc_info=True)
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during comprehensive validation"
        )


def create_request_fingerprint(request_data: dict) -> str:
    """
    Create a fingerprint for the request for duplicate detection.
    
    Args:
        request_data: Request data dictionary
        
    Returns:
        SHA256 hash of the request content
    """
    try:
        # Create a consistent string representation
        audio_data = request_data.get('audio_base64', '')
        
        # Hash the audio data for fingerprinting
        fingerprint = hashlib.sha256(audio_data.encode('utf-8')).hexdigest()
        
        return fingerprint
        
    except Exception as e:
        logger.error(f"Error creating request fingerprint: {e}", exc_info=True)
        return "unknown"


def validate_request_rate_and_duplicates(
    request_data: dict, 
    client_id: str,
    duplicate_window_seconds: int = 60
) -> ValidationResult:
    """
    Validate request for rate limiting and duplicate detection.
    
    Args:
        request_data: Request data
        client_id: Client identifier
        duplicate_window_seconds: Time window for duplicate detection
        
    Returns:
        ValidationResult with rate and duplicate validation status
    """
    try:
        if os.getenv("RATE_LIMIT_ENABLED", "").lower() in ("false", "0", "no", "off"):
            return ValidationResult(is_valid=True)

        if not get_config().rate_limit.enabled:
            return ValidationResult(is_valid=True)

        # Create request fingerprint
        fingerprint = create_request_fingerprint(request_data)
        current_time = time.time()
        
        logger.debug(
            f"Request fingerprint created",
            extra={
                "client_id": client_id,
                "fingerprint": fingerprint[:16] + "...",  # Log partial fingerprint
                "timestamp": int(current_time)
            }
        )

        global _last_duplicate_cleanup
        with _duplicate_request_lock:
            if current_time - _last_duplicate_cleanup >= _DUPLICATE_TRACKING_CLEANUP_INTERVAL_SECONDS:
                cutoff = current_time - duplicate_window_seconds
                stale_clients = []
                for tracked_client, fingerprints in _recent_request_fingerprints.items():
                    fresh_fingerprints = {
                        tracked_fingerprint: timestamp
                        for tracked_fingerprint, timestamp in fingerprints.items()
                        if timestamp >= cutoff
                    }
                    if fresh_fingerprints:
                        _recent_request_fingerprints[tracked_client] = fresh_fingerprints
                    else:
                        stale_clients.append(tracked_client)
                for stale_client in stale_clients:
                    _recent_request_fingerprints.pop(stale_client, None)
                _last_duplicate_cleanup = current_time

            client_fingerprints = _recent_request_fingerprints.setdefault(client_id, {})
            cutoff = current_time - duplicate_window_seconds
            client_fingerprints = {
                tracked_fingerprint: timestamp
                for tracked_fingerprint, timestamp in client_fingerprints.items()
                if timestamp >= cutoff
            }
            _recent_request_fingerprints[client_id] = client_fingerprints

            if fingerprint in client_fingerprints:
                retry_after = max(1, int(duplicate_window_seconds - (current_time - client_fingerprints[fingerprint])))
                return ValidationResult(
                    is_valid=False,
                    error_code="DUPLICATE_REQUEST",
                    error_message=(
                        f"Duplicate request detected for the same audio payload. "
                        f"Please wait {retry_after} seconds before retrying."
                    )
                )

            client_fingerprints[fingerprint] = current_time
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(f"Error during rate and duplicate validation: {e}", exc_info=True)
        return ValidationResult(is_valid=True)  # Don't fail validation on error
