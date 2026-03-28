"""
Property-Based Tests for API Response Structure

Tests Property 10: API Response Structure
Validates: Requirements 4.3, 4.4, 4.5
"""

import pytest
import json
from hypothesis import given, strategies as st, assume, settings
from ai_voice_detector.models.data_models import DetectionResponse, ErrorResponse


# Test data generators
@st.composite
def valid_classification(draw):
    """Generate valid classification values"""
    return draw(st.sampled_from(["AI_GENERATED", "HUMAN"]))


@st.composite
def valid_confidence_score(draw):
    """Generate valid confidence scores between 0.0 and 1.0"""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_detection_response(draw):
    """Generate valid DetectionResponse objects"""
    classification = draw(valid_classification())
    confidence_score = draw(valid_confidence_score())
    
    return DetectionResponse(
        classification=classification,
        confidence_score=confidence_score
    )


@st.composite
def valid_error_response(draw):
    """Generate valid ErrorResponse objects"""
    error_codes = ["INVALID_INPUT", "PROCESSING_ERROR", "INVALID_AUDIO", "UNSUPPORTED_FORMAT"]
    error_code = draw(st.sampled_from(error_codes))
    message = draw(st.text(min_size=1, max_size=200))
    
    return ErrorResponse(error=error_code, message=message)


class TestAPIResponseStructureProperties:
    """Property-based tests for API response structure"""
    
    @given(response=valid_detection_response())
    @settings(max_examples=25)
    def test_api_response_structure_property(self, response):
        """
        **Feature: ai-voice-detector, Property 10: API Response Structure**
        
        For any successful processing request, the API_Gateway should return a JSON response 
        containing exactly "classification" and "confidence_score" fields with correct data 
        types and value constraints.
        
        **Validates: Requirements 4.3, 4.4, 4.5**
        """
        # Convert response to dictionary (simulating JSON serialization)
        response_dict = response.to_dict()
        
        # Property 1: Response must contain exactly the required fields
        expected_fields = {"classification", "confidence_score"}
        actual_fields = set(response_dict.keys())
        assert actual_fields == expected_fields, \
            f"Response must contain exactly {expected_fields}, got {actual_fields}"
        
        # Property 2: Classification field must be exactly "AI_GENERATED" or "HUMAN"
        classification = response_dict["classification"]
        assert classification in ["AI_GENERATED", "HUMAN"], \
            f"Classification must be 'AI_GENERATED' or 'HUMAN', got '{classification}'"
        assert isinstance(classification, str), \
            f"Classification must be string, got {type(classification)}"
        
        # Property 3: Confidence score must be float between 0.0 and 1.0
        confidence_score = response_dict["confidence_score"]
        assert isinstance(confidence_score, (int, float)), \
            f"Confidence score must be numeric, got {type(confidence_score)}"
        assert 0.0 <= confidence_score <= 1.0, \
            f"Confidence score must be between 0.0 and 1.0, got {confidence_score}"
        
        # Property 4: Response should be JSON serializable
        json_str = response.to_json()
        assert isinstance(json_str, str), "Response must be JSON serializable to string"
        
        # Property 5: JSON should be parseable and maintain structure
        parsed_response = json.loads(json_str)
        assert parsed_response == response_dict, \
            "JSON serialization should preserve response structure"
        
        # Property 6: Parsed JSON should maintain data types
        assert isinstance(parsed_response["classification"], str), \
            "Classification should remain string after JSON round-trip"
        assert isinstance(parsed_response["confidence_score"], (int, float)), \
            "Confidence score should remain numeric after JSON round-trip"
    
    @given(
        classification=valid_classification(),
        confidence_score=valid_confidence_score()
    )
    @settings(max_examples=15)
    def test_detection_response_field_constraints(self, classification, confidence_score):
        """
        Property: DetectionResponse fields must satisfy individual constraints
        """
        # Create response
        response = DetectionResponse(
            classification=classification,
            confidence_score=confidence_score
        )
        
        # Verify field constraints are maintained
        assert response.classification == classification
        assert response.confidence_score == confidence_score
        
        # Verify constraints through to_dict
        response_dict = response.to_dict()
        assert response_dict["classification"] == classification
        assert abs(response_dict["confidence_score"] - confidence_score) < 1e-10
    
    @given(error_response=valid_error_response())
    @settings(max_examples=15)
    def test_error_response_structure_property(self, error_response):
        """
        Property: Error responses should have consistent structure
        """
        # Convert to dictionary
        error_dict = error_response.to_dict()
        
        # Property 1: Must contain exactly error and message fields
        expected_fields = {"error", "message"}
        actual_fields = set(error_dict.keys())
        assert actual_fields == expected_fields, \
            f"Error response must contain exactly {expected_fields}, got {actual_fields}"
        
        # Property 2: Both fields must be strings
        assert isinstance(error_dict["error"], str), \
            f"Error code must be string, got {type(error_dict['error'])}"
        assert isinstance(error_dict["message"], str), \
            f"Error message must be string, got {type(error_dict['message'])}"
        
        # Property 3: Fields must not be empty
        assert len(error_dict["error"]) > 0, "Error code must not be empty"
        assert len(error_dict["message"]) > 0, "Error message must not be empty"
        
        # Property 4: Should be JSON serializable
        json_str = error_response.to_json()
        parsed_error = json.loads(json_str)
        assert parsed_error == error_dict, \
            "Error response JSON serialization should preserve structure"
    
    @given(
        classification=st.text().filter(lambda x: x not in ["AI_GENERATED", "HUMAN"]),
        confidence_score=valid_confidence_score()
    )
    @settings(max_examples=8)
    def test_invalid_classification_rejected(self, classification, confidence_score):
        """
        Property: Invalid classification values should be rejected
        """
        assume(classification not in ["AI_GENERATED", "HUMAN"])
        assume(len(classification) > 0)  # Avoid empty strings
        
        # Should raise ValueError for invalid classification
        with pytest.raises(ValueError, match="Invalid classification"):
            DetectionResponse(
                classification=classification,
                confidence_score=confidence_score
            )
    
    @given(
        classification=valid_classification(),
        confidence_score=st.floats().filter(lambda x: not (0.0 <= x <= 1.0) and not (x != x))  # Outside range, not NaN
    )
    @settings(max_examples=8)
    def test_invalid_confidence_score_rejected(self, classification, confidence_score):
        """
        Property: Invalid confidence scores should be rejected
        """
        assume(not (0.0 <= confidence_score <= 1.0))
        assume(confidence_score == confidence_score)  # Not NaN
        
        # Should raise ValueError for out-of-range confidence score
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            DetectionResponse(
                classification=classification,
                confidence_score=confidence_score
            )
    
    @given(response=valid_detection_response())
    @settings(max_examples=10)
    def test_response_immutability_property(self, response):
        """
        Property: Response objects should maintain consistency across multiple serializations
        """
        # Get multiple serializations
        dict1 = response.to_dict()
        dict2 = response.to_dict()
        json1 = response.to_json()
        json2 = response.to_json()
        
        # Should be identical
        assert dict1 == dict2, "Multiple to_dict() calls should return identical results"
        assert json1 == json2, "Multiple to_json() calls should return identical results"
        
        # JSON parsing should yield same dict
        parsed1 = json.loads(json1)
        parsed2 = json.loads(json2)
        assert parsed1 == parsed2 == dict1 == dict2, \
            "All serialization methods should yield consistent results"