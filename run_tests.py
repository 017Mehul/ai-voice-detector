#!/usr/bin/env python3
"""
Quick test runner for the voice detection system
"""

import os
import sys
import subprocess
from pathlib import Path

def run_basic_tests():
    """Run basic functionality tests"""
    print("Running basic tests...")
    
    # Test model loading
    try:
        import joblib
        model_path = Path("models/production/model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            print("✓ Model loads successfully")
        else:
            print("✗ Model file not found")
            return False
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    # Test feature extraction
    try:
        from voice_detector.features.extractor import FeatureExtractor
        extractor = FeatureExtractor()
        print("✓ Feature extractor initialized")
    except Exception as e:
        print(f"✗ Feature extractor failed: {e}")
        return False
    
    return True

def run_pytest():
    """Run pytest if available"""
    try:
        result = subprocess.run(['pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Some tests failed")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("pytest not available, skipping")
        return True

def main():
    print("Voice Detection System - Test Runner")
    print("=" * 40)
    
    success = True
    
    # Basic tests
    if not run_basic_tests():
        success = False
    
    # Pytest
    if not run_pytest():
        success = False
    
    print("=" * 40)
    if success:
        print("All tests completed successfully!")
        return 0
    else:
        print("Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())