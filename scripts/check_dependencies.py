#!/usr/bin/env python3
"""
Dependency Check Script for AI Voice Detection System

This script verifies that all required system dependencies are properly installed.
"""

import subprocess
import sys
import importlib
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} âœ“"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+) âœ—"


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from output
            lines = result.stdout.split('\n')
            version_line = next((line for line in lines if line.startswith('ffmpeg version')), '')
            if version_line:
                version = version_line.split()[2]
                return True, f"FFmpeg {version} âœ“"
            else:
                return True, "FFmpeg installed âœ“"
        else:
            return False, "FFmpeg not working properly âœ—"
    except FileNotFoundError:
        return False, "FFmpeg not found âœ—"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out âœ—"
    except Exception as e:
        return False, f"FFmpeg check failed: {str(e)} âœ—"


def check_python_package(package_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package_name} {version} âœ“"
    except ImportError:
        return False, f"{package_name} not installed âœ—"


def check_required_packages() -> List[Tuple[bool, str]]:
    """Check all required Python packages."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'librosa',
        'soundfile',
        'pydub',
        'sklearn',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pytest',
        'hypothesis'
    ]
    
    results = []
    for package in required_packages:
        # Handle package name variations
        import_name = package
        if package == 'sklearn':
            import_name = 'sklearn'
        elif package == 'pydub':
            import_name = 'pydub'
        
        results.append(check_python_package(import_name))
    
    return results


def print_installation_help():
    """Print installation instructions for missing dependencies."""
    print("\n" + "="*60)
    print("INSTALLATION INSTRUCTIONS")
    print("="*60)
    
    print("\nðŸ”§ FFmpeg Installation:")
    print("Ubuntu/Debian:  sudo apt update && sudo apt install ffmpeg")
    print("macOS:          brew install ffmpeg")
    print("Windows:        choco install ffmpeg")
    print("                (or download from https://ffmpeg.org/)")
    
    print("\nðŸ Python Packages:")
    print("pip install -r requirements.txt")
    
    print("\nðŸ“¦ Package Installation:")
    print("pip install -e .")
    
    print("\nðŸ³ Docker Alternative:")
    print("Use Docker deployment to avoid manual dependency installation:")
    print("./scripts/deploy.sh")


def main():
    """Main dependency check function."""
    print("AI Voice Detection System - Dependency Check")
    print("="*50)
    
    all_good = True
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    print(f"Python Version: {python_msg}")
    if not python_ok:
        all_good = False
    
    # Check FFmpeg
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    print(f"FFmpeg:         {ffmpeg_msg}")
    if not ffmpeg_ok:
        all_good = False
    
    # Check Python packages
    print("\nPython Packages:")
    package_results = check_required_packages()
    for package_ok, package_msg in package_results:
        print(f"  {package_msg}")
        if not package_ok:
            all_good = False
    
    # Summary
    print("\n" + "="*50)
    if all_good:
        print("âœ… All dependencies are properly installed!")
        print("You can now run the AI Voice Detection System.")
        print("\nTo start the service:")
        print("  python -m voice_detector.main")
        print("\nTo run tests:")
        print("  pytest")
    else:
        print("âŒ Some dependencies are missing or not properly installed.")
        print_installation_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
