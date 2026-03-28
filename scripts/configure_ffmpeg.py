#!/usr/bin/env python3
"""
FFmpeg Configuration Helper for AI Voice Detection System

This script helps configure FFmpeg paths and validates the installation.
"""

import os
import subprocess
import sys
from pathlib import Path


def find_ffmpeg_executable():
    """Find FFmpeg executable in system PATH or common locations."""
    # Check system PATH first
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'ffmpeg'  # Available in PATH
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check common Windows installation paths
    if sys.platform == 'win32':
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            os.path.expanduser(r'~\ffmpeg\bin\ffmpeg.exe'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, '-version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return path
                except subprocess.TimeoutExpired:
                    continue
    
    return None


def find_ffprobe_executable():
    """Find FFprobe executable in system PATH or common locations."""
    # Check system PATH first
    try:
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'ffprobe'  # Available in PATH
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check common Windows installation paths
    if sys.platform == 'win32':
        common_paths = [
            r'C:\ffmpeg\bin\ffprobe.exe',
            r'C:\Program Files\ffmpeg\bin\ffprobe.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe',
            os.path.expanduser(r'~\ffmpeg\bin\ffprobe.exe'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    result = subprocess.run([path, '-version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return path
                except subprocess.TimeoutExpired:
                    continue
    
    return None


def configure_pydub_ffmpeg():
    """Configure pydub to use the correct FFmpeg paths."""
    ffmpeg_path = find_ffmpeg_executable()
    ffprobe_path = find_ffprobe_executable()
    
    if not ffmpeg_path:
        print("âŒ FFmpeg not found. Please install FFmpeg first.")
        print("\nInstallation options:")
        print("  Windows: winget install ffmpeg")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        return False
    
    if not ffprobe_path:
        print("âš ï¸  FFprobe not found, but FFmpeg should work for basic operations.")
    
    # Set environment variables for pydub
    if ffmpeg_path != 'ffmpeg':  # Not in PATH
        os.environ['FFMPEG_BINARY'] = ffmpeg_path
        print(f"âœ“ Configured FFmpeg path: {ffmpeg_path}")
    else:
        print("âœ“ FFmpeg found in system PATH")
    
    if ffprobe_path and ffprobe_path != 'ffprobe':  # Not in PATH
        os.environ['FFPROBE_BINARY'] = ffprobe_path
        print(f"âœ“ Configured FFprobe path: {ffprobe_path}")
    elif ffprobe_path:
        print("âœ“ FFprobe found in system PATH")
    
    return True


def test_mp3_processing():
    """Test MP3 processing functionality."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        import io
        
        print("\nTesting MP3 processing...")
        
        # Generate a test tone
        tone = Sine(440).to_audio_segment(duration=1000)  # 1 second at 440 Hz
        
        # Export to MP3
        mp3_buffer = io.BytesIO()
        tone.export(mp3_buffer, format='mp3')
        
        # Try to read it back
        mp3_buffer.seek(0)
        loaded_audio = AudioSegment.from_mp3(mp3_buffer)
        
        print("âœ“ MP3 processing test successful!")
        print(f"  - Duration: {len(loaded_audio)} ms")
        print(f"  - Sample rate: {loaded_audio.frame_rate} Hz")
        print(f"  - Channels: {loaded_audio.channels}")
        
        return True
        
    except Exception as e:
        print(f"âŒ MP3 processing test failed: {e}")
        return False


def main():
    """Main configuration function."""
    print("AI Voice Detection System - FFmpeg Configuration")
    print("=" * 50)
    
    # Configure FFmpeg paths
    if not configure_pydub_ffmpeg():
        sys.exit(1)
    
    # Test MP3 processing
    if not test_mp3_processing():
        print("\nâš ï¸  MP3 processing test failed.")
        print("The system will still work with WAV format audio.")
        print("For full MP3 support, please ensure FFmpeg is properly installed.")
    
    print("\n" + "=" * 50)
    print("Configuration completed!")
    print("\nTo verify all dependencies:")
    print("  python scripts/check_dependencies.py")
    print("\nTo start the AI Voice Detection System:")
    print("  python -m voice_detector.main")


if __name__ == "__main__":
    main()
