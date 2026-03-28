@echo off
REM AI Voice Detection System - FFmpeg Installation Script for Windows
REM This script attempts to install FFmpeg using winget (preferred) or Chocolatey

echo AI Voice Detection System - FFmpeg Installation
echo ==================================================

REM Check if winget is available (Windows 10/11)
where winget >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Winget found. Installing FFmpeg using winget...
    winget install ffmpeg
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✓ FFmpeg installation completed successfully!
        echo ✓ Please restart your command prompt to use FFmpeg.
        echo.
        echo You can verify the installation by running:
        echo   python scripts/check_dependencies.py
        echo.
        pause
        exit /b 0
    ) else (
        echo.
        echo ✗ Winget installation failed. Trying alternative methods...
    )
)

REM Fallback to Chocolatey if winget fails or is not available
echo Checking for Chocolatey...
where choco >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Chocolatey is not installed.
    echo.
    echo INSTALLATION OPTIONS:
    echo 1. Install Chocolatey and run this script again
    echo 2. Use manual installation
    echo.
    echo To install Chocolatey, run this command in an Administrator PowerShell:
    echo Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    echo.
    echo For manual installation:
    echo 1. Download FFmpeg from https://ffmpeg.org/download.html
    echo 2. Extract to C:\ffmpeg
    echo 3. Add C:\ffmpeg\bin to your system PATH
    echo.
    pause
    exit /b 1
)

echo Chocolatey found. Installing FFmpeg...
choco install ffmpeg -y

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ FFmpeg installation completed successfully!
    echo.
    echo Verifying installation...
    ffmpeg -version
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✓ FFmpeg is working correctly!
        echo You can now run the AI Voice Detection System.
    ) else (
        echo.
        echo ✗ FFmpeg installation may have issues. Please restart your command prompt and try again.
    )
) else (
    echo.
    echo ✗ FFmpeg installation failed. Please try manual installation:
    echo 1. Download FFmpeg from https://ffmpeg.org/download.html
    echo 2. Extract to C:\ffmpeg
    echo 3. Add C:\ffmpeg\bin to your system PATH
)

echo.
echo You can verify the installation by running:
echo   python scripts/check_dependencies.py
echo.
pause