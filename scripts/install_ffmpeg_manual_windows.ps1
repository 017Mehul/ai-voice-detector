# AI Voice Detection System - Manual FFmpeg Installation Guide for Windows
# This script provides step-by-step instructions for manual FFmpeg installation

Write-Host "AI Voice Detection System - Manual FFmpeg Installation Guide" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "MANUAL INSTALLATION STEPS:" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. Download FFmpeg:" -ForegroundColor Green
Write-Host "   - Go to: https://ffmpeg.org/download.html"
Write-Host "   - Click 'Windows' section"
Write-Host "   - Choose 'Windows builds by BtbN' or 'gyan.dev builds'"
Write-Host "   - Download the 'essentials' build (smaller) or 'full' build"
Write-Host ""

Write-Host "2. Extract FFmpeg:" -ForegroundColor Green
Write-Host "   - Extract the downloaded ZIP file"
Write-Host "   - Recommended location: C:\ffmpeg"
Write-Host "   - The folder should contain: bin, doc, presets folders"
Write-Host ""

Write-Host "3. Add to System PATH:" -ForegroundColor Green
Write-Host "   - Open System Properties (Win + R, type 'sysdm.cpl')"
Write-Host "   - Click 'Environment Variables'"
Write-Host "   - Under 'System Variables', find and select 'Path'"
Write-Host "   - Click 'Edit' then 'New'"
Write-Host "   - Add: C:\ffmpeg\bin (or your installation path + \bin)"
Write-Host "   - Click 'OK' to save all dialogs"
Write-Host ""

Write-Host "4. Verify Installation:" -ForegroundColor Green
Write-Host "   - Open a NEW command prompt or PowerShell"
Write-Host "   - Run: ffmpeg -version"
Write-Host "   - You should see FFmpeg version information"
Write-Host ""

Write-Host "ALTERNATIVE METHODS:" -ForegroundColor Yellow
Write-Host ""

Write-Host "Using Chocolatey (if available):" -ForegroundColor Green
Write-Host "   choco install ffmpeg"
Write-Host ""

Write-Host "Using Scoop (if available):" -ForegroundColor Green
Write-Host "   scoop install ffmpeg"
Write-Host ""

Write-Host "Using winget (Windows 10/11):" -ForegroundColor Green
Write-Host "   winget install ffmpeg"
Write-Host ""

Write-Host "AUTOMATED INSTALLATION:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run the automated installer (requires internet connection):" -ForegroundColor Green
Write-Host "   .\scripts\install_ffmpeg_windows.ps1"
Write-Host ""
Write-Host "Or with custom path:" -ForegroundColor Green
Write-Host "   .\scripts\install_ffmpeg_windows.ps1 -InstallPath 'D:\tools\ffmpeg'"
Write-Host ""

Write-Host "TROUBLESHOOTING:" -ForegroundColor Yellow
Write-Host ""
Write-Host "If 'ffmpeg' is not recognized after installation:" -ForegroundColor Green
Write-Host "   - Restart your command prompt/PowerShell"
Write-Host "   - Verify PATH was added correctly"
Write-Host "   - Check that ffmpeg.exe exists in the bin folder"
Write-Host ""

Write-Host "For permission issues:" -ForegroundColor Green
Write-Host "   - Run PowerShell as Administrator"
Write-Host "   - Or install to a user directory (e.g., C:\Users\YourName\ffmpeg)"
Write-Host ""

Write-Host "VERIFICATION:" -ForegroundColor Yellow
Write-Host ""
Write-Host "After installation, verify all dependencies:" -ForegroundColor Green
Write-Host "   python scripts\check_dependencies.py"
Write-Host ""

Write-Host "Start the AI Voice Detection System:" -ForegroundColor Green
Write-Host "   python -m voice_detector.main"
Write-Host ""

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "For more help, see: README.md or DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
