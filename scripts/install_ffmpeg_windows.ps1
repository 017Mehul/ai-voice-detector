# AI Voice Detection System - FFmpeg Installation Script for Windows
param(
    [string]$InstallPath = "C:\ffmpeg",
    [switch]$Force = $false
)

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-FFmpegInstalled {
    try {
        $result = & ffmpeg -version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "FFmpeg is already installed:"
            Write-Host ($result | Select-Object -First 1)
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

function Install-FFmpeg {
    param([string]$InstallPath)
    
    Write-Info "Installing FFmpeg to: $InstallPath"
    
    if (Test-Path $InstallPath) {
        if ($Force) {
            Write-Warn "Removing existing installation at $InstallPath"
            Remove-Item -Path $InstallPath -Recurse -Force
        } else {
            Write-ErrorMsg "Installation path $InstallPath already exists. Use -Force to overwrite."
            return $false
        }
    }
    
    try {
        New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        Write-Info "Created installation directory: $InstallPath"
    }
    catch {
        Write-ErrorMsg "Failed to create installation directory: $($_.Exception.Message)"
        return $false
    }
    
    $downloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $zipPath = Join-Path $env:TEMP "ffmpeg.zip"
    
    Write-Info "Downloading FFmpeg..."
    Write-Info "This may take a few minutes..."
    
    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($downloadUrl, $zipPath)
        Write-Info "Download completed"
    }
    catch {
        Write-ErrorMsg "Failed to download FFmpeg: $($_.Exception.Message)"
        return $false
    }
    
    Write-Info "Extracting FFmpeg..."
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $InstallPath)
        
        $extractedFolders = Get-ChildItem -Path $InstallPath -Directory
        if ($extractedFolders.Count -eq 1) {
            $ffmpegFolder = $extractedFolders[0].FullName
            $binPath = Join-Path $ffmpegFolder "bin"
            
            if (Test-Path $binPath) {
                $targetBinPath = Join-Path $InstallPath "bin"
                Move-Item -Path $binPath -Destination $targetBinPath -Force
                Remove-Item -Path $ffmpegFolder -Recurse -Force
                Write-Info "FFmpeg extracted successfully"
            } else {
                Write-ErrorMsg "Could not find bin folder in extracted FFmpeg"
                return $false
            }
        } else {
            Write-ErrorMsg "Unexpected extraction structure"
            return $false
        }
    }
    catch {
        Write-ErrorMsg "Failed to extract FFmpeg: $($_.Exception.Message)"
        return $false
    }
    finally {
        if (Test-Path $zipPath) {
            Remove-Item -Path $zipPath -Force
        }
    }
    
    return $true
}

function Add-ToPath {
    param([string]$Path)
    
    $binPath = Join-Path $Path "bin"
    
    if (-not (Test-Path $binPath)) {
        Write-ErrorMsg "FFmpeg bin directory not found: $binPath"
        return $false
    }
    
    Write-Info "Adding FFmpeg to system PATH..."
    
    try {
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        
        if ($currentPath -split ";" | Where-Object { $_ -eq $binPath }) {
            Write-Info "FFmpeg is already in system PATH"
            return $true
        }
        
        $newPath = "$currentPath;$binPath"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
        $env:PATH += ";$binPath"
        
        Write-Info "FFmpeg added to system PATH"
        Write-Warn "You may need to restart your command prompt for PATH changes to take effect"
        
        return $true
    }
    catch {
        Write-ErrorMsg "Failed to add FFmpeg to PATH: $($_.Exception.Message)"
        Write-Info "You can manually add this path to your system PATH: $binPath"
        return $false
    }
}

function Test-Installation {
    Write-Info "Verifying FFmpeg installation..."
    
    $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
    
    try {
        $result = & ffmpeg -version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "✓ FFmpeg installed successfully:"
            Write-Host ($result | Select-Object -First 1)
            
            try {
                $probeResult = & ffprobe -version 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Info "✓ FFprobe is also available"
                }
            }
            catch {
                Write-Warn "FFprobe not found, but FFmpeg should work"
            }
            
            Write-Info ""
            Write-Info "You can now run the AI Voice Detection System!"
            Write-Info "Verify all dependencies with: python scripts/check_dependencies.py"
            
            return $true
        }
    }
    catch {
        Write-ErrorMsg "✗ FFmpeg installation verification failed"
        Write-Info "Please restart your command prompt and try again"
        return $false
    }
    
    return $false
}

# Main execution
Write-Host "AI Voice Detection System - FFmpeg Installation" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-FFmpegInstalled) {
    Write-Info "FFmpeg is already installed and working!"
    exit 0
}

if (-not (Test-Administrator)) {
    Write-Warn "Running without administrator privileges."
    Write-Warn "FFmpeg will be installed, but you may need to manually add it to PATH."
    Write-Info "For automatic PATH setup, run PowerShell as Administrator."
    Write-Host ""
}

Write-Info "FFmpeg not found. Installing..."

if (-not (Install-FFmpeg -InstallPath $InstallPath)) {
    Write-ErrorMsg "FFmpeg installation failed"
    exit 1
}

if (Test-Administrator) {
    Add-ToPath -Path $InstallPath | Out-Null
} else {
    Write-Warn "Cannot modify system PATH without administrator privileges"
    Write-Info "Please add this path to your system PATH manually: $(Join-Path $InstallPath 'bin')"
    Write-Info "Or run this script as Administrator for automatic PATH setup"
}

if (Test-Installation) {
    Write-Info "Installation completed successfully!"
    exit 0
} else {
    Write-ErrorMsg "Installation verification failed"
    Write-Info "Please check the installation and try running: ffmpeg -version"
    exit 1
}