# AI Voice Detection System - Windows Deployment Script

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "stop", "start", "restart", "logs", "status", "clean")]
    [string]$Command = "deploy"
)

# Configuration
$ImageName = "ai-voice-detector"
$ContainerName = "ai-voice-detector"
$Port = 8000
$EnvFile = ".env"

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Docker is installed and running
function Test-Docker {
    try {
        $null = Get-Command docker -ErrorAction Stop
        $null = docker info 2>$null
        Write-Info "Docker is available and running"
        return $true
    }
    catch {
        Write-Error "Docker is not installed or not running. Please install and start Docker first."
        return $false
    }
}

# Check if environment file exists
function Test-EnvFile {
    if (-not (Test-Path $EnvFile)) {
        Write-Warn "Environment file $EnvFile not found. Creating from template..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" $EnvFile
            Write-Info "Created $EnvFile from template. Please review and update the configuration."
        }
        else {
            Write-Error "No .env.example template found. Please create $EnvFile manually."
            return $false
        }
    }
    else {
        Write-Info "Environment file $EnvFile found"
    }
    return $true
}

# Build Docker image
function Build-Image {
    Write-Info "Building Docker image: $ImageName"
    $result = docker build -t $ImageName .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Docker image built successfully"
        return $true
    }
    else {
        Write-Error "Failed to build Docker image"
        return $false
    }
}

# Stop and remove existing container
function Stop-ExistingContainer {
    $existing = docker ps -a --format "table {{.Names}}" | Select-String "^$ContainerName$"
    if ($existing) {
        Write-Info "Stopping existing container: $ContainerName"
        docker stop $ContainerName
        docker rm $ContainerName
    }
}

# Create models directory if it doesn't exist
function Initialize-Directories {
    Write-Info "Preparing directories..."
    
    if (-not (Test-Path "models")) {
        New-Item -ItemType Directory -Path "models" | Out-Null
    }
    
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" | Out-Null
    }
    
    Write-Info "Directories prepared"
}

# Run the container
function Start-Container {
    Write-Info "Starting container: $ContainerName"
    
    $currentDir = (Get-Location).Path
    $modelsPath = Join-Path $currentDir "models"
    $logsPath = Join-Path $currentDir "logs"
    
    docker run -d `
        --name $ContainerName `
        --env-file $EnvFile `
        -p "${Port}:8000" `
        -v "${modelsPath}:/app/models:ro" `
        -v "${logsPath}:/app/logs" `
        --restart unless-stopped `
        $ImageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Container started successfully"
        Write-Info "Service is available at: http://localhost:$Port"
        Write-Info "Health check: http://localhost:$Port/health"
        Write-Info "API documentation: http://localhost:$Port/docs"
        return $true
    }
    else {
        Write-Error "Failed to start container"
        return $false
    }
}

# Wait for service to be ready
function Wait-ForService {
    Write-Info "Waiting for service to be ready..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Info "Service is ready!"
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Write-Info "Attempt $attempt/$maxAttempts - Service not ready yet, waiting..."
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    Write-Error "Service failed to become ready within expected time"
    Write-Info "Check container logs: docker logs $ContainerName"
    return $false
}

# Show container status and logs
function Show-Status {
    Write-Info "Container status:"
    docker ps --filter "name=$ContainerName" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
    
    Write-Info "Recent logs:"
    docker logs --tail 20 $ContainerName
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting deployment of AI Voice Detection System"
    
    if (-not (Test-Docker)) { exit 1 }
    if (-not (Test-EnvFile)) { exit 1 }
    
    Initialize-Directories
    
    if (-not (Build-Image)) { exit 1 }
    
    Stop-ExistingContainer
    
    if (-not (Start-Container)) { exit 1 }
    
    if (Wait-ForService) {
        Show-Status
        Write-Info "Deployment completed successfully!"
        Write-Info ""
        Write-Info "Service endpoints:"
        Write-Info "  - Health check: http://localhost:$Port/health"
        Write-Info "  - API endpoint: http://localhost:$Port/api/v1/detect-voice"
        Write-Info "  - Documentation: http://localhost:$Port/docs"
        Write-Info ""
        Write-Info "Management commands:"
        Write-Info "  - View logs: docker logs -f $ContainerName"
        Write-Info "  - Stop service: docker stop $ContainerName"
        Write-Info "  - Restart service: docker restart $ContainerName"
    }
    else {
        Write-Error "Deployment failed - service is not responding"
        exit 1
    }
}

# Handle commands
switch ($Command) {
    "deploy" {
        Start-Deployment
    }
    "stop" {
        Write-Info "Stopping AI Voice Detection System"
        docker stop $ContainerName
    }
    "start" {
        Write-Info "Starting AI Voice Detection System"
        docker start $ContainerName
    }
    "restart" {
        Write-Info "Restarting AI Voice Detection System"
        docker restart $ContainerName
    }
    "logs" {
        docker logs -f $ContainerName
    }
    "status" {
        Show-Status
    }
    "clean" {
        Write-Info "Cleaning up AI Voice Detection System"
        docker stop $ContainerName 2>$null
        docker rm $ContainerName 2>$null
        docker rmi $ImageName 2>$null
        Write-Info "Cleanup completed"
    }
}

Write-Info "Script completed"