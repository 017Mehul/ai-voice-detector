#!/bin/bash
# AI Voice Detection System - FFmpeg Installation Script for Linux/macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null; then
            echo "centos"
        elif command -v dnf &> /dev/null; then
            echo "fedora"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check if FFmpeg is already installed
check_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        log_info "FFmpeg is already installed:"
        ffmpeg -version | head -1
        return 0
    else
        return 1
    fi
}

# Install FFmpeg on Ubuntu/Debian
install_ubuntu() {
    log_info "Installing FFmpeg on Ubuntu/Debian..."
    sudo apt update
    sudo apt install -y ffmpeg libsndfile1
}

# Install FFmpeg on CentOS/RHEL
install_centos() {
    log_info "Installing FFmpeg on CentOS/RHEL..."
    # Enable EPEL repository
    sudo yum install -y epel-release
    sudo yum install -y ffmpeg libsndfile
}

# Install FFmpeg on Fedora
install_fedora() {
    log_info "Installing FFmpeg on Fedora..."
    sudo dnf install -y ffmpeg libsndfile
}

# Install FFmpeg on macOS
install_macos() {
    log_info "Installing FFmpeg on macOS..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg libsndfile
    elif command -v port &> /dev/null; then
        sudo port install ffmpeg libsndfile
    else
        log_error "Neither Homebrew nor MacPorts found."
        log_info "Please install Homebrew first:"
        log_info "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
}

# Main installation function
install_ffmpeg() {
    local os=$(detect_os)
    
    log_info "Detected OS: $os"
    
    case $os in
        "ubuntu")
            install_ubuntu
            ;;
        "centos")
            install_centos
            ;;
        "fedora")
            install_fedora
            ;;
        "macos")
            install_macos
            ;;
        "linux")
            log_error "Unsupported Linux distribution. Please install FFmpeg manually:"
            log_info "  - Check your distribution's package manager"
            log_info "  - Or compile from source: https://ffmpeg.org/download.html"
            exit 1
            ;;
        *)
            log_error "Unsupported operating system: $OSTYPE"
            log_info "Please install FFmpeg manually: https://ffmpeg.org/download.html"
            exit 1
            ;;
    esac
}

# Verify installation
verify_installation() {
    log_info "Verifying FFmpeg installation..."
    
    if command -v ffmpeg &> /dev/null; then
        log_info "✓ FFmpeg installed successfully:"
        ffmpeg -version | head -1
        
        if command -v ffprobe &> /dev/null; then
            log_info "✓ FFprobe is also available"
        else
            log_warn "FFprobe not found, but FFmpeg should work"
        fi
        
        log_info ""
        log_info "You can now run the AI Voice Detection System!"
        log_info "Verify all dependencies with: python scripts/check_dependencies.py"
        
    else
        log_error "✗ FFmpeg installation failed"
        log_info "Please try manual installation or check the error messages above"
        exit 1
    fi
}

# Main script
main() {
    echo "AI Voice Detection System - FFmpeg Installation"
    echo "================================================"
    echo ""
    
    if check_ffmpeg; then
        log_info "FFmpeg is already installed and working!"
        exit 0
    fi
    
    log_info "FFmpeg not found. Installing..."
    
    install_ffmpeg
    verify_installation
    
    log_info "Installation completed successfully!"
}

# Run main function
main "$@"