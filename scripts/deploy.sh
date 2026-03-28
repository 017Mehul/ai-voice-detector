#!/bin/bash
# AI Voice Detection System - Deployment Script

set -e

# Configuration
IMAGE_NAME="ai-voice-detector"
CONTAINER_NAME="ai-voice-detector"
PORT=8000
ENV_FILE=".env"

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

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_info "Docker is available and running"
}

# Check if environment file exists
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_warn "Environment file $ENV_FILE not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example $ENV_FILE
            log_info "Created $ENV_FILE from template. Please review and update the configuration."
        else
            log_error "No .env.example template found. Please create $ENV_FILE manually."
            exit 1
        fi
    else
        log_info "Environment file $ENV_FILE found"
    fi
}

# Build Docker image
build_image() {
    log_info "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Stop and remove existing container
stop_existing_container() {
    if docker ps -a --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        log_info "Stopping existing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
}

# Create models directory if it doesn't exist
prepare_directories() {
    log_info "Preparing directories..."
    mkdir -p models
    mkdir -p logs
    
    # Set proper permissions
    chmod 755 models logs
    
    log_info "Directories prepared"
}

# Run the container
run_container() {
    log_info "Starting container: $CONTAINER_NAME"
    
    docker run -d \
        --name $CONTAINER_NAME \
        --env-file $ENV_FILE \
        -p $PORT:8000 \
        -v "$(pwd)/models:/app/models:ro" \
        -v "$(pwd)/logs:/app/logs" \
        --restart unless-stopped \
        $IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        log_info "Container started successfully"
        log_info "Service is available at: http://localhost:$PORT"
        log_info "Health check: http://localhost:$PORT/health"
        log_info "API documentation: http://localhost:$PORT/docs"
    else
        log_error "Failed to start container"
        exit 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    log_info "Waiting for service to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$PORT/health" > /dev/null; then
            log_info "Service is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - Service not ready yet, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Service failed to become ready within expected time"
    log_info "Check container logs: docker logs $CONTAINER_NAME"
    return 1
}

# Show container status and logs
show_status() {
    log_info "Container status:"
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    log_info "Recent logs:"
    docker logs --tail 20 $CONTAINER_NAME
}

# Main deployment function
deploy() {
    log_info "Starting deployment of AI Voice Detection System"
    
    check_docker
    check_env_file
    prepare_directories
    build_image
    stop_existing_container
    run_container
    
    if wait_for_service; then
        show_status
        log_info "Deployment completed successfully!"
        log_info ""
        log_info "Service endpoints:"
        log_info "  - Health check: http://localhost:$PORT/health"
        log_info "  - API endpoint: http://localhost:$PORT/api/v1/detect-voice"
        log_info "  - Documentation: http://localhost:$PORT/docs"
        log_info ""
        log_info "Management commands:"
        log_info "  - View logs: docker logs -f $CONTAINER_NAME"
        log_info "  - Stop service: docker stop $CONTAINER_NAME"
        log_info "  - Restart service: docker restart $CONTAINER_NAME"
    else
        log_error "Deployment failed - service is not responding"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        log_info "Stopping AI Voice Detection System"
        docker stop $CONTAINER_NAME
        ;;
    "start")
        log_info "Starting AI Voice Detection System"
        docker start $CONTAINER_NAME
        ;;
    "restart")
        log_info "Restarting AI Voice Detection System"
        docker restart $CONTAINER_NAME
        ;;
    "logs")
        docker logs -f $CONTAINER_NAME
        ;;
    "status")
        show_status
        ;;
    "clean")
        log_info "Cleaning up AI Voice Detection System"
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        log_info "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|start|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Build and deploy the service (default)"
        echo "  stop    - Stop the running service"
        echo "  start   - Start the stopped service"
        echo "  restart - Restart the service"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  clean   - Remove container and image"
        exit 1
        ;;
esac