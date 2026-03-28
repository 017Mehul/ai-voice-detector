# AI Voice Detection System - Comprehensive Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start Deployment](#quick-start-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Container Orchestration](#container-orchestration)
7. [Environment Configuration](#environment-configuration)
8. [Security Hardening](#security-hardening)
9. [Monitoring and Observability](#monitoring-and-observability)
10. [Troubleshooting](#troubleshooting)
11. [Performance Tuning](#performance-tuning)
12. [Backup and Recovery](#backup-and-recovery)

## Overview

This comprehensive deployment guide covers all aspects of deploying the AI Voice Detection System in various environments, from development to production. The system is designed to be highly available, scalable, and secure.

### Architecture Overview

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Load Balancer â”‚â”€â”€â”€â”€â”‚  API Gateway    â”‚â”€â”€â”€â”€â”‚  Processing     â”‚
â”‚   (nginx/ALB)   â”‚    â”‚  (FastAPI)      â”‚    â”‚  Pipeline       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                â”‚                       â”‚
                       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                       â”‚   Monitoring    â”‚    â”‚   Model Store   â”‚
                       â”‚ (Prometheus)    â”‚    â”‚   (File/S3)     â”‚
                       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Prerequisites

### System Requirements

#### Minimum Requirements (Development/Testing)
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 2 GB
- **Storage**: 1 GB free space
- **Network**: HTTP/HTTPS access

#### Recommended Requirements (Production)
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8+ GB
- **Storage**: 10+ GB SSD
- **Network**: Load balancer with health checks

#### High-Performance Requirements (Large Scale)
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16+ GB
- **Storage**: 50+ GB NVMe SSD
- **Network**: CDN with edge caching

### Software Dependencies

#### Required
- **Docker**: 20.10+ (recommended deployment method)
- **Python**: 3.10+ (for development/manual deployment)
- **FFmpeg**: For MP3 audio processing

#### Optional
- **Kubernetes**: 1.20+ (for container orchestration)
- **Nginx**: 1.18+ (for load balancing)
- **Prometheus**: 2.30+ (for monitoring)
- **Grafana**: 8.0+ (for visualization)

## Quick Start Deployment

### 1. Docker Deployment (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd ai-voice-detection

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Deploy with Docker
./scripts/deploy.sh

# Verify deployment
curl http://localhost:8000/health
```

### 2. Manual Python Deployment

```bash
# Install system dependencies
sudo apt update && sudo apt install ffmpeg libsndfile1  # Ubuntu
# brew install ffmpeg libsndfile                        # macOS

# Setup Python environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run application
python -m voice_detector.main
```

## Production Deployment

### 1. Docker Compose Production Setup

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  ai-voice-detector:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
      - API_WORKERS=4
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - ai-voice-detector
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

### 2. Production Dockerfile

Create `Dockerfile.prod`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY voice_detector/ ./voice_detector/
COPY setup.py README.md ./

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/config \
    && chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production settings
CMD ["python", "-m", "uvicorn", "voice_detector.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

### 3. Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream voice_detector {
        server ai-voice-detector:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://voice_detector;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://voice_detector;
            proxy_connect_timeout 5s;
            proxy_send_timeout 5s;
            proxy_read_timeout 5s;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## Cloud Deployment

### AWS Deployment

#### 1. ECS Deployment

Create `aws/task-definition.json`:

```json
{
  "family": "ai-voice-detector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ai-voice-detector",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ai-voice-detector:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:ai-voice-detector/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-voice-detector",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 2. CloudFormation Template

Create `aws/cloudformation.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AI Voice Detection System Infrastructure'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID for the deployment
  
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs for the deployment
  
  ImageUri:
    Type: String
    Description: ECR image URI

Resources:
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: ai-voice-detector-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: ai-voice-detector-alb
      Scheme: internet-facing
      Type: application
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Target Group
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: ai-voice-detector-tg
      Port: 8000
      Protocol: HTTP
      VpcId: !Ref VpcId
      TargetType: ip
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: ai-voice-detector-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref ECSSecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: ai-voice-detector
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for ALB
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  ECSSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for ECS tasks
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          SourceSecurityGroupId: !Ref ALBSecurityGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt LoadBalancer.DNSName
```

### Google Cloud Platform Deployment

#### 1. Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-voice-detector

# Deploy to Cloud Run
gcloud run deploy ai-voice-detector \
  --image gcr.io/PROJECT_ID/ai-voice-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars LOG_LEVEL=INFO,ENVIRONMENT=production
```

#### 2. GKE Deployment

Create `gcp/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-voice-detector
  labels:
    app: ai-voice-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-voice-detector
  template:
    metadata:
      labels:
        app: ai-voice-detector
    spec:
      containers:
      - name: ai-voice-detector
        image: gcr.io/PROJECT_ID/ai-voice-detector:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-voice-detector-service
spec:
  selector:
    app: ai-voice-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### 1. Container Instances

```bash
# Create resource group
az group create --name ai-voice-detector-rg --location eastus

# Deploy container
az container create \
  --resource-group ai-voice-detector-rg \
  --name ai-voice-detector \
  --image your-registry.azurecr.io/ai-voice-detector:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables LOG_LEVEL=INFO ENVIRONMENT=production \
  --restart-policy Always
```

## Container Orchestration

### Kubernetes Deployment

#### 1. Complete Kubernetes Manifests

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-voice-detector
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-voice-detector-config
  namespace: ai-voice-detector
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  API_WORKERS: "4"
  MAX_AUDIO_SIZE_MB: "10"
  RATE_LIMIT_REQUESTS: "100"
  RATE_LIMIT_WINDOW: "60"
```

Create `k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-voice-detector-secrets
  namespace: ai-voice-detector
type: Opaque
data:
  # Base64 encoded values
  database-url: <base64-encoded-database-url>
  api-key: <base64-encoded-api-key>
```

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-voice-detector
  namespace: ai-voice-detector
  labels:
    app: ai-voice-detector
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ai-voice-detector
  template:
    metadata:
      labels:
        app: ai-voice-detector
        version: v1
    spec:
      containers:
      - name: ai-voice-detector
        image: ai-voice-detector:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: ai-voice-detector-config
        - secretRef:
            name: ai-voice-detector-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
```

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-voice-detector-service
  namespace: ai-voice-detector
  labels:
    app: ai-voice-detector
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: ai-voice-detector
```

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-voice-detector-ingress
  namespace: ai-voice-detector
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: ai-voice-detector-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-voice-detector-service
            port:
              number: 80
```

Create `k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-voice-detector-hpa
  namespace: ai-voice-detector
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-voice-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 2. Deployment Script

Create `k8s/deploy.sh`:

```bash
#!/bin/bash
set -e

NAMESPACE="ai-voice-detector"
KUBECTL="kubectl"

echo "Deploying AI Voice Detection System to Kubernetes..."

# Apply namespace
$KUBECTL apply -f namespace.yaml

# Apply configurations
$KUBECTL apply -f configmap.yaml
$KUBECTL apply -f secret.yaml

# Apply storage
$KUBECTL apply -f pvc.yaml

# Apply application
$KUBECTL apply -f deployment.yaml
$KUBECTL apply -f service.yaml
$KUBECTL apply -f ingress.yaml
$KUBECTL apply -f hpa.yaml

# Wait for deployment
echo "Waiting for deployment to be ready..."
$KUBECTL wait --for=condition=available --timeout=300s deployment/ai-voice-detector -n $NAMESPACE

# Show status
echo "Deployment completed!"
$KUBECTL get pods -n $NAMESPACE
$KUBECTL get services -n $NAMESPACE
$KUBECTL get ingress -n $NAMESPACE
```

## Environment Configuration

### Production Environment Variables

Create `config/production.env`:

```bash
# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=60

# Model Configuration
MODEL_PATH=/app/models/production/model.joblib
CALIBRATOR_PATH=/app/models/production/calibrator.joblib
MODEL_CONFIDENCE_THRESHOLD=0.5

# Audio Processing
MAX_AUDIO_SIZE_MB=10
MAX_AUDIO_DURATION_SECONDS=30
MIN_AUDIO_DURATION_SECONDS=2
AUDIO_SAMPLE_RATE=22050

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
RATE_LIMIT_STORAGE=memory

# Security
CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com
CORS_METHODS=GET,POST,OPTIONS
CORS_HEADERS=Content-Type,Authorization
CORS_CREDENTIALS=false

# Monitoring
HEALTH_CHECK_ENABLED=true
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn

# Database (if needed)
DATABASE_URL=postgresql://user:password@localhost:5432/voice_detector
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Caching (if needed)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# Feature Flags
ENABLE_DETAILED_LOGGING=false
ENABLE_FEATURE_IMPORTANCE=true
ENABLE_BATCH_PROCESSING=false
ENABLE_ASYNC_PROCESSING=true
```

### Staging Environment Variables

Create `config/staging.env`:

```bash
# Application Configuration
ENVIRONMENT=staging
LOG_LEVEL=DEBUG
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Model Configuration
MODEL_PATH=/app/models/staging/model.joblib
CALIBRATOR_PATH=/app/models/staging/calibrator.joblib

# Rate Limiting (more permissive for testing)
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Security (more permissive for testing)
CORS_ORIGINS=*

# Monitoring
ENABLE_DETAILED_LOGGING=true
ENABLE_FEATURE_IMPORTANCE=true
```

## Security Hardening

### 1. Container Security

Create `security/Dockerfile.secure`:

```dockerfile
FROM python:3.10-slim

# Security updates
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID
RUN groupadd -r -g 1000 app \
    && useradd -r -g app -u 1000 -m -d /app -s /bin/bash app

# Set working directory
WORKDIR /app

# Copy and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app voice_detector/ ./voice_detector/
COPY --chown=app:app setup.py README.md ./

# Install the package
RUN pip install -e .

# Create directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/config /app/tmp \
    && chown -R app:app /app \
    && chmod 755 /app/logs /app/models /app/config \
    && chmod 700 /app/tmp

# Remove unnecessary packages and clean up
RUN apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch to non-root user
USER app

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with security settings
CMD ["python", "-m", "uvicorn", "voice_detector.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--access-log", "--no-server-header"]
```

### 2. Network Security

Create `security/network-policy.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-voice-detector-netpol
  namespace: ai-voice-detector
spec:
  podSelector:
    matchLabels:
      app: ai-voice-detector
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
```

### 3. Security Scanning

Create `security/scan.sh`:

```bash
#!/bin/bash
# Security scanning script

set -e

IMAGE_NAME="ai-voice-detector:latest"

echo "Running security scans for $IMAGE_NAME..."

# Trivy vulnerability scan
echo "Running Trivy vulnerability scan..."
trivy image --severity HIGH,CRITICAL $IMAGE_NAME

# Docker bench security
echo "Running Docker Bench Security..."
docker run --rm --net host --pid host --userns host --cap-add audit_control \
    -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
    -v /etc:/etc:ro \
    -v /usr/bin/containerd:/usr/bin/containerd:ro \
    -v /usr/bin/runc:/usr/bin/runc:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    --label docker_bench_security \
    docker/docker-bench-security

# Hadolint Dockerfile linting
echo "Running Hadolint Dockerfile linting..."
hadolint Dockerfile

echo "Security scans completed!"
```

## Monitoring and Observability

### 1. Prometheus Configuration

Create `monitoring/prometheus-config.yaml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ai-voice-detector'
    environment: 'production'

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'ai-voice-detector'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - ai-voice-detector
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: ai-voice-detector
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
    - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
    - role: node
    relabel_configs:
    - action: labelmap
      regex: __meta_kubernetes_node_label_(.+)

alerting:
  alertmanagers:
  - kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - monitoring
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: alertmanager
```

### 2. Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
- name: ai-voice-detector
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }} seconds"

  - alert: PodCrashLooping
    expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Pod is crash looping"
      description: "Pod {{ $labels.pod }} is restarting frequently"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) / container_spec_cpu_quota * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}%"
```

### 3. Grafana Dashboard

Create `monitoring/grafana-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "AI Voice Detection System",
    "tags": ["ai", "voice", "detection"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          },
          {
            "expr": "rate(http_requests_total{status=~\"4..\"}[5m])",
            "legendFormat": "4xx errors"
          }
        ]
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes",
            "legendFormat": "{{pod}}"
          }
        ],
        "yAxes": [
          {
            "label": "Bytes"
          }
        ]
      },
      {
        "id": 5,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m])",
            "legendFormat": "{{pod}}"
          }
        ],
        "yAxes": [
          {
            "label": "CPU cores"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Won't Start

**Symptoms:**
- Container exits immediately
- Health checks fail
- No response from API

**Diagnosis:**
```bash
# Check container logs
docker logs ai-voice-detector

# Check container status
docker ps -a

# Inspect container configuration
docker inspect ai-voice-detector
```

**Common Causes and Solutions:**

1. **Port already in use:**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process or use different port
   docker run -p 8001:8000 ai-voice-detector
   ```

2. **Missing environment variables:**
   ```bash
   # Check required environment variables
   docker run --env-file .env ai-voice-detector
   ```

3. **Model file not found:**
   ```bash
   # Ensure model directory is mounted
   docker run -v $(pwd)/models:/app/models ai-voice-detector
   ```

#### 2. High Memory Usage

**Symptoms:**
- Container killed by OOM killer
- Slow response times
- Memory warnings in logs

**Diagnosis:**
```bash
# Monitor memory usage
docker stats ai-voice-detector

# Check memory limits
docker inspect ai-voice-detector | grep -i memory
```

**Solutions:**
1. **Increase memory limits:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

2. **Optimize model loading:**
   ```python
   # Use memory mapping for large models
   model = joblib.load('model.joblib', mmap_mode='r')
   ```

3. **Enable garbage collection:**
   ```python
   import gc
   gc.collect()
   ```

#### 3. Audio Processing Errors

**Symptoms:**
- "FFmpeg not found" errors
- MP3 conversion failures
- Audio format errors

**Diagnosis:**
```bash
# Check FFmpeg installation
docker exec ai-voice-detector ffmpeg -version

# Test audio processing
python scripts/check_dependencies.py
```

**Solutions:**
1. **Install FFmpeg in container:**
   ```dockerfile
   RUN apt-get update && apt-get install -y ffmpeg
   ```

2. **Use WAV format instead of MP3:**
   ```python
   # Convert MP3 to WAV before Base64 encoding
   ```

3. **Mount FFmpeg binary:**
   ```bash
   docker run -v /usr/bin/ffmpeg:/usr/bin/ffmpeg ai-voice-detector
   ```

#### 4. Performance Issues

**Symptoms:**
- Slow response times
- High CPU usage
- Request timeouts

**Diagnosis:**
```bash
# Monitor performance
docker stats ai-voice-detector

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

**Solutions:**
1. **Scale horizontally:**
   ```bash
   docker-compose up --scale ai-voice-detector=3
   ```

2. **Optimize model inference:**
   ```python
   # Use batch processing
   # Cache model predictions
   # Use faster model variants
   ```

3. **Add caching:**
   ```python
   # Cache feature extraction results
   # Use Redis for response caching
   ```

### Debugging Tools

#### 1. Health Check Script

Create `scripts/health_check.sh`:

```bash
#!/bin/bash
# Comprehensive health check script

set -e

HOST=${1:-localhost}
PORT=${2:-8000}
TIMEOUT=${3:-30}

echo "Running health checks for $HOST:$PORT..."

# Basic connectivity
echo "1. Testing basic connectivity..."
if curl -f -s --connect-timeout $TIMEOUT "http://$HOST:$PORT/health" > /dev/null; then
    echo "âœ“ Basic connectivity OK"
else
    echo "âœ— Basic connectivity FAILED"
    exit 1
fi

# Detailed health check
echo "2. Testing detailed health check..."
HEALTH_RESPONSE=$(curl -s "http://$HOST:$PORT/health/detailed")
echo "Health response: $HEALTH_RESPONSE"

# API functionality
echo "3. Testing API functionality..."
# Create a small test audio file (Base64 encoded silence)
TEST_AUDIO="UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."
API_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"audio_base64\": \"$TEST_AUDIO\"}" \
    "http://$HOST:$PORT/api/v1/detect-voice")

if echo "$API_RESPONSE" | grep -q "classification"; then
    echo "âœ“ API functionality OK"
    echo "API response: $API_RESPONSE"
else
    echo "âœ— API functionality FAILED"
    echo "Response: $API_RESPONSE"
    exit 1
fi

# Performance test
echo "4. Testing performance..."
START_TIME=$(date +%s.%N)
curl -s "http://$HOST:$PORT/health" > /dev/null
END_TIME=$(date +%s.%N)
RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc)

if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
    echo "âœ“ Performance OK (${RESPONSE_TIME}s)"
else
    echo "âš  Performance slow (${RESPONSE_TIME}s)"
fi

echo "All health checks completed successfully!"
```

#### 2. Load Testing Script

Create `scripts/load_test.sh`:

```bash
#!/bin/bash
# Load testing script using Apache Bench

HOST=${1:-localhost}
PORT=${2:-8000}
REQUESTS=${3:-100}
CONCURRENCY=${4:-10}

echo "Running load test: $REQUESTS requests with $CONCURRENCY concurrent connections"

# Test health endpoint
echo "Testing health endpoint..."
ab -n $REQUESTS -c $CONCURRENCY "http://$HOST:$PORT/health"

# Test API endpoint (requires test data file)
if [ -f "test_audio.json" ]; then
    echo "Testing API endpoint..."
    ab -n $REQUESTS -c $CONCURRENCY -p test_audio.json -T application/json \
        "http://$HOST:$PORT/api/v1/detect-voice"
else
    echo "Skipping API load test (test_audio.json not found)"
fi
```

## Performance Tuning

### 1. Application Performance

#### Model Optimization

```python
# voice_detector/models/optimized_classifier.py
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class OptimizedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path, use_mmap=True):
        self.model_path = model_path
        self.use_mmap = use_mmap
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            # Use memory mapping for large models
            mmap_mode = 'r' if self.use_mmap else None
            self._model = joblib.load(self.model_path, mmap_mode=mmap_mode)
        return self._model
    
    def predict(self, X):
        # Batch processing for better performance
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)
```

#### Caching Layer

```python
# voice_detector/utils/cache.py
import hashlib
import json
from functools import wraps
from typing import Any, Dict, Optional
import redis

class CacheManager:
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self.ttl = ttl
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = None
        self.memory_cache: Dict[str, Any] = {}
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception:
                pass
        return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, json.dumps(value))
                return
            except Exception:
                pass
        self.memory_cache[key] = value
    
    def cache_result(self, ttl: Optional[int] = None):
        """Decorator to cache function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self._get_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result)
                return result
            
            return wrapper
        return decorator

# Global cache instance
cache_manager = CacheManager()
```

#### Async Processing

```python
# voice_detector/api/async_endpoints.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks
from typing import Dict, Any

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

async def process_audio_async(audio_data: bytes) -> Dict[str, Any]:
    """Process audio asynchronously."""
    loop = asyncio.get_event_loop()
    
    # Run CPU-intensive tasks in thread pool
    result = await loop.run_in_executor(
        executor,
        process_audio_sync,
        audio_data
    )
    
    return result

def process_audio_sync(audio_data: bytes) -> Dict[str, Any]:
    """Synchronous audio processing."""
    # Your existing audio processing logic
    pass

@app.post("/api/v1/detect-voice-async")
async def detect_voice_async(request: DetectionRequest):
    """Async version of voice detection endpoint."""
    audio_data = decode_base64_audio(request.audio_base64)
    result = await process_audio_async(audio_data)
    return result
```

### 2. Infrastructure Performance

#### Load Balancer Configuration

Create `nginx/performance.conf`:

```nginx
# Performance-optimized Nginx configuration
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    
    # Buffer settings
    client_body_buffer_size 128k;
    client_max_body_size 10m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    output_buffers 1 32k;
    postpone_output 1460;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g inactive=60m use_temp_path=off;
    
    upstream voice_detector {
        least_conn;
        server ai-voice-detector-1:8000 max_fails=3 fail_timeout=30s;
        server ai-voice-detector-2:8000 max_fails=3 fail_timeout=30s;
        server ai-voice-detector-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    server {
        listen 80;
        
        # Connection settings
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        
        # Health check caching
        location /health {
            proxy_pass http://voice_detector;
            proxy_cache api_cache;
            proxy_cache_valid 200 30s;
            proxy_cache_key "$scheme$request_method$host$request_uri";
        }
        
        # API endpoints
        location /api/ {
            proxy_pass http://voice_detector;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_http_version 1.1;
            
            # No caching for API endpoints
            proxy_cache off;
        }
    }
}
```

#### Database Optimization (if using database)

Create `database/optimize.sql`:

```sql
-- Database optimization for AI Voice Detection System

-- Create indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_timestamp 
ON requests (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_status 
ON requests (status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_user_id 
ON requests (user_id);

-- Optimize table settings
ALTER TABLE requests SET (
    fillfactor = 90,
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- Create materialized view for analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS request_stats AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE status = 'success') as successful_requests,
    COUNT(*) FILTER (WHERE status = 'error') as failed_requests,
    AVG(processing_time) as avg_processing_time
FROM requests 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_request_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY request_stats;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to refresh stats (requires pg_cron extension)
-- SELECT cron.schedule('refresh-stats', '0 * * * *', 'SELECT refresh_request_stats();');
```

## Backup and Recovery

### 1. Data Backup Strategy

Create `backup/backup.sh`:

```bash
#!/bin/bash
# Comprehensive backup script

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Configuration
MODEL_DIR="/app/models"
CONFIG_DIR="/app/config"
LOGS_DIR="/app/logs"
DATABASE_URL=${DATABASE_URL:-""}

echo "Starting backup process at $(date)"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models
if [ -d "$MODEL_DIR" ]; then
    echo "Backing up models..."
    tar -czf "$BACKUP_DIR/$DATE/models.tar.gz" -C "$MODEL_DIR" .
fi

# Backup configuration
if [ -d "$CONFIG_DIR" ]; then
    echo "Backing up configuration..."
    tar -czf "$BACKUP_DIR/$DATE/config.tar.gz" -C "$CONFIG_DIR" .
fi

# Backup logs (last 7 days)
if [ -d "$LOGS_DIR" ]; then
    echo "Backing up recent logs..."
    find "$LOGS_DIR" -name "*.log" -mtime -7 -exec tar -czf "$BACKUP_DIR/$DATE/logs.tar.gz" {} +
fi

# Backup database (if configured)
if [ -n "$DATABASE_URL" ]; then
    echo "Backing up database..."
    pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/$DATE/database.sql.gz"
fi

# Create backup manifest
cat > "$BACKUP_DIR/$DATE/manifest.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "files": [
        $([ -f "$BACKUP_DIR/$DATE/models.tar.gz" ] && echo '"models.tar.gz",')
        $([ -f "$BACKUP_DIR/$DATE/config.tar.gz" ] && echo '"config.tar.gz",')
        $([ -f "$BACKUP_DIR/$DATE/logs.tar.gz" ] && echo '"logs.tar.gz",')
        $([ -f "$BACKUP_DIR/$DATE/database.sql.gz" ] && echo '"database.sql.gz",')
        "manifest.json"
    ]
}
EOF

# Clean up old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -type d -name "20*" -mtime +$RETENTION_DAYS -exec rm -rf {} +

# Upload to cloud storage (optional)
if [ -n "$AWS_S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 sync "$BACKUP_DIR/$DATE" "s3://$AWS_S3_BUCKET/backups/$DATE/"
fi

echo "Backup completed successfully at $(date)"
echo "Backup location: $BACKUP_DIR/$DATE"
```

### 2. Disaster Recovery Plan

Create `recovery/disaster_recovery.md`:

```markdown
# Disaster Recovery Plan

## Recovery Time Objectives (RTO)
- **Critical**: 15 minutes
- **High**: 1 hour
- **Medium**: 4 hours
- **Low**: 24 hours

## Recovery Point Objectives (RPO)
- **Database**: 5 minutes
- **Models**: 1 hour
- **Configuration**: 1 hour
- **Logs**: 24 hours

## Recovery Procedures

### 1. Complete System Failure

1. **Assess the situation**
   ```bash
   # Check system status
   kubectl get pods -n ai-voice-detector
   docker ps
   ```

2. **Restore from backup**
   ```bash
   # Download latest backup
   aws s3 sync s3://backup-bucket/backups/latest/ ./restore/
   
   # Restore models
   tar -xzf restore/models.tar.gz -C /app/models/
   
   # Restore configuration
   tar -xzf restore/config.tar.gz -C /app/config/
   
   # Restore database
   gunzip -c restore/database.sql.gz | psql $DATABASE_URL
   ```

3. **Redeploy application**
   ```bash
   # Kubernetes
   kubectl apply -f k8s/
   
   # Docker Compose
   docker-compose up -d
   ```

### 2. Database Failure

1. **Switch to read replica**
   ```bash
   # Update DATABASE_URL to point to replica
   kubectl set env deployment/ai-voice-detector DATABASE_URL=$REPLICA_URL
   ```

2. **Restore primary database**
   ```bash
   # Restore from backup
   pg_restore -d $DATABASE_URL backup/database.sql.gz
   ```

3. **Switch back to primary**
   ```bash
   kubectl set env deployment/ai-voice-detector DATABASE_URL=$PRIMARY_URL
   ```

### 3. Model Corruption

1. **Validate model integrity**
   ```python
   import joblib
   try:
       model = joblib.load('model.joblib')
       print("Model loaded successfully")
   except Exception as e:
       print(f"Model corrupted: {e}")
   ```

2. **Restore from backup**
   ```bash
   # Download model backup
   aws s3 cp s3://backup-bucket/models/latest/model.joblib ./models/
   
   # Restart application
   kubectl rollout restart deployment/ai-voice-detector
   ```

## Testing Recovery Procedures

### Monthly DR Test

```bash
#!/bin/bash
# Monthly disaster recovery test

echo "Starting DR test at $(date)"

# 1. Create test environment
kubectl create namespace ai-voice-detector-dr-test

# 2. Deploy from backup
kubectl apply -f k8s/ -n ai-voice-detector-dr-test

# 3. Run health checks
./scripts/health_check.sh localhost 8000

# 4. Run functional tests
pytest tests/integration/ --env=dr-test

# 5. Measure recovery time
echo "DR test completed at $(date)"

# 6. Cleanup
kubectl delete namespace ai-voice-detector-dr-test
```
```

This comprehensive deployment documentation covers all aspects of deploying the AI Voice Detection System from development to production, including cloud deployments, security hardening, monitoring, troubleshooting, performance tuning, and disaster recovery. The documentation is structured to be practical and actionable for different deployment scenarios.

<function_calls>
<invoke name="fsWrite">
<parameter name="path">docs/ENVIRONMENT_SETUP.md
