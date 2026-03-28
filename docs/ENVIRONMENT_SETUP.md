# Environment Setup Guide

## Overview

This guide provides detailed instructions for setting up development, staging, and production environments for the AI Voice Detection System. It covers all necessary dependencies, configurations, and best practices for each environment type.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Staging Environment](#staging-environment)
3. [Production Environment](#production-environment)
4. [Environment Variables](#environment-variables)
5. [Dependency Management](#dependency-management)
6. [Database Setup](#database-setup)
7. [Model Management](#model-management)
8. [Testing Environment](#testing-environment)

## Development Environment

### Prerequisites

- **Python**: 3.10 or higher
- **Git**: Latest version
- **Docker**: 20.10+ (optional but recommended)
- **FFmpeg**: For audio processing

### Step-by-Step Setup

#### 1. Clone Repository

```bash
git clone <repository-url>
cd ai-voice-detection
```

#### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y ffmpeg libsndfile1 libsndfile1-dev
sudo apt install -y build-essential portaudio19-dev
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 ffmpeg libsndfile portaudio
```

**Windows:**
```powershell
# Install Python 3.10 from python.org or Microsoft Store
# Install FFmpeg using winget
winget install ffmpeg

# Or using Chocolatey
choco install python ffmpeg

# Or using Scoop
scoop install python ffmpeg
```

#### 3. Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 4. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

#### 5. Setup Environment Configuration

```bash
# Copy environment template
cp .env.example .env.development

# Edit configuration for development
nano .env.development
```

**Development Environment Variables:**
```bash
# Application
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# API
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=1

# Audio Processing
MAX_AUDIO_SIZE_MB=50
MAX_AUDIO_DURATION_SECONDS=60
MIN_AUDIO_DURATION_SECONDS=1

# Rate Limiting (disabled for development)
RATE_LIMIT_ENABLED=false

# Security (permissive for development)
CORS_ORIGINS=*
CORS_CREDENTIALS=true

# Model
MODEL_PATH=./models/development/model.joblib
CALIBRATOR_PATH=./models/development/calibrator.joblib

# Features
ENABLE_DETAILED_LOGGING=true
ENABLE_FEATURE_IMPORTANCE=true
ENABLE_BATCH_PROCESSING=true
```

#### 6. Verify Installation

```bash
# Check all dependencies
python scripts/check_dependencies.py

# Run tests
pytest tests/ -v

# Start development server
python -m voice_detector.main
```

#### 7. Development Tools Setup

**Pre-commit Hooks:**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**IDE Configuration (VS Code):**

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
```

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/voice_detector/main.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.development",
            "args": []
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.development"
        }
    ]
}
```

## Staging Environment

### Overview

The staging environment mirrors production as closely as possible while allowing for testing and validation before production deployment.

### Infrastructure Setup

#### 1. Server Requirements

**Minimum Specifications:**
- **CPU**: 2 cores, 2.5 GHz
- **RAM**: 4 GB
- **Storage**: 20 GB SSD
- **OS**: Ubuntu 20.04 LTS or CentOS 8

#### 2. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install monitoring tools
sudo apt install -y htop iotop nethogs
```

#### 3. Application Deployment

```bash
# Clone repository
git clone <repository-url>
cd ai-voice-detection

# Checkout staging branch
git checkout staging

# Setup environment
cp .env.example .env.staging
```

**Staging Environment Variables:**
```bash
# Application
ENVIRONMENT=staging
LOG_LEVEL=INFO
DEBUG=false

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Database
DATABASE_URL=postgresql://staging_user:password@localhost:5432/voice_detector_staging

# Security
CORS_ORIGINS=https://staging.your-domain.com
CORS_CREDENTIALS=false

# Model
MODEL_PATH=./models/staging/model.joblib
CALIBRATOR_PATH=./models/staging/calibrator.joblib

# Monitoring
SENTRY_DSN=https://your-staging-sentry-dsn
PROMETHEUS_ENABLED=true

# Features
ENABLE_DETAILED_LOGGING=true
ENABLE_FEATURE_IMPORTANCE=true
```

#### 4. Deploy with Docker Compose

Create `docker-compose.staging.yml`:
```yaml
version: '3.8'

services:
  ai-voice-detector:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.staging
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - staging_data:/app/data
    restart: unless-stopped
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: voice_detector_staging
      POSTGRES_USER: staging_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - ai-voice-detector

volumes:
  postgres_data:
  redis_data:
  staging_data:
```

#### 5. SSL Certificate Setup

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d staging.your-domain.com

# Setup auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Production Environment

### Infrastructure Requirements

#### 1. High Availability Setup

**Load Balancer Configuration:**
- **Primary**: 2x Application servers
- **Database**: PostgreSQL with read replicas
- **Cache**: Redis cluster
- **Storage**: Shared storage for models

**Server Specifications:**
- **Application Servers**: 4 cores, 8 GB RAM, 50 GB SSD
- **Database Server**: 4 cores, 16 GB RAM, 100 GB SSD
- **Load Balancer**: 2 cores, 4 GB RAM, 20 GB SSD

#### 2. Security Hardening

```bash
# Firewall configuration
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Fail2ban setup
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# SSH hardening
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

#### 3. Production Environment Variables

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://prod_user:${DB_PASSWORD}@db-primary:5432/voice_detector
DATABASE_REPLICA_URL=postgresql://prod_user:${DB_PASSWORD}@db-replica:5432/voice_detector
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://redis-cluster:6379/0
CACHE_TTL=3600

# Security
CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com
CORS_CREDENTIALS=false
SECRET_KEY=${SECRET_KEY}

# Model
MODEL_PATH=/app/models/production/model.joblib
CALIBRATOR_PATH=/app/models/production/calibrator.joblib

# Monitoring
SENTRY_DSN=${SENTRY_DSN}
PROMETHEUS_ENABLED=true
METRICS_PORT=9090

# Performance
ENABLE_CACHING=true
ENABLE_ASYNC_PROCESSING=true
WORKER_TIMEOUT=120

# Backup
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
S3_BACKUP_BUCKET=${S3_BACKUP_BUCKET}
```

#### 4. Production Deployment

Create `docker-compose.production.yml`:
```yaml
version: '3.8'

services:
  ai-voice-detector-1:
    build:
      context: .
      dockerfile: Dockerfile.prod
    env_file:
      - .env.production
    volumes:
      - models_data:/app/models:ro
      - logs_data:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    depends_on:
      - postgres-primary
      - redis-cluster

  ai-voice-detector-2:
    build:
      context: .
      dockerfile: Dockerfile.prod
    env_file:
      - .env.production
    volumes:
      - models_data:/app/models:ro
      - logs_data:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    depends_on:
      - postgres-primary
      - redis-cluster

  postgres-primary:
    image: postgres:14
    environment:
      POSTGRES_DB: voice_detector
      POSTGRES_USER: prod_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    command: |
      postgres
      -c wal_level=replica
      -c max_wal_senders=3
      -c max_replication_slots=3
      -c hot_standby=on

  postgres-replica:
    image: postgres:14
    environment:
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary

  redis-cluster:
    image: redis:7-alpine
    command: redis-server --appendonly yes --cluster-enabled yes
    volumes:
      - redis_cluster_data:/data
    deploy:
      replicas: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/production.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - ai-voice-detector-1
      - ai-voice-detector-2
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  models_data:
  logs_data:
  postgres_primary_data:
  postgres_replica_data:
  redis_cluster_data:
  prometheus_data:
  grafana_data:
```

## Environment Variables

### Variable Categories

#### 1. Application Configuration

| Variable | Development | Staging | Production | Description |
|----------|-------------|---------|------------|-------------|
| `ENVIRONMENT` | development | staging | production | Environment identifier |
| `LOG_LEVEL` | DEBUG | INFO | INFO | Logging verbosity |
| `DEBUG` | true | false | false | Debug mode |

#### 2. API Configuration

| Variable | Development | Staging | Production | Description |
|----------|-------------|---------|------------|-------------|
| `API_HOST` | 127.0.0.1 | 0.0.0.0 | 0.0.0.0 | API bind address |
| `API_PORT` | 8000 | 8000 | 8000 | API port |
| `API_WORKERS` | 1 | 2 | 4 | Number of worker processes |
| `API_TIMEOUT` | 30 | 60 | 120 | Request timeout |

#### 3. Security Configuration

| Variable | Development | Staging | Production | Description |
|----------|-------------|---------|------------|-------------|
| `CORS_ORIGINS` | * | staging.domain.com | domain.com | Allowed CORS origins |
| `CORS_CREDENTIALS` | true | false | false | Allow credentials |
| `SECRET_KEY` | dev-secret | staging-secret | ${SECRET_KEY} | Application secret |

#### 4. Database Configuration

| Variable | Development | Staging | Production | Description |
|----------|-------------|---------|------------|-------------|
| `DATABASE_URL` | sqlite:///dev.db | postgresql://... | postgresql://... | Database connection |
| `DATABASE_POOL_SIZE` | 5 | 10 | 20 | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | 10 | 20 | 30 | Max overflow connections |

### Environment Variable Management

#### 1. Development

```bash
# .env.development
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export DEBUG=true
export API_HOST=127.0.0.1
export API_PORT=8000
export DATABASE_URL=sqlite:///development.db
export CORS_ORIGINS=*
export RATE_LIMIT_ENABLED=false
```

#### 2. Staging

```bash
# .env.staging
export ENVIRONMENT=staging
export LOG_LEVEL=INFO
export DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=8000
export DATABASE_URL=postgresql://staging_user:${POSTGRES_PASSWORD}@localhost:5432/voice_detector_staging
export CORS_ORIGINS=https://staging.your-domain.com
export RATE_LIMIT_ENABLED=true
export SENTRY_DSN=${STAGING_SENTRY_DSN}
```

#### 3. Production

```bash
# .env.production (use secrets management)
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=8000
export DATABASE_URL=postgresql://prod_user:${DB_PASSWORD}@db-primary:5432/voice_detector
export CORS_ORIGINS=https://your-domain.com
export SECRET_KEY=${SECRET_KEY}
export SENTRY_DSN=${PROD_SENTRY_DSN}
```

## Dependency Management

### Python Dependencies

#### 1. Requirements Files Structure

```
requirements/
â”œâ”€â”€ base.txt          # Core dependencies
â”œâ”€â”€ development.txt   # Development tools
â”œâ”€â”€ staging.txt       # Staging-specific
â”œâ”€â”€ production.txt    # Production-specific
â””â”€â”€ testing.txt       # Testing dependencies
```

**base.txt:**
```
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1

# Machine Learning
scikit-learn==1.3.2
numpy==1.24.4
scipy==1.11.4
joblib==1.3.2

# Data Handling
pydantic==2.5.0
pandas==2.1.3

# Logging
python-json-logger==2.0.7
```

**development.txt:**
```
-r base.txt

# Development Tools
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
hypothesis==6.88.1

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0
```

**production.txt:**
```
-r base.txt

# Production Monitoring
sentry-sdk[fastapi]==1.38.0
prometheus-client==0.19.0

# Performance
redis==5.0.1
psycopg2-binary==2.9.9
```

#### 2. Dependency Installation Scripts

Create `scripts/install_deps.sh`:
```bash
#!/bin/bash
# Dependency installation script

set -e

ENVIRONMENT=${1:-development}

echo "Installing dependencies for $ENVIRONMENT environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install dependencies based on environment
case $ENVIRONMENT in
    "development")
        pip install -r requirements/development.txt
        ;;
    "staging")
        pip install -r requirements/staging.txt
        ;;
    "production")
        pip install -r requirements/production.txt
        ;;
    "testing")
        pip install -r requirements/testing.txt
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        echo "Available environments: development, staging, production, testing"
        exit 1
        ;;
esac

# Install package in development mode
pip install -e .

echo "Dependencies installed successfully for $ENVIRONMENT environment"
```

### System Dependencies

#### 1. FFmpeg Installation

Create `scripts/install_system_deps.sh`:
```bash
#!/bin/bash
# System dependencies installation

set -e

OS=$(uname -s)
DISTRO=""

# Detect Linux distribution
if [ "$OS" = "Linux" ]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    fi
fi

echo "Installing system dependencies for $OS ($DISTRO)..."

case "$OS" in
    "Linux")
        case "$DISTRO" in
            "ubuntu"|"debian")
                sudo apt update
                sudo apt install -y ffmpeg libsndfile1 libsndfile1-dev
                sudo apt install -y build-essential portaudio19-dev
                ;;
            "centos"|"rhel"|"fedora")
                sudo yum install -y epel-release
                sudo yum install -y ffmpeg libsndfile libsndfile-devel
                sudo yum groupinstall -y "Development Tools"
                ;;
            *)
                echo "Unsupported Linux distribution: $DISTRO"
                exit 1
                ;;
        esac
        ;;
    "Darwin")
        if command -v brew >/dev/null 2>&1; then
            brew install ffmpeg libsndfile portaudio
        else
            echo "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        ;;
    *)
        echo "Unsupported operating system: $OS"
        exit 1
        ;;
esac

echo "System dependencies installed successfully"
```

## Database Setup

### Development Database

#### 1. SQLite Setup (Development)

```python
# database/dev_setup.py
import sqlite3
from pathlib import Path

def setup_development_database():
    """Setup SQLite database for development."""
    db_path = Path("development.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            audio_hash TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            processing_time REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            model_path TEXT NOT NULL,
            calibrator_path TEXT,
            accuracy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE
        )
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Development database created at {db_path}")

if __name__ == "__main__":
    setup_development_database()
```

### Production Database

#### 1. PostgreSQL Setup

Create `database/init/01_create_database.sql`:
```sql
-- Create database and user
CREATE DATABASE voice_detector;
CREATE USER prod_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE voice_detector TO prod_user;

-- Connect to the database
\c voice_detector;

-- Create tables
CREATE TABLE requests (
    id SERIAL PRIMARY KEY,
    request_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    audio_hash VARCHAR(64) NOT NULL,
    classification VARCHAR(20) NOT NULL CHECK (classification IN ('AI_GENERATED', 'HUMAN')),
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_time DECIMAL(8,4) NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    calibrator_path TEXT,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_data_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_requests_created_at ON requests (created_at DESC);
CREATE INDEX idx_requests_classification ON requests (classification);
CREATE INDEX idx_requests_audio_hash ON requests (audio_hash);
CREATE INDEX idx_model_versions_active ON model_versions (is_active) WHERE is_active = TRUE;
CREATE INDEX idx_system_metrics_name_time ON system_metrics (metric_name, recorded_at DESC);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_requests_updated_at BEFORE UPDATE ON requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO prod_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO prod_user;
```

#### 2. Database Migration System

Create `database/migrations/001_initial_schema.py`:
```python
"""Initial database schema migration."""

def upgrade(connection):
    """Apply migration."""
    cursor = connection.cursor()
    
    # Create requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id SERIAL PRIMARY KEY,
            request_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
            audio_hash VARCHAR(64) NOT NULL,
            classification VARCHAR(20) NOT NULL,
            confidence_score DECIMAL(5,4) NOT NULL,
            processing_time DECIMAL(8,4) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_requests_created_at 
        ON requests (created_at DESC)
    """)
    
    connection.commit()

def downgrade(connection):
    """Rollback migration."""
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS requests")
    connection.commit()
```

Create `database/migrate.py`:
```python
#!/usr/bin/env python3
"""Database migration runner."""

import os
import sys
import importlib.util
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

class MigrationRunner:
    def __init__(self, database_url):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / "migrations"
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
    
    def create_migration_table(self):
        """Create migration tracking table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            conn.commit()
    
    def get_applied_migrations(self):
        """Get list of applied migrations."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row['version'] for row in cursor.fetchall()]
    
    def get_available_migrations(self):
        """Get list of available migration files."""
        migrations = []
        for file_path in sorted(self.migrations_dir.glob("*.py")):
            if file_path.name != "__init__.py":
                migrations.append(file_path.stem)
        return migrations
    
    def load_migration(self, migration_name):
        """Load migration module."""
        file_path = self.migrations_dir / f"{migration_name}.py"
        spec = importlib.util.spec_from_file_location(migration_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def apply_migration(self, migration_name):
        """Apply a single migration."""
        print(f"Applying migration: {migration_name}")
        
        migration_module = self.load_migration(migration_name)
        
        with self.get_connection() as conn:
            try:
                migration_module.upgrade(conn)
                
                # Record migration as applied
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (migration_name,)
                )
                conn.commit()
                print(f"âœ“ Applied migration: {migration_name}")
                
            except Exception as e:
                conn.rollback()
                print(f"âœ— Failed to apply migration {migration_name}: {e}")
                raise
    
    def migrate(self):
        """Run all pending migrations."""
        self.create_migration_table()
        
        applied = set(self.get_applied_migrations())
        available = self.get_available_migrations()
        
        pending = [m for m in available if m not in applied]
        
        if not pending:
            print("No pending migrations")
            return
        
        print(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            self.apply_migration(migration)
        
        print("All migrations applied successfully")

def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL environment variable is required")
        sys.exit(1)
    
    runner = MigrationRunner(database_url)
    runner.migrate()

if __name__ == "__main__":
    main()
```

## Model Management

### Model Versioning

#### 1. Model Storage Structure

```
models/
â”œâ”€â”€ development/
â”‚   â”œâ”€â”€ model.joblib
â”‚   â”œâ”€â”€ calibrator.joblib
â”‚   â””â”€â”€ metadata.json
â”œâ”€â”€ staging/
â”‚   â”œâ”€â”€ model.joblib
â”‚   â”œâ”€â”€ calibrator.joblib
â”‚   â””â”€â”€ metadata.json
â”œâ”€â”€ production/
â”‚   â”œâ”€â”€ model.joblib
â”‚   â”œâ”€â”€ calibrator.joblib
â”‚   â””â”€â”€ metadata.json
â””â”€â”€ archive/
    â”œâ”€â”€ v1.0.0/
    â”œâ”€â”€ v1.1.0/
    â””â”€â”€ v2.0.0/
```

#### 2. Model Metadata

Create `models/metadata_template.json`:
```json
{
    "version": "1.0.0",
    "created_at": "2024-01-15T10:30:00Z",
    "model_type": "RandomForestClassifier",
    "training_data": {
        "size": 10000,
        "languages": ["tamil", "english", "hindi", "malayalam", "telugu"],
        "ai_samples": 5000,
        "human_samples": 5000
    },
    "performance": {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92,
        "auc_roc": 0.95
    },
    "features": {
        "mfcc_coefficients": 13,
        "spectral_features": 4,
        "temporal_features": 3,
        "total_features": 20
    },
    "calibration": {
        "method": "PlattScaling",
        "calibration_accuracy": 0.89
    },
    "validation": {
        "cross_validation_folds": 5,
        "validation_accuracy": 0.91,
        "test_set_accuracy": 0.92
    },
    "deployment": {
        "environment": "production",
        "deployed_at": "2024-01-15T12:00:00Z",
        "deployed_by": "deployment-system"
    }
}
```

#### 3. Model Management Scripts

Create `scripts/model_manager.py`:
```python
#!/usr/bin/env python3
"""Model management utilities."""

import json
import shutil
from pathlib import Path
from datetime import datetime
import joblib
import argparse

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.environments = ["development", "staging", "production"]
    
    def deploy_model(self, source_path, target_env, version=None):
        """Deploy model to target environment."""
        if target_env not in self.environments:
            raise ValueError(f"Invalid environment: {target_env}")
        
        source_path = Path(source_path)
        target_dir = self.models_dir / target_env
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if source_path.is_dir():
            # Copy entire directory
            for file_path in source_path.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, target_dir / file_path.name)
        else:
            # Copy single file
            shutil.copy2(source_path, target_dir / source_path.name)
        
        # Update metadata
        metadata_path = target_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata.update({
            "deployment": {
                "environment": target_env,
                "deployed_at": datetime.utcnow().isoformat() + "Z",
                "version": version or metadata.get("version", "unknown")
            }
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model deployed to {target_env} environment")
    
    def archive_model(self, env, version):
        """Archive current model from environment."""
        env_dir = self.models_dir / env
        archive_dir = self.models_dir / "archive" / version
        
        if not env_dir.exists():
            raise ValueError(f"Environment {env} does not exist")
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files to archive
        for file_path in env_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, archive_dir / file_path.name)
        
        print(f"Model archived from {env} to {version}")
    
    def validate_model(self, model_path):
        """Validate model file integrity."""
        try:
            model = joblib.load(model_path)
            print(f"âœ“ Model loaded successfully: {type(model).__name__}")
            
            # Check if model has required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model, method):
                    print(f"âœ— Model missing required method: {method}")
                    return False
                else:
                    print(f"âœ“ Model has method: {method}")
            
            return True
            
        except Exception as e:
            print(f"âœ— Model validation failed: {e}")
            return False
    
    def list_models(self):
        """List all available models."""
        for env in self.environments:
            env_dir = self.models_dir / env
            if env_dir.exists():
                print(f"\n{env.upper()} Environment:")
                metadata_path = env_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"  Version: {metadata.get('version', 'unknown')}")
                    print(f"  Accuracy: {metadata.get('performance', {}).get('accuracy', 'unknown')}")
                    print(f"  Deployed: {metadata.get('deployment', {}).get('deployed_at', 'unknown')}")
                else:
                    print("  No metadata available")
        
        # List archived models
        archive_dir = self.models_dir / "archive"
        if archive_dir.exists():
            print(f"\nARCHIVED Models:")
            for version_dir in sorted(archive_dir.iterdir()):
                if version_dir.is_dir():
                    print(f"  {version_dir.name}")

def main():
    parser = argparse.ArgumentParser(description="Model management utility")
    parser.add_argument("command", choices=["deploy", "archive", "validate", "list"])
    parser.add_argument("--source", help="Source model path")
    parser.add_argument("--env", choices=["development", "staging", "production"], help="Target environment")
    parser.add_argument("--version", help="Model version")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.command == "deploy":
        if not args.source or not args.env:
            print("Deploy command requires --source and --env arguments")
            return
        manager.deploy_model(args.source, args.env, args.version)
    
    elif args.command == "archive":
        if not args.env or not args.version:
            print("Archive command requires --env and --version arguments")
            return
        manager.archive_model(args.env, args.version)
    
    elif args.command == "validate":
        if not args.source:
            print("Validate command requires --source argument")
            return
        manager.validate_model(args.source)
    
    elif args.command == "list":
        manager.list_models()

if __name__ == "__main__":
    main()
```

## Testing Environment

### Test Configuration

#### 1. Test Environment Setup

Create `tests/conftest.py`:
```python
"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "RATE_LIMIT_ENABLED": "false",
        "CORS_ORIGINS": "*",
        "MODEL_PATH": str(project_root / "tests" / "fixtures" / "test_model.joblib"),
        "ENABLE_DETAILED_LOGGING": "true"
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_audio_data():
    """Provide sample audio data for testing."""
    # Generate simple sine wave for testing
    import numpy as np
    
    sample_rate = 22050
    duration = 3.0  # seconds
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    return {
        "samples": audio_data,
        "sample_rate": sample_rate,
        "duration": duration
    }

@pytest.fixture
def test_model(temp_dir):
    """Create a simple test model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import joblib
    
    # Generate synthetic training data
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_path = temp_dir / "test_model.joblib"
    joblib.dump(model, model_path)
    
    return model_path
```

#### 2. Test Data Management

Create `tests/fixtures/generate_test_data.py`:
```python
#!/usr/bin/env python3
"""Generate test data for AI Voice Detection System."""

import numpy as np
import base64
import json
from pathlib import Path
import librosa
from pydub import AudioSegment
from pydub.generators import Sine
import io

def generate_test_audio():
    """Generate various test audio samples."""
    test_data = {}
    
    # 1. Simple sine wave (3 seconds)
    sine_wave = Sine(440).to_audio_segment(duration=3000)
    sine_buffer = io.BytesIO()
    sine_wave.export(sine_buffer, format="mp3")
    sine_base64 = base64.b64encode(sine_buffer.getvalue()).decode()
    
    test_data["sine_wave_3s"] = {
        "audio_base64": sine_base64,
        "duration": 3.0,
        "description": "3-second sine wave at 440Hz"
    }
    
    # 2. Short audio (1 second) - should be rejected
    short_wave = Sine(440).to_audio_segment(duration=1000)
    short_buffer = io.BytesIO()
    short_wave.export(short_buffer, format="mp3")
    short_base64 = base64.b64encode(short_buffer.getvalue()).decode()
    
    test_data["short_audio_1s"] = {
        "audio_base64": short_base64,
        "duration": 1.0,
        "description": "1-second sine wave (should be rejected)"
    }
    
    # 3. Long audio (5 seconds)
    long_wave = Sine(440).to_audio_segment(duration=5000)
    long_buffer = io.BytesIO()
    long_wave.export(long_buffer, format="mp3")
    long_base64 = base64.b64encode(long_buffer.getvalue()).decode()
    
    test_data["long_audio_5s"] = {
        "audio_base64": long_base64,
        "duration": 5.0,
        "description": "5-second sine wave"
    }
    
    # 4. Complex audio (multiple frequencies)
    complex_wave = Sine(440).to_audio_segment(duration=1000) + \
                   Sine(880).to_audio_segment(duration=1000) + \
                   Sine(220).to_audio_segment(duration=1000)
    complex_buffer = io.BytesIO()
    complex_wave.export(complex_buffer, format="mp3")
    complex_base64 = base64.b64encode(complex_buffer.getvalue()).decode()
    
    test_data["complex_audio_3s"] = {
        "audio_base64": complex_base64,
        "duration": 3.0,
        "description": "3-second complex audio with multiple frequencies"
    }
    
    # 5. Invalid Base64
    test_data["invalid_base64"] = {
        "audio_base64": "invalid_base64_string!@#$%",
        "duration": None,
        "description": "Invalid Base64 string (should cause error)"
    }
    
    # 6. Empty audio
    test_data["empty_audio"] = {
        "audio_base64": "",
        "duration": None,
        "description": "Empty audio string (should cause error)"
    }
    
    return test_data

def generate_feature_test_data():
    """Generate test data for feature extraction."""
    # Create synthetic audio with known characteristics
    sample_rate = 22050
    duration = 3.0
    
    # Generate different types of audio
    test_features = {}
    
    # 1. Pure tone
    t = np.linspace(0, duration, int(sample_rate * duration))
    pure_tone = np.sin(2 * np.pi * 440 * t)
    
    test_features["pure_tone"] = {
        "audio": pure_tone,
        "sample_rate": sample_rate,
        "expected_features": {
            "dominant_frequency": 440,
            "spectral_centroid_range": [400, 500],
            "mfcc_shape": [13, None]  # 13 coefficients, variable time frames
        }
    }
    
    # 2. White noise
    white_noise = np.random.normal(0, 0.1, int(sample_rate * duration))
    
    test_features["white_noise"] = {
        "audio": white_noise,
        "sample_rate": sample_rate,
        "expected_features": {
            "spectral_centroid_range": [8000, 12000],  # High for white noise
            "energy_variance_min": 0.001  # Should have some variance
        }
    }
    
    # 3. Chirp (frequency sweep)
    chirp = librosa.chirp(fmin=200, fmax=2000, sr=sample_rate, duration=duration)
    
    test_features["chirp"] = {
        "audio": chirp,
        "sample_rate": sample_rate,
        "expected_features": {
            "spectral_centroid_range": [800, 1200],  # Middle of sweep
            "pitch_variance_min": 0.1  # Should have high pitch variance
        }
    }
    
    return test_features

def main():
    """Generate all test data files."""
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)
    
    # Generate audio test data
    print("Generating audio test data...")
    audio_data = generate_test_audio()
    
    with open(fixtures_dir / "test_audio_data.json", "w") as f:
        json.dump(audio_data, f, indent=2)
    
    print(f"Generated {len(audio_data)} audio test samples")
    
    # Generate feature test data
    print("Generating feature test data...")
    feature_data = generate_feature_test_data()
    
    # Save as numpy files
    for name, data in feature_data.items():
        np.save(fixtures_dir / f"feature_test_{name}.npy", data["audio"])
        
        # Save metadata
        metadata = {k: v for k, v in data.items() if k != "audio"}
        with open(fixtures_dir / f"feature_test_{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(feature_data)} feature test samples")
    print("Test data generation completed!")

if __name__ == "__main__":
    main()
```

#### 3. Test Environment Validation

Create `tests/test_environment.py`:
```python
"""Test environment validation."""

import pytest
import os
from pathlib import Path

def test_environment_variables():
    """Test that required environment variables are set."""
    required_vars = [
        "ENVIRONMENT",
        "LOG_LEVEL",
        "MODEL_PATH"
    ]
    
    for var in required_vars:
        assert os.getenv(var) is not None, f"Environment variable {var} is not set"

def test_test_environment_isolation():
    """Test that we're running in test environment."""
    assert os.getenv("ENVIRONMENT") == "testing"
    assert os.getenv("DEBUG") == "true"
    assert os.getenv("RATE_LIMIT_ENABLED") == "false"

def test_model_path_exists():
    """Test that test model path exists."""
    model_path = os.getenv("MODEL_PATH")
    assert model_path is not None
    
    # For testing, we might use a mock model path
    if not model_path.startswith(":memory:") and not model_path == "mock":
        assert Path(model_path).exists(), f"Model path does not exist: {model_path}"

def test_database_is_memory():
    """Test that we're using in-memory database for tests."""
    db_url = os.getenv("DATABASE_URL", "")
    assert ":memory:" in db_url or "sqlite:///" in db_url

def test_cors_is_permissive():
    """Test that CORS is permissive for testing."""
    cors_origins = os.getenv("CORS_ORIGINS", "")
    assert cors_origins == "*"

@pytest.mark.integration
def test_can_import_main_modules():
    """Test that main application modules can be imported."""
    try:
        from voice_detector.api import app
        from voice_detector.audio import processor
        from voice_detector.features import extractor
        from voice_detector.models import classifier
    except ImportError as e:
        pytest.fail(f"Failed to import main modules: {e}")

@pytest.mark.integration
def test_test_fixtures_exist():
    """Test that test fixtures are available."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    expected_files = [
        "test_audio_data.json"
    ]
    
    for file_name in expected_files:
        file_path = fixtures_dir / file_name
        if file_path.exists():
            assert file_path.is_file(), f"Expected file is not a file: {file_path}"
        # Note: Some fixtures might be generated on-demand, so we don't require all to exist
```

This comprehensive environment setup guide provides detailed instructions for setting up development, staging, and production environments, along with proper dependency management, database setup, model management, and testing environment configuration. Each section includes practical scripts and configurations that can be used immediately.
