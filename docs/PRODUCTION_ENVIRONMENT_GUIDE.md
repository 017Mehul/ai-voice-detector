# Production Environment Configuration Guide

This guide provides comprehensive instructions for configuring the AI Voice Detection System for production deployment. It covers environment variables, security considerations, performance tuning, and deployment best practices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Variables Reference](#environment-variables-reference)
3. [Security Configuration](#security-configuration)
4. [Performance Tuning](#performance-tuning)
5. [Deployment Profiles](#deployment-profiles)
6. [Validation and Testing](#validation-and-testing)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Automated Configuration (Recommended)

Use the interactive configuration script to set up your production environment:

```bash
# Run interactive setup
python scripts/configure_production_env.py --interactive

# Or use a specific deployment profile
python scripts/configure_production_env.py --profile medium
```

### Manual Configuration

1. Copy the production template:
   ```bash
   cp .env.production .env
   ```

2. Edit `.env` with your specific values:
   ```bash
   # Required changes for production
   ENVIRONMENT=production
   DEBUG=false
   CORS_ORIGINS=https://yourdomain.com
   MODEL_PATH=./models/production/model.joblib
   ```

3. Validate your configuration:
   ```bash
   python scripts/validate_production_config.py --config .env
   ```

## Environment Variables Reference

### Core Application Settings

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `ENVIRONMENT` | `production` | Deployment environment | ✅ |
| `DEBUG` | `false` | Debug mode (MUST be false in production) | ✅ |
| `LOG_LEVEL` | `INFO` | Logging level (INFO, WARNING, ERROR, CRITICAL) | ✅ |
| `ENABLE_DETAILED_LOGGING` | `false` | Enable detailed request/response logging | ❌ |

### API Server Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `API_HOST` | `0.0.0.0` | Server bind address | ✅ |
| `API_PORT` | `8000` | Server port | ✅ |
| `API_WORKERS` | `4` | Number of worker processes | ✅ |
| `REQUEST_TIMEOUT` | `30` | Request timeout in seconds | ❌ |
| `MAX_REQUEST_SIZE` | `52428800` | Maximum request size in bytes (50MB) | ❌ |

### CORS Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `ENABLE_CORS` | `true` | Enable CORS support | ❌ |
| `CORS_ORIGINS` | `*` | Allowed origins (specify exact domains in production) | ❌ |
| `CORS_METHODS` | `GET,POST,OPTIONS` | Allowed HTTP methods | ❌ |
| `CORS_HEADERS` | `Content-Type,Authorization` | Allowed headers | ❌ |

### Model Configuration

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `MODEL_PATH` | `./models/production/model.joblib` | Path to trained model file | ✅ |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.5` | Classification confidence threshold | ❌ |
| `ENABLE_FEATURE_IMPORTANCE` | `true` | Enable feature importance calculation | ❌ |
| `ENABLE_BATCH_PROCESSING` | `false` | Enable batch processing (if implemented) | ❌ |

### Audio Processing Settings

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `MAX_AUDIO_SIZE_MB` | `10` | Maximum audio file size in MB | ❌ |
| `MAX_AUDIO_DURATION_SECONDS` | `30` | Maximum audio duration | ❌ |
| `MIN_AUDIO_DURATION_SECONDS` | `2` | Minimum audio duration | ❌ |
| `AUDIO_SAMPLE_RATE` | `22050` | Audio sample rate for processing | ❌ |
| `AUDIO_N_MFCC` | `13` | Number of MFCC coefficients | ❌ |

### Rate Limiting

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting | ❌ |
| `RATE_LIMIT_REQUESTS` | `100` | Maximum requests per window | ❌ |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds | ❌ |

### Monitoring and Health Checks

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `HEALTH_CHECK_ENABLED` | `true` | Enable health check endpoint | ❌ |
| `METRICS_ENABLED` | `true` | Enable metrics collection | ❌ |
| `PROMETHEUS_PORT` | `9090` | Prometheus metrics port | ❌ |

### Security Settings

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `API_KEY` | - | API key for authentication (if implemented) | ❌ |
| `JWT_SECRET` | - | JWT secret for token authentication | ❌ |
| `ENABLE_REQUEST_TRACKING` | `true` | Enable request ID tracking | ❌ |

## Security Configuration

### Essential Security Settings

1. **Disable Debug Mode**:
   ```bash
   DEBUG=false
   ENVIRONMENT=production
   ```

2. **Restrict CORS Origins**:
   ```bash
   # Instead of wildcard
   CORS_ORIGINS=*
   
   # Use specific domains
   CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
   ```

3. **Set Appropriate Log Level**:
   ```bash
   LOG_LEVEL=INFO  # or WARNING/ERROR for less verbose logging
   ENABLE_DETAILED_LOGGING=false
   ```

4. **Generate Secure Keys**:
   ```bash
   # Use the configuration script to generate secure keys
   python scripts/configure_production_env.py --interactive
   ```

### Security Best Practices

- **Never commit `.env` files** to version control
- **Use environment-specific secrets management** (AWS Secrets Manager, Azure Key Vault, etc.)
- **Regularly rotate API keys and secrets**
- **Monitor for suspicious request patterns**
- **Use HTTPS in production** (configure at reverse proxy level)

## Performance Tuning

### Deployment Profiles

The system provides three pre-configured deployment profiles:

#### Small Deployment (Low Traffic)
```bash
API_WORKERS=2
MAX_CONCURRENT_REQUESTS=50
RATE_LIMIT_REQUESTS=50
MAX_AUDIO_SIZE_MB=5
```

#### Medium Deployment (Moderate Traffic)
```bash
API_WORKERS=4
MAX_CONCURRENT_REQUESTS=100
RATE_LIMIT_REQUESTS=100
MAX_AUDIO_SIZE_MB=10
```

#### Large Deployment (High Traffic)
```bash
API_WORKERS=8
MAX_CONCURRENT_REQUESTS=200
RATE_LIMIT_REQUESTS=200
MAX_AUDIO_SIZE_MB=15
```

### Performance Optimization Guidelines

1. **Worker Processes**: Set `API_WORKERS` to 2 × CPU cores
2. **Memory Management**: Monitor memory usage and adjust `MAX_AUDIO_SIZE_MB`
3. **Rate Limiting**: Adjust based on expected traffic patterns
4. **Caching**: Enable response caching for identical requests:
   ```bash
   ENABLE_RESPONSE_CACHING=true
   CACHE_TTL=300
   ```

### Resource Requirements

| Deployment Size | CPU | Memory | Storage | Concurrent Users |
|----------------|-----|--------|---------|------------------|
| Small | 1-2 cores | 2-4 GB | 10 GB | 10-50 |
| Medium | 2-4 cores | 4-8 GB | 20 GB | 50-200 |
| Large | 4-8 cores | 8-16 GB | 50 GB | 200-1000 |

## Deployment Scenarios

### Docker Deployment

1. **Create Docker environment file**:
   ```bash
   python scripts/configure_production_env.py --profile medium
   cp .env .env.docker
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Environment variables in docker-compose.yml**:
   ```yaml
   services:
     ai-voice-detector:
       environment:
         - ENVIRONMENT=production
         - DEBUG=false
         - LOG_LEVEL=INFO
       env_file:
         - .env.docker
   ```

### Kubernetes Deployment

1. **Generate Kubernetes ConfigMap**:
   ```bash
   python scripts/configure_production_env.py --interactive
   # This creates k8s-config.yaml
   ```

2. **Apply configuration**:
   ```bash
   kubectl apply -f k8s-config.yaml
   ```

3. **Reference in deployment**:
   ```yaml
   spec:
     containers:
     - name: ai-voice-detector
       envFrom:
       - configMapRef:
           name: ai-voice-detector-config
       - secretRef:
           name: ai-voice-detector-secrets
   ```

### Cloud Platform Deployment

#### AWS ECS/Fargate
```bash
# Set AWS-specific variables
AWS_REGION=us-west-2
DEPLOYMENT_VERSION=1.0.0
ENABLE_AUDIT_LOGGING=true
```

#### Google Cloud Run
```bash
# Set GCP-specific variables
GCP_PROJECT_ID=your-project-id
DEPLOYMENT_VERSION=1.0.0
```

#### Azure Container Instances
```bash
# Set Azure-specific variables
AZURE_RESOURCE_GROUP=your-resource-group
DEPLOYMENT_VERSION=1.0.0
```

## Validation and Testing

### Configuration Validation

Always validate your configuration before deployment:

```bash
# Validate configuration
python scripts/validate_production_config.py --config .env

# Strict validation (treat warnings as errors)
python scripts/validate_production_config.py --config .env --strict

# JSON output for automation
python scripts/validate_production_config.py --config .env --json
```

### Health Checks

The system provides built-in health checks:

```bash
# Health check endpoint
curl http://localhost:8000/health

# Metrics endpoint (if enabled)
curl http://localhost:9090/metrics
```

### Load Testing

Test your configuration under load:

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host http://localhost:8000
```

## Monitoring and Observability

### Logging Configuration

```bash
# Production logging settings
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=false
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=90
```

### Metrics Collection

```bash
# Enable metrics
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Optional: External monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

### Log Aggregation

For centralized logging, configure log forwarding:

```bash
# Example: ELK Stack integration
ELASTICSEARCH_URL=https://your-elasticsearch-cluster
KIBANA_URL=https://your-kibana-instance
```

## Troubleshooting

### Common Issues

1. **Model File Not Found**:
   ```bash
   # Check model path
   ls -la ./models/production/model.joblib
   
   # Update MODEL_PATH if needed
   MODEL_PATH=/absolute/path/to/model.joblib
   ```

2. **Permission Errors**:
   ```bash
   # Ensure proper file permissions
   chmod 644 .env
   chmod +x scripts/*.py
   ```

3. **Port Already in Use**:
   ```bash
   # Change API port
   API_PORT=8001
   
   # Or find and kill process using port
   lsof -ti:8000 | xargs kill -9
   ```

4. **Memory Issues**:
   ```bash
   # Reduce resource usage
   MAX_AUDIO_SIZE_MB=5
   API_WORKERS=2
   MAX_CONCURRENT_REQUESTS=50
   ```

### Validation Errors

Common validation errors and fixes:

| Error | Fix |
|-------|-----|
| `DEBUG must be false in production` | Set `DEBUG=false` |
| `CORS_ORIGINS should specify exact domains` | Replace `*` with specific domains |
| `Model file not found` | Ensure model file exists at specified path |
| `Invalid port range` | Set `API_PORT` between 1-65535 |

### Performance Issues

1. **High Response Times**:
   - Increase `API_WORKERS`
   - Reduce `MAX_AUDIO_SIZE_MB`
   - Enable caching

2. **Memory Usage**:
   - Monitor with `docker stats` or `kubectl top`
   - Adjust worker count and audio limits

3. **Rate Limiting Issues**:
   - Adjust `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`
   - Monitor request patterns

## Environment File Templates

### Minimal Production Configuration

```bash
# .env.minimal
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./models/production/model.joblib
CORS_ORIGINS=https://yourdomain.com
```

### Complete Production Configuration

See `.env.production` for a comprehensive template with all available options.

## Security Checklist

Before deploying to production, ensure:

- [ ] `DEBUG=false`
- [ ] `ENVIRONMENT=production`
- [ ] CORS origins are specific domains, not `*`
- [ ] Secure API keys and secrets are generated
- [ ] Model file exists and is accessible
- [ ] Log level is appropriate (INFO or higher)
- [ ] Rate limiting is enabled and configured
- [ ] Health checks are enabled
- [ ] Configuration validation passes
- [ ] Load testing completed successfully
- [ ] Monitoring and alerting configured

## Support and Maintenance

### Regular Maintenance Tasks

1. **Monitor system health** via health check endpoints
2. **Review logs** for errors and performance issues
3. **Update model files** as needed
4. **Rotate secrets** regularly
5. **Monitor resource usage** and scale as needed

### Getting Help

- Check the validation output for specific guidance
- Review logs for error details
- Use health check endpoints to diagnose issues
- Monitor metrics for performance insights

For additional support, refer to the main documentation or contact your system administrator.