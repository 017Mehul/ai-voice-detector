# Operations Guide

## Overview

This operations guide provides comprehensive instructions for managing, monitoring, and maintaining the AI Voice Detection System in production environments. It covers day-to-day operations, troubleshooting, performance optimization, and incident response procedures.

## Table of Contents

1. [System Monitoring](#system-monitoring)
2. [Log Management](#log-management)
3. [Performance Monitoring](#performance-monitoring)
4. [Health Checks](#health-checks)
5. [Incident Response](#incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Scaling Operations](#scaling-operations)
8. [Security Operations](#security-operations)
9. [Backup and Recovery Operations](#backup-and-recovery-operations)
10. [Troubleshooting Guide](#troubleshooting-guide)

## System Monitoring

### 1. Key Metrics to Monitor

**Application Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `audio_processing_duration_seconds` - Audio processing time
- `model_inference_duration_seconds` - Model inference time
- `classification_accuracy_score` - Real-time accuracy tracking

**System Metrics:**
- `process_cpu_seconds_total` - CPU usage
- `process_resident_memory_bytes` - Memory usage
- `process_open_fds` - Open file descriptors
- `python_gc_objects_collected_total` - Garbage collection

### 2. Alerting Thresholds

**Critical Alerts:**
- Service down for > 1 minute
- Error rate > 5% for 5 minutes
- 95th percentile latency > 10 seconds
- Memory usage > 6GB for 10 minutes

**Warning Alerts:**
- Error rate > 1% for 10 minutes
- 95th percentile latency > 5 seconds
- CPU usage > 80% for 15 minutes
- Disk space < 20%

## Log Management

### 1. Log Levels and Categories

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General operational messages
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical error conditions

**Log Categories:**
- `request_start/end`: HTTP request lifecycle
- `audio_processing`: Audio processing events
- `model_inference`: Model prediction events
- `error`: Error conditions
- `performance`: Performance metrics

### 2. Log Analysis

Use the provided log analysis script:
```bash
python scripts/log_analysis.py /path/to/logs/app.log --hours 24 --analysis all
```

## Performance Monitoring

### 1. Performance Benchmarks

Run performance benchmarks regularly:
```bash
python scripts/performance/benchmark.py --test full --output benchmark_results.json
```

**Expected Performance:**
- Health check: < 100ms (95th percentile)
- API requests: < 3 seconds (95th percentile)
- Throughput: > 20 requests/second per instance

### 2. Performance Optimization

**Memory Optimization:**
- Monitor memory usage trends
- Restart services if memory usage exceeds 80%
- Use memory profiling tools for memory leaks

**CPU Optimization:**
- Scale horizontally when CPU usage > 70%
- Optimize model inference for better CPU utilization
- Use caching for repeated requests

## Health Checks

### 1. Health Check Endpoints

**Basic Health Check:**
```bash
curl http://localhost:8000/health
```

**Detailed Health Check:**
```bash
curl http://localhost:8000/health/detailed
```

**Kubernetes Probes:**
- Liveness: `/health/live`
- Readiness: `/health/ready`

### 2. Health Check Automation

Use the automated health monitoring script:
```bash
./scripts/monitoring/health_monitor.sh
```

## Incident Response

### 1. Incident Classification

**P0 - Critical:**
- Service completely down
- Data corruption
- Security breach

**P1 - High:**
- Significant performance degradation
- High error rates
- Partial service outage

**P2 - Medium:**
- Minor performance issues
- Non-critical feature failures

**P3 - Low:**
- Cosmetic issues
- Enhancement requests

### 2. Response Procedures

**P0/P1 Incidents:**
1. Acknowledge alert within 5 minutes
2. Assess impact and scope
3. Implement immediate mitigation
4. Communicate status to stakeholders
5. Investigate root cause
6. Implement permanent fix
7. Conduct post-incident review

**P2/P3 Incidents:**
1. Acknowledge within 1 hour
2. Schedule fix during maintenance window
3. Document issue and resolution

## Maintenance Procedures

### 1. Regular Maintenance Tasks

**Daily:**
- Check system health dashboards
- Review error logs
- Monitor resource usage
- Verify backup completion

**Weekly:**
- Analyze performance trends
- Review security logs
- Update dependencies (if needed)
- Test disaster recovery procedures

**Monthly:**
- Performance benchmark testing
- Security vulnerability scanning
- Capacity planning review
- Documentation updates

### 2. Deployment Procedures

**Rolling Deployment:**
1. Deploy to staging environment
2. Run full test suite
3. Deploy to production (one instance at a time)
4. Monitor health and performance
5. Complete deployment or rollback if issues

**Blue-Green Deployment:**
1. Deploy to green environment
2. Run smoke tests
3. Switch traffic to green
4. Monitor for issues
5. Keep blue as fallback

## Scaling Operations

### 1. Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up --scale ai-voice-detector=3
```

**Kubernetes:**
```bash
kubectl scale deployment ai-voice-detector --replicas=5
```

### 2. Auto-scaling Configuration

**CPU-based scaling:**
- Scale up when CPU > 70% for 5 minutes
- Scale down when CPU < 30% for 10 minutes
- Min replicas: 2
- Max replicas: 10

**Memory-based scaling:**
- Scale up when memory > 80% for 5 minutes
- Scale down when memory < 40% for 10 minutes

## Security Operations

### 1. Security Monitoring

**Monitor for:**
- Unusual request patterns
- Failed authentication attempts
- Suspicious file uploads
- Abnormal resource usage

### 2. Security Procedures

**Daily:**
- Review security logs
- Check for failed login attempts
- Monitor resource usage patterns

**Weekly:**
- Update security patches
- Review access logs
- Scan for vulnerabilities

**Monthly:**
- Security audit
- Penetration testing
- Access review

## Backup and Recovery Operations

### 1. Backup Schedule

**Daily:**
- Application logs
- Configuration files
- Model files

**Weekly:**
- Full system backup
- Database backup (if applicable)

**Monthly:**
- Archive old backups
- Test restore procedures

### 2. Recovery Procedures

**Model Recovery:**
1. Identify backup version
2. Stop application
3. Replace model files
4. Restart application
5. Verify functionality

**Configuration Recovery:**
1. Restore configuration files
2. Restart services
3. Verify settings
4. Test functionality

## Troubleshooting Guide

### 1. Common Issues

**Service Won't Start:**
- Check port availability
- Verify environment variables
- Check model file existence
- Review startup logs

**High Memory Usage:**
- Check for memory leaks
- Monitor garbage collection
- Restart service if needed
- Scale horizontally

**Slow Response Times:**
- Check system resources
- Analyze request patterns
- Optimize model inference
- Add caching

**Audio Processing Errors:**
- Verify FFmpeg installation
- Check audio file format
- Validate input data
- Review error logs

### 2. Diagnostic Commands

**System Resources:**
```bash
# CPU and memory usage
top -p $(pgrep -f ai-voice-detector)

# Disk usage
df -h

# Network connections
netstat -tulpn | grep :8000
```

**Application Diagnostics:**
```bash
# Check service status
curl -f http://localhost:8000/health/detailed

# View recent logs
docker logs --tail 100 ai-voice-detector

# Check process information
ps aux | grep ai-voice-detector
```

**Performance Analysis:**
```bash
# Run performance benchmark
python scripts/performance/benchmark.py --test api --requests 10

# Analyze logs
python scripts/log_analysis.py logs/app.log --hours 1
```

### 3. Emergency Procedures

**Service Recovery:**
1. Check service status
2. Review recent logs
3. Restart service
4. Verify health checks
5. Monitor for stability

**Data Recovery:**
1. Identify affected data
2. Stop write operations
3. Restore from backup
4. Verify data integrity
5. Resume operations

**Security Incident:**
1. Isolate affected systems
2. Preserve evidence
3. Assess impact
4. Implement containment
5. Notify stakeholders
6. Begin recovery

This operations guide provides the essential procedures and scripts needed to effectively operate the AI Voice Detection System in production environments.