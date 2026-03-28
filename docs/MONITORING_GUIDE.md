# AI Voice Detection System - Monitoring Guide

## Overview

The AI Voice Detection System includes comprehensive monitoring and observability features using Prometheus metrics, Grafana dashboards, and alerting capabilities. This guide covers setup, configuration, and usage of the monitoring infrastructure.

## Architecture

The monitoring stack consists of:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **Application Metrics**: Built-in Prometheus metrics in the AI Voice Detector

## Metrics Collected

### Request Metrics
- `ai_voice_detector_requests_total`: Total number of requests by method, endpoint, and status
- `ai_voice_detector_request_duration_seconds`: Request duration histogram

### Processing Metrics
- `ai_voice_detector_audio_processing_seconds`: Audio processing duration by stage
- `ai_voice_detector_feature_extraction_seconds`: Feature extraction duration by type
- `ai_voice_detector_model_inference_seconds`: Model inference duration

### Error Metrics
- `ai_voice_detector_errors_total`: Total errors by type and component

### System Metrics
- `ai_voice_detector_memory_usage_bytes`: Current memory usage
- `ai_voice_detector_cpu_usage_percent`: Current CPU usage

### Cache Metrics
- `ai_voice_detector_cache_hits_total`: Cache hits by type
- `ai_voice_detector_cache_misses_total`: Cache misses by type

### Model Performance Metrics
- `ai_voice_detector_classifications_total`: Classifications by result and language
- `ai_voice_detector_confidence_score`: Confidence score distribution

### Audio Quality Metrics
- `ai_voice_detector_audio_duration_seconds`: Audio sample duration distribution
- `ai_voice_detector_audio_sample_rate_hz`: Audio sample rate distribution

## Setup Instructions

### 1. Install Dependencies

```bash
pip install prometheus_client psutil
```

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# Monitoring Configuration
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
HEALTH_CHECK_ENABLED=true
ENABLE_DETAILED_LOGGING=true
```

### 3. Start the Monitoring Stack

```bash
# Start Prometheus, Grafana, and Alertmanager
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 4. Start the AI Voice Detector

```bash
# The application will automatically start the metrics server
python -m uvicorn ai_voice_detector.api.app:create_app --host 0.0.0.0 --port 8000
```

## Accessing Monitoring Tools

### Prometheus
- URL: http://localhost:9091
- Use to query raw metrics and test alert rules

### Grafana
- URL: http://localhost:3000
- Default credentials: admin/admin
- Pre-configured dashboard for AI Voice Detection System

### Alertmanager
- URL: http://localhost:9093
- Manage alert routing and silencing

### Application Metrics Endpoint
- URL: http://localhost:8000/metrics
- Raw Prometheus metrics from the application

## Key Dashboards and Queries

### Request Rate
```promql
rate(ai_voice_detector_requests_total[5m])
```

### Response Time (95th percentile)
```promql
histogram_quantile(0.95, rate(ai_voice_detector_request_duration_seconds_bucket[5m]))
```

### Error Rate
```promql
rate(ai_voice_detector_errors_total[5m])
```

### Memory Usage
```promql
ai_voice_detector_memory_usage_bytes / 1024 / 1024
```

### Cache Hit Rate
```promql
rate(ai_voice_detector_cache_hits_total[5m]) / 
(rate(ai_voice_detector_cache_hits_total[5m]) + rate(ai_voice_detector_cache_misses_total[5m]))
```

## Alerting Rules

### High Error Rate
Triggers when error rate exceeds 0.1 errors/second for 2 minutes.

### High Response Time
Triggers when 95th percentile response time exceeds 5 seconds for 5 minutes.

### High Memory Usage
Triggers when memory usage exceeds 2GB for 5 minutes.

### Service Down
Triggers when the service is unreachable for 1 minute.

### Low Cache Hit Rate
Triggers when cache hit rate falls below 50% for 10 minutes.

## Performance Optimization

### Memory Monitoring
The system automatically monitors memory usage and triggers garbage collection when needed:

```python
# Memory optimization is automatic
if self.memory_optimizer.is_memory_pressure():
    self.memory_optimizer.optimize_memory_if_needed()
```

### Cache Performance
Monitor cache effectiveness through metrics:
- High cache hit rates (>80%) indicate good performance
- Low hit rates may require cache tuning or increased cache size

### Processing Pipeline Performance
Track individual pipeline stages:
- Audio processing should complete in <100ms
- Feature extraction should complete in <500ms  
- Model inference should complete in <50ms

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
1. Check that `METRICS_ENABLED=true` in environment
2. Verify Prometheus is scraping the correct endpoint
3. Check application logs for metrics initialization errors

#### High Memory Usage
1. Monitor garbage collection frequency
2. Check for memory leaks in long-running processes
3. Consider reducing cache sizes if memory pressure persists

#### Slow Response Times
1. Check individual pipeline stage metrics
2. Monitor CPU and memory usage during high load
3. Consider scaling horizontally if needed

### Log Analysis
Enable detailed logging for troubleshooting:

```bash
ENABLE_DETAILED_LOGGING=true
LOG_LEVEL=DEBUG
```

## Production Considerations

### Security
- Restrict access to monitoring endpoints in production
- Use authentication for Grafana and Prometheus
- Consider TLS encryption for metrics transmission

### Retention
- Configure appropriate data retention policies
- Monitor disk usage for metrics storage
- Set up automated cleanup for old metrics

### Scaling
- Consider Prometheus federation for multiple instances
- Use load balancers for high-availability monitoring
- Implement horizontal scaling for the application

### Backup
- Backup Grafana dashboards and configurations
- Export Prometheus data for disaster recovery
- Document alert rule configurations

## Custom Metrics

To add custom metrics to the application:

```python
from ai_voice_detector.utils.prometheus_metrics import get_prometheus_metrics

metrics = get_prometheus_metrics()

# Record custom events
metrics.record_custom_event("event_type", "component")

# Time custom operations
with metrics.time_operation("custom_operation"):
    # Your code here
    pass
```

## Integration with External Systems

### Webhook Alerts
Configure webhook endpoints in Alertmanager for integration with:
- Slack notifications
- PagerDuty incidents
- Custom notification systems

### Log Aggregation
Integrate with log aggregation systems:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- Fluentd

### APM Integration
Consider additional APM tools for deeper insights:
- Jaeger for distributed tracing
- New Relic for application performance
- DataDog for comprehensive monitoring

## Maintenance

### Regular Tasks
1. Review and update alert thresholds monthly
2. Clean up old metrics data quarterly
3. Update monitoring stack components regularly
4. Test alert notifications periodically

### Capacity Planning
- Monitor metrics growth over time
- Plan storage capacity for Prometheus
- Scale monitoring infrastructure as needed

## Support

For monitoring-related issues:
1. Check the application logs first
2. Verify monitoring stack health
3. Review Prometheus targets and scrape status
4. Test alert rules in Prometheus UI

The monitoring system provides comprehensive observability into the AI Voice Detection System's performance, health, and usage patterns, enabling proactive maintenance and optimization.