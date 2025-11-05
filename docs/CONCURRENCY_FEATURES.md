# Concurrency and Resource Management Features

This document provides an overview of the concurrency control and resource management features implemented in the Voice Stack project.

## Overview

The Voice Stack now includes comprehensive safeguards and controls to prevent resource exhaustion, manage concurrent requests, and automatically free resources when idle. These features are essential for production deployments, especially on resource-constrained systems.

## Features Implemented

### 1. Resource Guard Middleware

**File:** [src/core/middleware.py](../src/core/middleware.py)

A FastAPI middleware that checks system resources **before** processing any request.

**Checks Performed:**
- **Memory Usage:** Rejects requests when RAM usage exceeds `MEMORY_THRESHOLD_PERCENT` (default: 90%)
- **Swap Usage:** Rejects requests when swap usage exceeds `SWAP_THRESHOLD_PERCENT` (default: 80%)
- **File Size:** Rejects file uploads larger than `MAX_UPLOAD_SIZE_MB` (default: 100MB)

**HTTP Status Codes:**
- `503 Service Unavailable`: Memory or swap pressure (client should retry after 60-120 seconds)
- `413 Payload Too Large`: File exceeds size limit (client error)

**Configuration (.env):**
```bash
MEMORY_THRESHOLD_PERCENT=90  # Reject when RAM > 90%
SWAP_THRESHOLD_PERCENT=80    # Reject when swap > 80%
MAX_UPLOAD_SIZE_MB=100       # Max file size in MB
```

### 2. Concurrency Control (Request Limiting)

**Files:**
- [src/asr/engine_factory.py](../src/asr/engine_factory.py)
- [src/tts/engine_factory.py](../src/tts/engine_factory.py)

Limits the number of simultaneous inference requests using asyncio Semaphores.

**How it Works:**
1. Each service (ASR/TTS) has a semaphore with `MAX_CONCURRENT_REQUESTS` slots
2. Endpoints use `acquire_engine()` to get an engine instance
3. If all slots are busy, requests are immediately rejected with HTTP 429
4. After processing, endpoints call `release_engine()` to free the slot

**Configuration (.env):**
```bash
ASR_MAX_CONCURRENT_REQUESTS=2  # Max concurrent ASR requests
TTS_MAX_CONCURRENT_REQUESTS=2  # Max concurrent TTS requests
```

**Example Usage:**
```python
@router.post("/transcribe")
async def transcribe(file: UploadFile):
    engine = await acquire_engine()
    try:
        result = engine.transcribe_file(...)
        return result
    finally:
        release_engine()
```

### 3. Idle Timeout (Automatic Model Unloading)

**Files:**
- [src/asr/engine/base.py](../src/asr/engine/base.py) (ASRBase class)
- [src/tts/engine/base.py](../src/tts/engine/base.py) (TTSBase class)
- [src/asr/app.py](../src/asr/app.py) (background task)
- [src/tts/app.py](../src/tts/app.py) (background task)

Automatically unloads models from GPU/RAM after a period of inactivity to free resources.

**How it Works:**
1. Base classes track `last_used` timestamp on every inference request
2. Background task runs every 60 seconds checking if models are idle
3. If idle time exceeds `IDLE_TIMEOUT_MINUTES`, model is unloaded
4. Model automatically reloads on next request

**Configuration (.env):**
```bash
ASR_IDLE_TIMEOUT_MINUTES=60  # Unload ASR model after 60 min idle (0=disabled)
TTS_IDLE_TIMEOUT_MINUTES=60  # Unload TTS model after 60 min idle (0=disabled)
```

**Implemented in Engines:**
- ✅ ASRFasterWhisper ([src/asr/engine/fasterwhisper.py](../src/asr/engine/fasterwhisper.py))
- ✅ ASRWhisperTorch ([src/asr/engine/whisper.py](../src/asr/engine/whisper.py))
- ✅ TTSCoqui ([src/tts/engine/coqui.py](../src/tts/engine/coqui.py))

### 4. Enhanced Configuration

**File:** [src/core/settings.py](../src/core/settings.py)

All new settings are properly defined with defaults and descriptions.

**New Settings Added:**
```python
# Resource Safeguards
MAX_UPLOAD_SIZE_MB: int = 100
MEMORY_THRESHOLD_PERCENT: int = 90
SWAP_THRESHOLD_PERCENT: int = 80

# ASR Resource Management
ASR_IDLE_TIMEOUT_MINUTES: int = 60
ASR_MAX_CONCURRENT_REQUESTS: int = 2

# TTS Resource Management
TTS_IDLE_TIMEOUT_MINUTES: int = 60
TTS_MAX_CONCURRENT_REQUESTS: int = 2
```

### 5. Detailed Health Endpoint

**Files:**
- [src/asr/app.py](../src/asr/app.py) (`/health/detailed`)
- [src/tts/app.py](../src/tts/app.py) (`/health/detailed`)

Provides comprehensive system metrics for monitoring.

**Example Response:**
```json
{
  "status": "healthy",
  "service": "asr",
  "timestamp": "2025-11-04T12:00:00.000Z",
  "memory": {
    "percent": 45.2,
    "available_mb": 8192.5,
    "total_mb": 16384.0,
    "threshold_percent": 90
  },
  "swap": {
    "percent": 12.3,
    "used_mb": 256.0,
    "total_mb": 2048.0,
    "threshold_percent": 80
  },
  "model": {
    "loaded": true,
    "engine": "fasterwhisper",
    "model_name": "large-v3",
    "device": "cuda"
  },
  "concurrency": {
    "active_requests": 1,
    "max_concurrent": 2,
    "available_slots": 1
  },
  "config": {
    "idle_timeout_minutes": 60,
    "max_upload_mb": 100
  }
}
```

### 6. Load Testing Infrastructure

**Files:**
- [scripts/load_test.py](../scripts/load_test.py) - Load testing script
- [docs/LOAD_TESTING.md](LOAD_TESTING.md) - Comprehensive testing guide

A complete load testing framework to verify concurrency controls and resource management.

**Features:**
- Concurrent worker simulation
- Both request-count and duration-based testing
- Detailed statistics and error reporting
- Support for testing ASR, TTS, or both services
- Configurable timeouts and verbosity

**Quick Start:**
```bash
# Smoke test
python scripts/load_test.py --service both --workers 2 --requests 5

# Standard load test
python scripts/load_test.py --service asr --workers 10 --requests 50

# Stress test
python scripts/load_test.py --service both --workers 20 --duration 300
```

## Architecture

### Request Flow with All Safeguards

```
                    ┌─────────────────────────────┐
                    │   Incoming HTTP Request     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ResourceGuardMiddleware    │
                    │  - Check Memory (< 90%)     │
                    │  - Check Swap (< 80%)       │
                    │  - Check File Size (< 100MB)│
                    └──────────────┬──────────────┘
                                   │
                            ┌──────▼───────┐
                            │   Pass?      │
                            └──┬────────┬──┘
                            No │        │ Yes
                     ┌─────────▼─┐      │
                     │ 503/413   │      │
                     │ Reject    │      │
                     └───────────┘      │
                                  ┌─────▼─────┐
                                  │  Endpoint │
                                  └─────┬─────┘
                                        │
                          ┌─────────────▼────────────────┐
                          │   acquire_engine()           │
                          │   - Check semaphore slots    │
                          └─────────────┬────────────────┘
                                        │
                                 ┌──────▼───────┐
                                 │ Slot Free?   │
                                 └──┬────────┬──┘
                              No   │        │ Yes
                       ┌───────────▼─┐      │
                       │  429 Reject │      │
                       │  (Retry 10s)│      │
                       └─────────────┘      │
                                      ┌─────▼─────────┐
                                      │ Process       │
                                      │ Inference     │
                                      └─────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │ release_engine() │
                                   │ Free slot        │
                                   └──────────────────┘

┌────────────────────────────────────────────────────────┐
│         Background Task (every 60 seconds)             │
│  - Check if model idle > IDLE_TIMEOUT_MINUTES          │
│  - If yes, unload model to free GPU/RAM                │
│  - Model reloads automatically on next request         │
└────────────────────────────────────────────────────────┘
```

## Configuration Examples

### Production Configuration (Resource-Constrained)

For systems with limited resources (e.g., 8GB RAM, shared GPU):

```bash
# .env
ENV=production

# Resource Safeguards - Conservative
MEMORY_THRESHOLD_PERCENT=85
SWAP_THRESHOLD_PERCENT=75
MAX_UPLOAD_SIZE_MB=50

# Concurrency - Limited
ASR_MAX_CONCURRENT_REQUESTS=1
TTS_MAX_CONCURRENT_REQUESTS=1

# Idle Timeout - Aggressive
ASR_IDLE_TIMEOUT_MINUTES=30
TTS_IDLE_TIMEOUT_MINUTES=30

# Models - Smaller sizes
ASR_MODEL=medium
ASR_COMPUTE_TYPE=int8
```

### Production Configuration (High-Performance)

For systems with ample resources (e.g., 32GB+ RAM, dedicated GPU):

```bash
# .env
ENV=production

# Resource Safeguards - Permissive
MEMORY_THRESHOLD_PERCENT=95
SWAP_THRESHOLD_PERCENT=90
MAX_UPLOAD_SIZE_MB=200

# Concurrency - Higher
ASR_MAX_CONCURRENT_REQUESTS=4
TTS_MAX_CONCURRENT_REQUESTS=4

# Idle Timeout - Less aggressive
ASR_IDLE_TIMEOUT_MINUTES=120
TTS_IDLE_TIMEOUT_MINUTES=120

# Models - Larger, better quality
ASR_MODEL=large-v3
ASR_COMPUTE_TYPE=int8_float16
ASR_DEVICE=cuda
TTS_DEVICE=cuda
```

### Development Configuration

For development with minimal resource constraints:

```bash
# .env
ENV=dev

# Resource Safeguards - Very permissive
MEMORY_THRESHOLD_PERCENT=98
SWAP_THRESHOLD_PERCENT=95
MAX_UPLOAD_SIZE_MB=500

# Concurrency - Unlimited
ASR_MAX_CONCURRENT_REQUESTS=10
TTS_MAX_CONCURRENT_REQUESTS=10

# Idle Timeout - Disabled (keep models loaded)
ASR_IDLE_TIMEOUT_MINUTES=0
TTS_IDLE_TIMEOUT_MINUTES=0

# Models - Smaller for faster iteration
ASR_MODEL=base
ASR_DEVICE=cpu
TTS_DEVICE=cpu
```

## Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:5001/health

# Detailed metrics
curl http://localhost:5001/health/detailed
```

### Service Logs

```bash
# ASR service logs
sudo journalctl -u voice-stack-asr -f

# TTS service logs
sudo journalctl -u voice-stack-tts -f

# Filter for resource events
sudo journalctl -u voice-stack-asr -f | grep -i "resource\|memory\|slot\|idle"
```

### System Monitoring

```bash
# Monitor memory and CPU
htop

# Monitor GPU usage (CUDA)
watch -n 1 nvidia-smi

# Monitor memory specifically
watch -n 1 free -h
```

## Troubleshooting

### Issue: Frequent 503 Errors

**Symptom:** Requests frequently rejected with "Insufficient memory available"

**Causes:**
- Thresholds set too low for actual usage
- Insufficient system RAM
- Memory leak (check with sustained monitoring)

**Solutions:**
```bash
# Option 1: Increase thresholds (if safe)
MEMORY_THRESHOLD_PERCENT=95

# Option 2: Use smaller models
ASR_MODEL=medium
ASR_COMPUTE_TYPE=int8

# Option 3: Enable more aggressive idle timeout
ASR_IDLE_TIMEOUT_MINUTES=15
```

### Issue: Frequent 429 Errors

**Symptom:** Requests rejected with "Too many concurrent requests"

**Causes:**
- Concurrency limit too low for traffic
- Slow inference causing request queuing

**Solutions:**
```bash
# Option 1: Increase concurrent slots (if resources allow)
ASR_MAX_CONCURRENT_REQUESTS=4

# Option 2: Optimize inference speed
ASR_COMPUTE_TYPE=int8  # Faster than float16
ASR_DEVICE=cuda        # GPU faster than CPU
ASR_VAD_ENABLED=true   # Reduces processing time
```

### Issue: Models Not Unloading

**Symptom:** Models stay in memory despite idle timeout configured

**Causes:**
- Idle timeout disabled (`IDLE_TIMEOUT_MINUTES=0`)
- Background task not running
- Continuous traffic preventing idle state

**Solutions:**
```bash
# Verify settings
ASR_IDLE_TIMEOUT_MINUTES=60  # Must be > 0

# Check logs for idle timeout messages
sudo journalctl -u voice-stack-asr -f | grep -i "idle"

# Verify background task started
# Look for "Idle timeout checker started" in logs
```

## Testing

### Verify Resource Guards

```bash
# Test memory threshold (requires filling RAM first)
# The middleware should reject with 503 when threshold exceeded

# Test file size limit
dd if=/dev/zero of=large.wav bs=1M count=150  # Create 150MB file
curl -X POST http://localhost:5001/v1/audio/transcriptions \
  -F file=@large.wav  # Should reject with 413
```

### Verify Concurrency Control

```bash
# Run load test with workers > MAX_CONCURRENT_REQUESTS
python scripts/load_test.py --service asr --workers 10 --requests 20

# Should see some 429 errors if MAX_CONCURRENT_REQUESTS < workers
# and requests take time to process
```

### Verify Idle Timeout

```bash
# 1. Set short timeout for testing
# Edit .env: ASR_IDLE_TIMEOUT_MINUTES=2

# 2. Restart service
sudo systemctl restart voice-stack-asr

# 3. Make a request (model loads)
curl http://localhost:5001/health/detailed  # model.loaded=true

# 4. Wait 3 minutes

# 5. Check if model unloaded
curl http://localhost:5001/health/detailed  # model.loaded should still be true
# (model stays loaded even after unload, just frees memory)

# Check logs for unload message
sudo journalctl -u voice-stack-asr -n 50 | grep -i "unload"
```

## Performance Impact

### Resource Guard Middleware

- **Overhead:** < 1ms per request (psutil memory check)
- **Impact:** Negligible on request latency
- **Benefit:** Prevents OOM crashes that would require service restart

### Concurrency Control

- **Overhead:** < 1ms per request (semaphore acquire/release)
- **Impact:** May increase queue wait times during high load
- **Benefit:** Prevents memory exhaustion from concurrent model inference

### Idle Timeout

- **Overhead:** Negligible (background task runs once per minute)
- **First Request Impact:** +2-10 seconds for model loading after unload
- **Benefit:** Frees 2-8GB+ of RAM/VRAM when idle

## Future Enhancements

Potential improvements for future versions:

1. **Request Queueing:** Instead of rejecting with 429, queue requests up to a limit
2. **Adaptive Thresholds:** Automatically adjust thresholds based on system behavior
3. **Metrics Export:** Prometheus/OpenTelemetry integration for monitoring
4. **Per-Model Concurrency:** Different limits for different model sizes
5. **Priority Queues:** Allow high-priority requests to bypass limits
6. **Graceful Degradation:** Automatically switch to smaller models under load

## Related Documentation

- [Load Testing Guide](LOAD_TESTING.md) - Comprehensive guide to load testing
- [Architecture Documentation](ARCHITECTURE.md) - System architecture overview
- [Service Installation](../scripts/SERVICE_INSTALLATION.md) - Production setup guide
- [Whisper Configuration](WHISPER_CHEAT_SHEET.md) - ASR model tuning guide

## Credits

These features were implemented to ensure Voice Stack can run reliably in production environments, especially on resource-constrained systems or shared GPU servers.
