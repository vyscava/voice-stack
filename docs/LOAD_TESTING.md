# Load Testing Guide

This guide explains how to perform load testing on the Voice Stack ASR and TTS services to verify performance, concurrency controls, and resource management.

## Overview

The `scripts/load_test.py` script provides comprehensive load testing capabilities for both ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) services. It helps you:

- Test concurrency controls (`MAX_CONCURRENT_REQUESTS`)
- Verify resource management (memory/swap thresholds)
- Measure performance under load
- Identify bottlenecks and issues
- Validate queue management and request handling

## Prerequisites

### Install Dependencies

```bash
pip install aiohttp aiofiles
```

### Ensure Services Are Running

Make sure the services you want to test are running:

```bash
# Check ASR service
curl http://localhost:5001/health

# Check TTS service
curl http://localhost:5002/health
```

## Basic Usage

### Quick Smoke Test

Test both services with minimal load:

```bash
python scripts/load_test.py --service both --workers 2 --requests 5
```

### Test ASR Service

Test the ASR endpoint with 10 concurrent workers making 50 total requests:

```bash
python scripts/load_test.py --service asr --workers 10 --requests 50
```

### Test TTS Service

Test the TTS endpoint with 5 concurrent workers making 20 total requests:

```bash
python scripts/load_test.py --service tts --workers 5 --requests 20
```

### Test Both Services

Test both ASR and TTS services:

```bash
python scripts/load_test.py --service both --workers 10 --requests 30
```

## Advanced Usage

### Duration-Based Testing

Run a test for a specific duration (in seconds) instead of a fixed number of requests:

```bash
# Run ASR load test for 60 seconds with 20 workers
python scripts/load_test.py --service asr --workers 20 --duration 60
```

### Custom Endpoints

Test services running on different hosts/ports:

```bash
python scripts/load_test.py \
  --service both \
  --asr-host http://192.168.1.100:5001 \
  --tts-host http://192.168.1.100:5002 \
  --workers 10 \
  --requests 50
```

### Verbose Output

Enable verbose logging to see individual request results:

```bash
python scripts/load_test.py --service asr --workers 5 --requests 20 --verbose
```

### Custom Timeout

Set a custom request timeout (in seconds):

```bash
python scripts/load_test.py --service tts --workers 10 --requests 50 --timeout 600
```

## Testing Scenarios

### 1. Test Concurrency Limits

Verify that `MAX_CONCURRENT_REQUESTS` is working correctly:

```bash
# If MAX_CONCURRENT_REQUESTS=2, test with more workers
python scripts/load_test.py --service asr --workers 10 --requests 20

# You should see requests queuing up, with only 2 executing simultaneously
```

**Expected Behavior:**
- Only 2 requests should process simultaneously
- Additional requests should queue
- No errors should occur (unless resources are exhausted)

### 2. Test Resource Thresholds

Verify memory and swap threshold protections:

```bash
# Increase load to trigger resource checks
python scripts/load_test.py --service both --workers 20 --requests 100
```

**Expected Behavior:**
- If memory usage exceeds `MEMORY_THRESHOLD_PERCENT`, requests should be rejected with HTTP 503
- If swap usage exceeds `SWAP_THRESHOLD_PERCENT`, requests should be rejected with HTTP 503
- Error messages should clearly indicate resource exhaustion

### 3. Test Idle Timeout

Verify that models are unloaded after idle timeout:

```bash
# Run a short test
python scripts/load_test.py --service asr --workers 2 --requests 5

# Wait for IDLE_TIMEOUT_MINUTES + 1 minute
# Then run another test - first request should take longer (model loading)
python scripts/load_test.py --service asr --workers 1 --requests 1
```

**Expected Behavior:**
- After idle timeout, models should be unloaded
- First request after timeout should take longer (model loading time)
- Subsequent requests should be faster (model already loaded)

### 4. Stress Test

Test system behavior under heavy load:

```bash
# Heavy load for 5 minutes
python scripts/load_test.py --service both --workers 50 --duration 300
```

**Expected Behavior:**
- System should remain stable
- Resource protections should prevent OOM crashes
- Error rates may increase under extreme load, but services should recover

### 5. Sustained Load Test

Test system stability over extended periods:

```bash
# Moderate load for 30 minutes
python scripts/load_test.py --service both --workers 10 --duration 1800
```

**Expected Behavior:**
- No memory leaks (monitor with `htop` or `nvidia-smi`)
- Consistent performance throughout test
- No crashes or service degradation

## Interpreting Results

### Success Metrics

The load test script provides detailed statistics:

```
ASR Load Test Results
====================
Total Requests:      50
Successful:          48 (96.0%)
Failed:              2

Timing Statistics:
  Average Duration:  2.34s
  Min Duration:      1.23s
  Max Duration:      5.67s
  Total Duration:    112.32s

Error Summary:
  [2x] HTTP 503: Resource threshold exceeded (memory: 92%)
```

### Key Metrics to Monitor

1. **Success Rate**: Should be >95% under normal load
   - Lower rates may indicate resource constraints or configuration issues

2. **Average Duration**: Baseline performance for your hardware
   - ASR (medium model): ~2-5s per request
   - TTS: ~1-3s per request

3. **Error Types**:
   - `HTTP 503` with "Resource threshold exceeded": Memory/swap limits reached (expected under heavy load)
   - `HTTP 503` with "Too many concurrent requests": Request queue is full
   - `Timeout`: Request took longer than configured timeout
   - `Connection errors`: Service may not be running

### Performance Baselines

Expected performance varies by hardware and model size:

#### ASR Performance (per request)

| Model      | CPU (4 cores) | GPU (CUDA)    |
|------------|---------------|---------------|
| tiny       | 0.5-1s        | 0.2-0.5s      |
| base       | 1-2s          | 0.3-0.7s      |
| small      | 2-4s          | 0.5-1.5s      |
| medium     | 4-8s          | 1-3s          |
| large-v3   | 10-20s        | 2-5s          |

#### TTS Performance (per request)

| Device     | Time per Request |
|------------|------------------|
| CPU        | 2-4s             |
| GPU (CUDA) | 1-2s             |

## Monitoring During Load Tests

### System Resources

Monitor system resources in a separate terminal:

```bash
# Monitor CPU, memory, and processes
htop

# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi

# Monitor memory usage
watch -n 1 free -h
```

### Service Logs

Monitor service logs for errors:

```bash
# ASR service logs
sudo journalctl -u voice-stack-asr -f

# TTS service logs
sudo journalctl -u voice-stack-tts -f
```

### Network Monitoring

Monitor network connections:

```bash
# Check open connections to ASR
sudo ss -tnp | grep :5001

# Check open connections to TTS
sudo ss -tnp | grep :5002
```

## Troubleshooting

### High Failure Rate

**Symptoms:** >10% request failure rate

**Possible Causes:**
1. Resource thresholds set too low
2. Insufficient system resources (RAM, VRAM, CPU)
3. Network issues
4. Service configuration issues

**Solutions:**
```bash
# Check resource usage
free -h
nvidia-smi  # If using GPU

# Increase thresholds in .env (if appropriate)
MEMORY_THRESHOLD_PERCENT=95
SWAP_THRESHOLD_PERCENT=90

# Reduce concurrent requests
ASR_MAX_CONCURRENT_REQUESTS=1
TTS_MAX_CONCURRENT_REQUESTS=1

# Restart services
sudo systemctl restart voice-stack-asr
sudo systemctl restart voice-stack-tts
```

### Slow Performance

**Symptoms:** Average duration significantly higher than expected

**Possible Causes:**
1. Wrong compute device (using CPU instead of GPU)
2. Inefficient compute type
3. CPU throttling or thermal issues
4. Insufficient system resources

**Solutions:**
```bash
# Verify GPU is being used (check logs)
sudo journalctl -u voice-stack-asr -n 50 | grep -i "device\|cuda\|gpu"

# Update .env for GPU
ASR_DEVICE=cuda
TTS_DEVICE=cuda

# Optimize compute type
ASR_COMPUTE_TYPE=int8_float16  # For GPU

# Check for thermal throttling
sensors  # Install: apt install lm-sensors
```

### Out of Memory Crashes

**Symptoms:** Services crash during load testing

**Possible Causes:**
1. Resource thresholds not configured
2. Thresholds set too high
3. Insufficient system RAM/VRAM
4. Memory leak (unlikely)

**Solutions:**
```bash
# Configure resource thresholds in .env
MEMORY_THRESHOLD_PERCENT=85
SWAP_THRESHOLD_PERCENT=75
MAX_UPLOAD_SIZE_MB=50

# Reduce model size
ASR_MODEL=medium  # Instead of large
ASR_COMPUTE_TYPE=int8  # Instead of float16

# Reduce concurrent requests
ASR_MAX_CONCURRENT_REQUESTS=1

# Monitor for memory leaks (run sustained test)
python scripts/load_test.py --service asr --workers 2 --duration 600
# Watch memory usage - should stabilize, not grow continuously
```

### Connection Refused Errors

**Symptoms:** `Connection refused` or `Cannot connect to host`

**Possible Causes:**
1. Service not running
2. Wrong host/port
3. Firewall blocking connections

**Solutions:**
```bash
# Check if services are running
sudo systemctl status voice-stack-asr
sudo systemctl status voice-stack-tts

# Check if ports are listening
sudo ss -tlnp | grep -E ':(5001|5002)'

# Start services if needed
sudo systemctl start voice-stack-asr
sudo systemctl start voice-stack-tts

# Test connectivity
curl http://localhost:5001/health
curl http://localhost:5002/health
```

## Best Practices

### 1. Start Small

Always start with a small smoke test before running heavy load tests:

```bash
python scripts/load_test.py --service both --workers 2 --requests 5
```

### 2. Gradually Increase Load

Incrementally increase load to find system limits:

```bash
# Start with 5 workers
python scripts/load_test.py --service asr --workers 5 --requests 20

# Increase to 10 workers
python scripts/load_test.py --service asr --workers 10 --requests 40

# Increase to 20 workers
python scripts/load_test.py --service asr --workers 20 --requests 80
```

### 3. Monitor System Resources

Always monitor system resources during load testing to identify bottlenecks.

### 4. Test One Service at a Time

When troubleshooting, test services individually:

```bash
# Test ASR only
python scripts/load_test.py --service asr --workers 10 --requests 50

# Test TTS only
python scripts/load_test.py --service tts --workers 10 --requests 50
```

### 5. Document Your Baselines

Record performance baselines for your hardware configuration for future reference.

### 6. Test After Configuration Changes

Always run load tests after changing:
- Model sizes
- Compute types
- Concurrency limits
- Resource thresholds
- Hardware upgrades

## Example Test Plan

Here's a recommended test sequence for validating a new deployment:

```bash
# 1. Smoke test - verify basic functionality
python scripts/load_test.py --service both --workers 2 --requests 5

# 2. Moderate load - test normal operation
python scripts/load_test.py --service both --workers 10 --requests 50

# 3. Concurrency test - verify queue management
python scripts/load_test.py --service asr --workers 20 --requests 100

# 4. Resource threshold test - trigger memory limits
python scripts/load_test.py --service both --workers 50 --requests 200

# 5. Sustained load - test stability
python scripts/load_test.py --service both --workers 10 --duration 1800

# 6. Stress test - find breaking point
python scripts/load_test.py --service both --workers 100 --duration 300
```

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md) - System architecture overview
- [Whisper Cheat Sheet](WHISPER_CHEAT_SHEET.md) - ASR model configuration
- [Service Installation](../scripts/SERVICE_INSTALLATION.md) - Service setup guide

## Support

If you encounter issues or have questions:

1. Check service logs: `sudo journalctl -u voice-stack-asr -f`
2. Review configuration in `.env`
3. Verify system resources are sufficient
4. Consult the troubleshooting section above
