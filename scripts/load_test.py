#!/usr/bin/env python3
"""
Load Testing Script for Voice Stack ASR & TTS Services

This script performs concurrent load testing on ASR and TTS endpoints to verify:
- Concurrency controls (MAX_CONCURRENT_REQUESTS)
- Resource management (memory/swap thresholds)
- Idle timeout behavior
- Queue management and request handling
- Error handling under load

Usage:
    # Test ASR endpoint
    python scripts/load_test.py --service asr --workers 10 --requests 50

    # Test TTS endpoint
    python scripts/load_test.py --service tts --workers 5 --requests 20

    # Test both services
    python scripts/load_test.py --service both --workers 10 --requests 30

    # Quick smoke test
    python scripts/load_test.py --service both --workers 2 --requests 5

    # Stress test with custom duration
    python scripts/load_test.py --service asr --workers 20 --duration 60

Requirements:
    pip install aiohttp aiofiles
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import aiofiles
    import aiohttp
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install aiohttp aiofiles")
    sys.exit(1)


@dataclass
class LoadTestConfig:
    """Configuration for load testing"""

    service: str  # 'asr', 'tts', or 'both'
    workers: int  # Number of concurrent workers
    requests: int  # Total number of requests (per service)
    duration: int | None = None  # Optional: run for N seconds instead
    asr_host: str = "http://localhost:5001"
    tts_host: str = "http://localhost:5002"
    asr_endpoint: str = "/v1/audio/transcriptions"
    tts_endpoint: str = "/v1/audio/speech"
    timeout: int = 300  # Request timeout in seconds
    verbose: bool = False


@dataclass
class TestResult:
    """Result of a single request"""

    service: str
    success: bool
    duration: float
    status_code: int | None = None
    error: str | None = None
    response_size: int | None = None


@dataclass
class LoadTestStats:
    """Statistics for load testing"""

    service: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    results: list[TestResult] = field(default_factory=list)

    def add_result(self, result: TestResult):
        """Add a test result and update statistics"""
        self.results.append(result)
        self.total_requests += 1

        if result.success:
            self.successful_requests += 1
            self.total_duration += result.duration
            self.min_duration = min(self.min_duration, result.duration)
            self.max_duration = max(self.max_duration, result.duration)
        else:
            self.failed_requests += 1

    @property
    def avg_duration(self) -> float:
        """Average duration of successful requests"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_duration / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def print_summary(self):
        """Print statistics summary"""
        print(f"\n{'=' * 60}")
        print(f"{self.service.upper()} Load Test Results")
        print(f"{'=' * 60}")
        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.success_rate:.1f}%)")
        print(f"Failed:              {self.failed_requests}")

        if self.successful_requests > 0:
            print("\nTiming Statistics:")
            print(f"  Average Duration:  {self.avg_duration:.2f}s")
            print(f"  Min Duration:      {self.min_duration:.2f}s")
            print(f"  Max Duration:      {self.max_duration:.2f}s")
            print(f"  Total Duration:    {self.total_duration:.2f}s")

        # Count error types
        if self.failed_requests > 0:
            error_types = {}
            for result in self.results:
                if not result.success and result.error:
                    error_msg = result.error[:100]  # Truncate long errors
                    error_types[error_msg] = error_types.get(error_msg, 0) + 1

            print("\nError Summary:")
            for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  [{count}x] {error}")


class LoadTester:
    """Main load testing class"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.asr_stats = LoadTestStats(service="asr")
        self.tts_stats = LoadTestStats(service="tts")
        self.start_time: float | None = None
        self.test_audio_file: Path | None = None

    async def setup(self):
        """Setup test environment"""
        print("Setting up load test environment...")

        # Find or create a test audio file for ASR
        test_audio_paths = [
            Path("tests/data/test_audio.wav"),
            Path("tests/data/sample.wav"),
            Path("tests/fixtures/audio/test.wav"),
        ]

        for path in test_audio_paths:
            if path.exists():
                self.test_audio_file = path
                print(f"Using test audio file: {path}")
                break

        if not self.test_audio_file:
            print("Warning: No test audio file found. Creating a minimal test file...")
            # Create a minimal WAV file for testing
            await self._create_test_audio()

    async def _create_test_audio(self):
        """Create a minimal test audio file"""
        import array
        import wave

        # Create tests/data directory if it doesn't exist
        test_dir = Path("tests/data")
        test_dir.mkdir(parents=True, exist_ok=True)

        self.test_audio_file = test_dir / "load_test_audio.wav"

        # Create a 1-second silent WAV file
        sample_rate = 16000
        duration = 1  # seconds

        with wave.open(str(self.test_audio_file), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Create silent audio
            samples = array.array("h", [0] * (sample_rate * duration))
            wav_file.writeframes(samples.tobytes())

        print(f"Created test audio file: {self.test_audio_file}")

    async def test_asr_request(self, session: aiohttp.ClientSession, request_id: int) -> TestResult:
        """Perform a single ASR transcription request"""
        start = time.time()

        try:
            # Read audio file
            async with aiofiles.open(self.test_audio_file, "rb") as f:
                audio_data = await f.read()

            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field("file", audio_data, filename="test_audio.wav", content_type="audio/wav")
            data.add_field("model", "whisper-1")
            data.add_field("response_format", "json")

            # Make request
            url = f"{self.config.asr_host}{self.config.asr_endpoint}"
            async with session.post(url, data=data, timeout=self.config.timeout) as response:
                duration = time.time() - start
                response_text = await response.text()

                if response.status == 200:
                    if self.config.verbose:
                        print(f"[ASR-{request_id}] SUCCESS in {duration:.2f}s")
                    return TestResult(
                        service="asr",
                        success=True,
                        duration=duration,
                        status_code=response.status,
                        response_size=len(response_text),
                    )
                else:
                    if self.config.verbose:
                        print(f"[ASR-{request_id}] FAILED: HTTP {response.status}")
                    return TestResult(
                        service="asr",
                        success=False,
                        duration=duration,
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_text[:200]}",
                    )

        except asyncio.TimeoutError:
            duration = time.time() - start
            error_msg = f"Timeout after {duration:.2f}s"
            if self.config.verbose:
                print(f"[ASR-{request_id}] {error_msg}")
            return TestResult(service="asr", success=False, duration=duration, error=error_msg)

        except Exception as e:
            duration = time.time() - start
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.config.verbose:
                print(f"[ASR-{request_id}] ERROR: {error_msg}")
            return TestResult(service="asr", success=False, duration=duration, error=error_msg)

    async def test_tts_request(self, session: aiohttp.ClientSession, request_id: int) -> TestResult:
        """Perform a single TTS synthesis request"""
        start = time.time()

        try:
            # Prepare request data
            request_data = {
                "model": "tts-1",
                "input": "This is a load test for the text to speech service.",
                "voice": "default",
            }

            # Make request
            url = f"{self.config.tts_host}{self.config.tts_endpoint}"
            async with session.post(url, json=request_data, timeout=self.config.timeout) as response:
                duration = time.time() - start
                response_data = await response.read()

                if response.status == 200:
                    if self.config.verbose:
                        print(f"[TTS-{request_id}] SUCCESS in {duration:.2f}s")
                    return TestResult(
                        service="tts",
                        success=True,
                        duration=duration,
                        status_code=response.status,
                        response_size=len(response_data),
                    )
                else:
                    response_text = response_data.decode("utf-8", errors="replace")
                    if self.config.verbose:
                        print(f"[TTS-{request_id}] FAILED: HTTP {response.status}")
                    return TestResult(
                        service="tts",
                        success=False,
                        duration=duration,
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_text[:200]}",
                    )

        except asyncio.TimeoutError:
            duration = time.time() - start
            error_msg = f"Timeout after {duration:.2f}s"
            if self.config.verbose:
                print(f"[TTS-{request_id}] {error_msg}")
            return TestResult(service="tts", success=False, duration=duration, error=error_msg)

        except Exception as e:
            duration = time.time() - start
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.config.verbose:
                print(f"[TTS-{request_id}] ERROR: {error_msg}")
            return TestResult(service="tts", success=False, duration=duration, error=error_msg)

    async def worker(self, worker_id: int, request_queue: asyncio.Queue, service: str):
        """Worker that processes requests from the queue"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Get request from queue (non-blocking)
                    request_id = await asyncio.wait_for(request_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check if we should continue running
                    if self.config.duration:
                        elapsed = time.time() - self.start_time
                        if elapsed >= self.config.duration:
                            break
                    continue

                try:
                    # Perform the request
                    if service == "asr":
                        result = await self.test_asr_request(session, request_id)
                        self.asr_stats.add_result(result)
                    else:  # tts
                        result = await self.test_tts_request(session, request_id)
                        self.tts_stats.add_result(result)

                finally:
                    request_queue.task_done()

    async def run_load_test(self, service: str):
        """Run load test for a specific service"""
        print(f"\nStarting {service.upper()} load test...")
        print(f"  Workers: {self.config.workers}")

        if self.config.duration:
            print(f"  Duration: {self.config.duration}s")
        else:
            print(f"  Requests: {self.config.requests}")

        # Create request queue
        request_queue = asyncio.Queue()

        # Populate queue with request IDs
        if self.config.duration:
            # For duration-based tests, we'll add requests dynamically
            num_initial_requests = self.config.workers * 10
            for i in range(num_initial_requests):
                await request_queue.put(i)
        else:
            for i in range(self.config.requests):
                await request_queue.put(i)

        # Create worker tasks
        workers = []
        for i in range(self.config.workers):
            worker_task = asyncio.create_task(self.worker(i, request_queue, service))
            workers.append(worker_task)

        # If duration-based, keep adding requests until time is up
        if self.config.duration:

            async def request_feeder():
                request_id = num_initial_requests
                while (time.time() - self.start_time) < self.config.duration:
                    await request_queue.put(request_id)
                    request_id += 1
                    await asyncio.sleep(0.1)  # Small delay to avoid overwhelming

            feeder_task = asyncio.create_task(request_feeder())
            await feeder_task

        # Wait for all requests to complete or timeout
        try:
            await asyncio.wait_for(request_queue.join(), timeout=self.config.timeout * 2)
        except asyncio.TimeoutError:
            print(f"Warning: {service.upper()} test timed out waiting for requests to complete")

        # Cancel workers
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    async def run(self):
        """Run the complete load test"""
        await self.setup()

        self.start_time = time.time()

        if self.config.service in ("asr", "both"):
            await self.run_load_test("asr")

        if self.config.service in ("tts", "both"):
            await self.run_load_test("tts")

        total_time = time.time() - self.start_time

        # Print results
        print(f"\n{'=' * 60}")
        print(f"LOAD TEST COMPLETED IN {total_time:.2f}s")
        print(f"{'=' * 60}")

        if self.config.service in ("asr", "both"):
            self.asr_stats.print_summary()

        if self.config.service in ("tts", "both"):
            self.tts_stats.print_summary()

        # Overall summary
        if self.config.service == "both":
            total_requests = self.asr_stats.total_requests + self.tts_stats.total_requests
            total_successful = self.asr_stats.successful_requests + self.tts_stats.successful_requests
            total_failed = self.asr_stats.failed_requests + self.tts_stats.failed_requests

            print(f"\n{'=' * 60}")
            print("OVERALL SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total Requests:      {total_requests}")
            print(f"Successful:          {total_successful} ({total_successful/total_requests*100:.1f}%)")
            print(f"Failed:              {total_failed}")
            print(f"Throughput:          {total_requests/total_time:.2f} req/s")
            print(f"{'=' * 60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Load testing script for Voice Stack services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ASR with 10 concurrent workers, 50 total requests
  python scripts/load_test.py --service asr --workers 10 --requests 50

  # Test TTS with 5 workers, 20 requests
  python scripts/load_test.py --service tts --workers 5 --requests 20

  # Test both services
  python scripts/load_test.py --service both --workers 10 --requests 30

  # Duration-based test (60 seconds)
  python scripts/load_test.py --service asr --workers 20 --duration 60

  # Quick smoke test
  python scripts/load_test.py --service both --workers 2 --requests 5
        """,
    )

    parser.add_argument(
        "--service", choices=["asr", "tts", "both"], default="both", help="Service to test (default: both)"
    )

    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers (default: 10)")

    parser.add_argument("--requests", type=int, default=50, help="Total number of requests per service (default: 50)")

    parser.add_argument("--duration", type=int, help="Run for N seconds instead of fixed request count")

    parser.add_argument(
        "--asr-host", default="http://localhost:5001", help="ASR service host (default: http://localhost:5001)"
    )

    parser.add_argument(
        "--tts-host", default="http://localhost:5002", help="TTS service host (default: http://localhost:5002)"
    )

    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Create config
    config = LoadTestConfig(
        service=args.service,
        workers=args.workers,
        requests=args.requests,
        duration=args.duration,
        asr_host=args.asr_host,
        tts_host=args.tts_host,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    # Run load test
    tester = LoadTester(config)

    try:
        asyncio.run(tester.run())
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
