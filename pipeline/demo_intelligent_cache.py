#!/usr/bin/env python3
"""
Demonstration of Intelligent Audio Processing Cache System

This script demonstrates how to use the intelligent caching system
to improve audio processing performance and reduce redundant operations.
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Import the intelligent cache system
from .intelligent_audio_cache import (
    AudioProcessingCache,
    CacheConfig,
    CacheLevel,
    get_audio_cache,
    initialize_audio_cache,
    shutdown_audio_cache
)

from .cache_analytics import (
    CacheAnalytics,
    CacheMonitor,
    create_cache_analytics,
    create_cache_monitor,
    generate_cache_report
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheDemo:
    """Demonstration class for intelligent caching features."""

    def __init__(self):
        self.cache_config = CacheConfig(
            max_memory_mb=256.0,  # 256MB memory cache
            max_disk_mb=1024.0,   # 1GB disk cache
            enable_compression=True,
            enable_async=True,
            cleanup_interval_minutes=5.0,
            enable_statistics=True
        )

        self.cache = None
        self.analytics = None
        self.monitor = None

    async def initialize_system(self):
        """Initialize the complete caching system."""
        logger.info("Initializing intelligent cache system...")

        # Initialize cache
        self.cache = await initialize_audio_cache(self.cache_config)

        # Initialize analytics
        self.analytics = create_cache_analytics(self.cache)

        # Initialize monitor
        self.monitor = create_cache_monitor(self.cache, self.analytics)

        # Start monitoring
        await self.monitor.start_monitoring()

        logger.info("Cache system initialized successfully")

    async def demonstrate_basic_caching(self, audio_file: Path):
        """Demonstrate basic caching functionality."""
        logger.info(f"Demonstrating basic caching with {audio_file}")

        if not audio_file.exists():
            logger.warning(f"Audio file {audio_file} does not exist, creating dummy data")
            # Create dummy audio data for demonstration
            audio_data = self._create_dummy_audio_data()
        else:
            # Load actual audio file
            audio_data = self._load_audio_file(audio_file)

        # Define processing configuration
        processing_config = {
            'quality': 'high',
            'normalize': True,
            'noise_reduction': True,
            'sample_rate': 44100
        }

        # Generate cache key
        cache_key = self.cache.generate_cache_key(
            audio_file,
            processing_config,
            algorithm_version="1.0"
        )

        logger.info(f"Generated cache key: {cache_key[:32]}...")

        # First access (cache miss)
        logger.info("First access - should be cache miss")
        start_time = time.time()
        result = self.cache.get(cache_key)
        first_access_time = time.time() - start_time

        if result is None:
            logger.info(f"✓ Cache miss as expected (took {first_access_time:.3f}s)")
        else:
            logger.warning("Unexpected cache hit on first access")

        # Store in cache
        logger.info("Storing result in cache...")
        start_time = time.time()
        success = await self.cache.put(
            cache_key,
            audio_data,
            audio_file,
            processing_config,
            algorithm_version="1.0",
            processing_time=0.1  # Simulated processing time
        )
        store_time = time.time() - start_time

        if success:
            logger.info(f"✓ Successfully stored in cache (took {store_time:.3f}s)")
        else:
            logger.error("✗ Failed to store in cache")
            return

        # Second access (cache hit)
        logger.info("Second access - should be cache hit")
        start_time = time.time()
        result = self.cache.get(cache_key)
        second_access_time = time.time() - start_time

        if result is not None:
            cached_data, metadata = result
            logger.info(f"✓ Cache hit! (took {second_access_time:.3f}s vs {first_access_time:.3f}s)")
            logger.info(f"  Cache level: {metadata.cache_level.value}")
            logger.info(f"  Access count: {metadata.access_count}")
            logger.info(f"  Data size: {metadata.compressed_size / (1024*1024):.2f}MB")
        else:
            logger.error("✗ Unexpected cache miss on second access")

        # Demonstrate cache statistics
        stats = self.cache.get_statistics()
        logger.info("Cache Statistics:")
        logger.info(f"  Hit ratio: {stats.hit_ratio:.2%}")
        logger.info(f"  Total requests: {stats.total_requests}")
        logger.info(f"  Memory usage: {stats.memory_usage_mb:.1f}MB")
        logger.info(f"  Disk usage: {stats.disk_usage_mb:.1f}MB")

    async def demonstrate_cache_invalidation(self, audio_file: Path):
        """Demonstrate cache invalidation when files change."""
        logger.info(f"Demonstrating cache invalidation with {audio_file}")

        processing_config = {'quality': 'medium'}
        cache_key = self.cache.generate_cache_key(audio_file, processing_config)

        # Store something in cache
        dummy_data = np.array([1, 2, 3, 4, 5])
        await self.cache.put(cache_key, dummy_data, audio_file, processing_config)

        # Verify it's cached
        result = self.cache.get(cache_key)
        if result is None:
            logger.error("Failed to store test data")
            return

        logger.info("✓ Test data cached successfully")

        # Invalidate cache for this file
        invalidated_count = self.cache.invalidate(audio_file)
        logger.info(f"Invalidated {invalidated_count} cache entries")

        # Verify it's gone
        result = self.cache.get(cache_key)
        if result is None:
            logger.info("✓ Cache invalidation successful")
        else:
            logger.error("✗ Cache invalidation failed")

    async def demonstrate_analytics(self):
        """Demonstrate cache analytics and monitoring."""
        logger.info("Demonstrating cache analytics...")

        # Collect some performance data
        for i in range(10):
            metrics = self.analytics.collect_metrics()
            logger.info(f"Metrics {i+1}: Hit ratio {metrics.hit_ratio:.2%}, "
                       f"Response time {metrics.avg_response_time_ms:.1f}ms")

            # Simulate some cache activity
            await asyncio.sleep(0.5)

        # Generate analysis report
        analysis = self.analytics.analyze_performance()
        logger.info("Performance Analysis:")
        logger.info(f"  Current hit ratio: {analysis['current_performance']['hit_ratio']:.2%}")
        logger.info(f"  Hit ratio trend: {analysis['trends']['hit_ratio_trend']}")
        logger.info(f"  Response time trend: {analysis['trends']['response_time_trend']}")

        # Generate recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            logger.info(f"Generated {len(recommendations)} optimization recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                logger.info(f"  - {rec['title']} ({rec['priority']} priority)")
        else:
            logger.info("No optimization recommendations generated")

        # Check for alerts
        alerts = self.analytics.check_alerts()
        if alerts:
            logger.info(f"Generated {len(alerts)} alerts:")
            for alert in alerts:
                logger.info(f"  - {alert.level.upper()}: {alert.title}")
        else:
            logger.info("No alerts generated")

    async def demonstrate_batch_processing(self, audio_files: List[Path]):
        """Demonstrate cache benefits for batch processing."""
        logger.info(f"Demonstrating batch processing with {len(audio_files)} files")

        processing_config = {
            'batch_mode': True,
            'quality': 'high',
            'optimize_for_batch': True
        }

        # First pass - process all files (cache misses expected)
        logger.info("First pass - processing all files...")
        first_pass_times = []

        for audio_file in audio_files:
            start_time = time.time()

            cache_key = self.cache.generate_cache_key(audio_file, processing_config)

            # Check cache
            result = self.cache.get(cache_key)
            if result is None:
                # Simulate processing
                await asyncio.sleep(0.1)  # Simulate processing time

                # Store in cache
                dummy_data = self._create_dummy_audio_data()
                await self.cache.put(
                    cache_key,
                    dummy_data,
                    audio_file,
                    processing_config,
                    processing_time=0.1
                )

            first_pass_times.append(time.time() - start_time)

        avg_first_pass = sum(first_pass_times) / len(first_pass_times)
        logger.info(f"First pass average time: {avg_first_pass:.3f}s")

        # Second pass - should be much faster due to caching
        logger.info("Second pass - should use cache...")
        second_pass_times = []

        for audio_file in audio_files:
            start_time = time.time()

            cache_key = self.cache.generate_cache_key(audio_file, processing_config)
            result = self.cache.get(cache_key)

            second_pass_times.append(time.time() - start_time)

        avg_second_pass = sum(second_pass_times) / len(second_pass_times)
        logger.info(f"Second pass average time: {avg_second_pass:.3f}s")

        # Calculate improvement
        if avg_first_pass > 0:
            improvement = (avg_first_pass - avg_second_pass) / avg_first_pass * 100
            logger.info(f"Cache improvement: {improvement:.1f}% faster")

        # Show final cache statistics
        stats = self.cache.get_statistics()
        logger.info("Final Cache Statistics:")
        logger.info(f"  Total requests: {stats.total_requests}")
        logger.info(f"  Cache hits: {stats.hits}")
        logger.info(f"  Cache misses: {stats.misses}")
        logger.info(f"  Hit ratio: {stats.hit_ratio:.2%}")
        logger.info(f"  Memory usage: {stats.memory_usage_mb:.1f}MB")
        logger.info(f"  Disk usage: {stats.disk_usage_mb:.1f}MB")

    def _create_dummy_audio_data(self):
        """Create dummy audio data for demonstration."""
        import numpy as np
        # Create 1 second of stereo audio at 44.1kHz
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        return np.random.randn(samples, 2).astype(np.float32)

    def _load_audio_file(self, audio_file: Path):
        """Load audio file data."""
        try:
            import librosa
            return librosa.load(audio_file, sr=None, mono=False)[0]
        except Exception as e:
            logger.warning(f"Failed to load audio file: {e}")
            return self._create_dummy_audio_data()

    async def generate_performance_report(self, output_path: Path):
        """Generate comprehensive performance report."""
        logger.info(f"Generating performance report: {output_path}")

        report = self.analytics.get_cache_efficiency_report()

        # Add system information
        report['system_info'] = {
            'cache_config': self.cache_config.__dict__,
            'generated_at': time.time(),
            'python_version': __import__('sys').version
        }

        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Performance report generated successfully")
        return report

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up cache system...")

        if self.monitor:
            await self.monitor.stop_monitoring()

        if self.cache:
            await shutdown_audio_cache()

        logger.info("Cleanup completed")


async def run_demo(audio_files: List[Path] = None):
    """Run the complete cache demonstration."""
    if audio_files is None:
        # Use default audio files if available
        audio_files = [
            Path("pipeline/audio_input/input.mp3"),
            Path("pipeline/audio_input/bear.mp3"),
            Path("other_audio/input.mp3")
        ]

        # Filter to existing files
        audio_files = [f for f in audio_files if f.exists()]

        if not audio_files:
            logger.info("No audio files found, using simulated data")
            audio_files = [Path("dummy1.mp3"), Path("dummy2.mp3"), Path("dummy3.mp3")]

    demo = CacheDemo()

    try:
        # Initialize system
        await demo.initialize_system()

        # Run demonstrations
        if audio_files:
            await demo.demonstrate_basic_caching(audio_files[0])
            await demo.demonstrate_cache_invalidation(audio_files[0])
            await demo.demonstrate_batch_processing(audio_files)

        # Analytics demonstration
        await demo.demonstrate_analytics()

        # Generate final report
        report_path = Path("cache_performance_report.json")
        report = await demo.generate_performance_report(report_path)

        logger.info("=== DEMONSTRATION SUMMARY ===")
        logger.info(f"Health Score: {report['summary']['overall_health_score']:.1f}/100")
        logger.info(f"Cache Efficiency: {report['summary']['cache_efficiency']:.2%}")
        logger.info(f"Avg Response Time: {report['summary']['average_response_time_ms']:.1f}ms")
        logger.info(f"Report saved to: {report_path}")

        return report

    finally:
        await demo.cleanup()


def main():
    """Main demonstration function."""
    print("Intelligent Audio Processing Cache System Demonstration")
    print("=" * 60)

    # Run demonstration
    try:
        report = asyncio.run(run_demo())

        print("\n🎉 Demonstration completed successfully!")
        print(f"📊 Overall health score: {report['summary']['overall_health_score']:.1f}/100")
        print(f"⚡ Cache efficiency: {report['summary']['cache_efficiency']:.2%}")
        print(f"📈 Performance improvement: Significant reduction in processing time for cached files")

    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    main()