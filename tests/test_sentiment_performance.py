"""
Performance tests for sentiment analysis batch processing capabilities.
Tests system performance under various load conditions and validates processing times.
"""
import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List

from app.models.sentiment import (
    SentimentLabel, SentimentCategory, SentimentResult, 
    AnalysisContext, NFLContext, DataSource
)
from app.services.sentiment_service import SentimentAnalysisService
from app.services.nfl_sentiment_engine import NFLSentimentEngine


class TestSentimentAnalysisPerformance:
    """Test performance characteristics of sentiment analysis service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SentimentAnalysisService()
    
    @pytest.mark.asyncio
    async def test_single_analysis_performance(self):
        """Test performance of single text analysis."""
        text = "Amazing touchdown pass by the quarterback in the fourth quarter!"
        
        # Measure processing time
        start_time = time.time()
        result = await self.service.analyze_text(text)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Verify result
        assert isinstance(result, SentimentResult)
        assert result.processing_time_ms > 0
        
        # Performance assertions
        assert processing_time < 1000  # Should complete within 1 second
        assert result.processing_time_ms < 500  # Internal timing should be under 500ms
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance_small(self):
        """Test performance of small batch analysis (10 texts)."""
        texts = [
            f"Amazing touchdown pass number {i}!" for i in range(10)
        ]
        
        start_time = time.time()
        results = await self.service.analyze_batch(texts)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_text = total_time / len(texts)
        
        # Verify results
        assert len(results) == 10
        assert all(isinstance(result, SentimentResult) for result in results)
        
        # Performance assertions
        assert total_time < 5000  # Should complete within 5 seconds
        assert avg_time_per_text < 500  # Average under 500ms per text
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance_medium(self):
        """Test performance of medium batch analysis (50 texts)."""
        texts = [
            f"Great performance by player {i} in the game!" for i in range(50)
        ]
        
        start_time = time.time()
        results = await self.service.analyze_batch(texts)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_text = total_time / len(texts)
        
        # Verify results
        assert len(results) == 50
        
        # Performance assertions
        assert total_time < 15000  # Should complete within 15 seconds
        assert avg_time_per_text < 300  # Should benefit from batch processing
    
    @pytest.mark.asyncio
    async def test_batch_analysis_performance_large(self):
        """Test performance of large batch analysis (100 texts)."""
        texts = [
            f"Team performance analysis text {i} with various NFL keywords touchdown fumble quarterback" 
            for i in range(100)
        ]
        
        start_time = time.time()
        results = await self.service.analyze_batch(texts)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_text = total_time / len(texts)
        
        # Verify results
        assert len(results) == 100
        
        # Performance assertions
        assert total_time < 30000  # Should complete within 30 seconds
        assert avg_time_per_text < 300  # Batch processing efficiency
    
    @pytest.mark.asyncio
    async def test_text_length_performance_impact(self):
        """Test how text length affects processing performance."""
        # Short text
        short_text = "TD!"
        
        # Medium text
        medium_text = "Amazing touchdown pass by the quarterback in the fourth quarter clutch performance!"
        
        # Long text
        long_text = " ".join([
            "This is a very long text about NFL performance analysis",
            "including multiple keywords like touchdown, quarterback, fumble,",
            "interception, amazing, terrible, injury, trade, coaching decisions,",
            "betting lines, fantasy football implications, and much more detailed",
            "analysis of the game situation and player performance metrics"
        ] * 5)  # Repeat to make it longer
        
        # Test each text length
        texts_to_test = [
            ("short", short_text),
            ("medium", medium_text),
            ("long", long_text)
        ]
        
        performance_results = {}
        
        for text_type, text in texts_to_test:
            times = []
            
            # Run multiple times for statistical significance
            for _ in range(5):
                start_time = time.time()
                result = await self.service.analyze_text(text)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)
            
            performance_results[text_type] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "text_length": len(text)
            }
        
        # Verify that longer texts don't cause exponential slowdown
        short_avg = performance_results["short"]["avg_time"]
        medium_avg = performance_results["medium"]["avg_time"]
        long_avg = performance_results["long"]["avg_time"]
        
        # Medium text should not be more than 3x slower than short
        assert medium_avg < short_avg * 3
        
        # Long text should not be more than 5x slower than short
        assert long_avg < short_avg * 5
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self):
        """Test performance under concurrent analysis load."""
        texts = [
            f"Concurrent analysis test {i} with NFL keywords touchdown quarterback"
            for i in range(20)
        ]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        for text in texts:
            result = await self.service.analyze_text(text)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        concurrent_tasks = [self.service.analyze_text(text) for text in texts]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(sequential_results) == len(concurrent_results) == 20
        
        # Concurrent should be faster (or at least not significantly slower)
        # Allow some overhead for task management
        assert concurrent_time < sequential_time * 1.2
    
    @pytest.mark.asyncio
    async def test_memory_usage_batch_processing(self):
        """Test memory efficiency of batch processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch
        texts = [
            f"Memory test text {i} with NFL analysis keywords touchdown quarterback fumble"
            for i in range(100)
        ]
        
        results = await self.service.analyze_batch(texts)
        
        # Measure memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify results
        assert len(results) == 100
        
        # Memory increase should be reasonable (less than 100MB for 100 texts)
        assert memory_increase < 100
    
    @pytest.mark.asyncio
    async def test_keyword_processing_performance(self):
        """Test performance impact of different keyword densities."""
        # Text with no NFL keywords
        no_keywords = "This is a generic text without any specific terms."
        
        # Text with few NFL keywords
        few_keywords = "The game was good and the team played well."
        
        # Text with many NFL keywords
        many_keywords = "Amazing touchdown pass by the quarterback, incredible performance, " \
                       "no fumbles or interceptions, elite player, clutch in fourth quarter, " \
                       "MVP candidate, pro bowl selection, record breaking, phenomenal game."
        
        texts_to_test = [
            ("no_keywords", no_keywords),
            ("few_keywords", few_keywords),
            ("many_keywords", many_keywords)
        ]
        
        performance_results = {}
        
        for text_type, text in texts_to_test:
            times = []
            
            for _ in range(10):  # More iterations for better statistics
                start_time = time.time()
                result = await self.service.analyze_text(text)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)
            
            performance_results[text_type] = {
                "avg_time": statistics.mean(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "keyword_count": len([kw for kw in self.service.nfl_config.get_all_keywords() 
                                    if kw in text.lower()])
            }
        
        # Verify that keyword density doesn't cause excessive slowdown
        no_kw_avg = performance_results["no_keywords"]["avg_time"]
        many_kw_avg = performance_results["many_keywords"]["avg_time"]
        
        # Many keywords should not be more than 2x slower
        assert many_kw_avg < no_kw_avg * 2


class TestNFLSentimentEnginePerformance:
    """Test performance characteristics of NFL sentiment engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = NFLSentimentEngine()
    
    @pytest.mark.asyncio
    async def test_enhanced_analysis_performance(self):
        """Test performance of enhanced NFL analysis."""
        text = "The patriots quarterback threw an amazing touchdown pass in the playoff game!"
        
        start_time = time.time()
        result = await self.engine.analyze_with_context(
            text=text,
            team_context="patriots",
            include_detailed_analysis=True
        )
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        
        # Verify enhanced result
        assert isinstance(result, SentimentResult)
        assert result.context.nfl_context is not None
        assert len(result.aspect_sentiments) > 0
        
        # Performance assertion
        assert processing_time < 2000  # Enhanced analysis should complete within 2 seconds
    
    @pytest.mark.asyncio
    async def test_batch_enhanced_analysis_performance(self):
        """Test performance of batch enhanced analysis."""
        texts = [
            f"Patriots quarterback {i} threw touchdown in playoff game!"
            for i in range(25)
        ]
        
        start_time = time.time()
        response = await self.engine.batch_analyze_with_context(
            texts=texts,
            team_context="patriots",
            include_detailed_analysis=True
        )
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_text = total_time / len(texts)
        
        # Verify results
        assert len(response.results) == 25
        assert response.total_processed == 25
        assert response.processing_time_ms > 0
        
        # Performance assertions
        assert total_time < 15000  # Should complete within 15 seconds
        assert avg_time_per_text < 600  # Average under 600ms per enhanced analysis
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing_performance(self):
        """Test performance of concurrent batch processing with limits."""
        texts = [f"Test text {i} with NFL keywords touchdown quarterback" for i in range(50)]
        
        # Test with different concurrency limits
        concurrency_limits = [5, 10, 20]
        performance_results = {}
        
        for limit in concurrency_limits:
            start_time = time.time()
            response = await self.engine.batch_analyze_with_context(
                texts=texts,
                max_concurrent=limit
            )
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            performance_results[limit] = {
                "total_time": total_time,
                "avg_time_per_text": total_time / len(texts),
                "results_count": len(response.results)
            }
        
        # Verify all results are complete
        for limit in concurrency_limits:
            assert performance_results[limit]["results_count"] == 50
        
        # Higher concurrency should generally be faster (within reason)
        # Allow for some variance due to overhead
        assert performance_results[20]["total_time"] <= performance_results[5]["total_time"] * 1.5
    
    @pytest.mark.asyncio
    async def test_context_extraction_performance(self):
        """Test performance of NFL context extraction."""
        texts_with_varying_context = [
            "Simple text",  # No context
            "The patriots played well",  # Team context
            "The quarterback threw to the wide receiver",  # Position context
            "Player suffered injury and is questionable for playoff game",  # Multiple contexts
            "Trade rumors about the star quarterback going to patriots in playoff push"  # Rich context
        ]
        
        performance_results = []
        
        for text in texts_with_varying_context:
            times = []
            
            for _ in range(10):
                start_time = time.time()
                context = self.engine._extract_enhanced_nfl_context(text)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(times)
            context_complexity = (
                len(context.team_mentions) + 
                len(context.position_mentions) + 
                int(context.injury_related) + 
                int(context.trade_related) +
                (1 if context.game_situation else 0)
            )
            
            performance_results.append({
                "text": text[:30] + "..." if len(text) > 30 else text,
                "avg_time": avg_time,
                "context_complexity": context_complexity
            })
        
        # Verify that context extraction is fast regardless of complexity
        for result in performance_results:
            assert result["avg_time"] < 50  # Should be under 50ms
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_performance(self):
        """Test performance of confidence score calculation."""
        test_cases = [
            ("short text", 0.5),
            ("medium length text with some NFL keywords touchdown quarterback", 0.7),
            ("very long text with many NFL keywords including touchdown, quarterback, " +
             "fumble, interception, amazing, incredible, elite, clutch, playoff, " +
             "championship, mvp, pro bowl, record breaking performance", 0.9)
        ]
        
        nfl_context = NFLContext(
            team_mentions=["patriots"],
            position_mentions=["QB"],
            game_situation="playoff"
        )
        
        for text, sentiment_score in test_cases:
            times = []
            
            for _ in range(100):  # Many iterations for micro-benchmark
                start_time = time.time()
                confidence = self.engine.calculate_confidence_score(text, sentiment_score, nfl_context)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000000)  # Microseconds
            
            avg_time = statistics.mean(times)
            
            # Verify confidence is calculated
            assert 0 <= confidence <= 1
            
            # Performance assertion - should be very fast
            assert avg_time < 1000  # Under 1000 microseconds (1ms)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test that performance metrics are accurately tracked."""
        # Reset metrics
        self.engine.total_analyses = 0
        self.engine.total_processing_time = 0.0
        
        # Perform several analyses
        texts = [f"Performance test {i}" for i in range(10)]
        
        for text in texts:
            await self.engine.analyze_with_context(text)
        
        # Check metrics
        metrics = self.engine.get_performance_metrics()
        
        assert metrics["total_analyses"] == 10
        assert metrics["total_processing_time_seconds"] > 0
        assert metrics["average_processing_time_seconds"] > 0
        assert metrics["average_processing_time_seconds"] == (
            metrics["total_processing_time_seconds"] / metrics["total_analyses"]
        )
    
    @pytest.mark.asyncio
    async def test_large_batch_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process maximum allowed batch size
        texts = [
            f"Large batch test {i} with NFL keywords touchdown quarterback fumble amazing"
            for i in range(100)
        ]
        
        response = await self.engine.batch_analyze_with_context(
            texts=texts,
            include_detailed_analysis=True
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify results
        assert len(response.results) == 100
        assert response.total_processed == 100
        
        # Memory usage should be reasonable
        assert memory_increase < 200  # Less than 200MB increase
    
    @pytest.mark.asyncio
    async def test_error_handling_performance_impact(self):
        """Test that error handling doesn't significantly impact performance."""
        # Mix of valid and problematic texts
        texts = [
            "Valid NFL text with touchdown",
            "",  # Empty text
            "Valid text again",
            "A" * 10000,  # Very long text
            "Normal text with quarterback",
            None,  # This would cause an error in real scenario
        ]
        
        # Filter out None values for actual test
        valid_texts = [text for text in texts if text is not None]
        
        start_time = time.time()
        
        # Process with error handling
        results = []
        for text in valid_texts:
            try:
                result = await self.engine.analyze_with_context(text)
                results.append(result)
            except Exception:
                # Error handling shouldn't significantly slow down processing
                pass
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Should still process valid texts efficiently
        assert len(results) >= 4  # At least the valid texts
        assert total_time < 5000  # Should complete within 5 seconds despite errors


class TestPerformanceBenchmarks:
    """Benchmark tests for performance comparison and regression detection."""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark throughput for different batch sizes."""
        engine = NFLSentimentEngine()
        batch_sizes = [1, 5, 10, 25, 50, 100]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            texts = [
                f"Benchmark text {i} with NFL keywords touchdown quarterback amazing performance"
                for i in range(batch_size)
            ]
            
            start_time = time.time()
            response = await engine.batch_analyze_with_context(texts)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = batch_size / total_time  # texts per second
            
            throughput_results[batch_size] = {
                "total_time": total_time,
                "throughput": throughput,
                "avg_time_per_text": total_time / batch_size
            }
        
        # Verify throughput increases with batch size (up to a point)
        assert throughput_results[10]["throughput"] > throughput_results[1]["throughput"]
        assert throughput_results[50]["throughput"] > throughput_results[10]["throughput"]
        
        # Log results for performance monitoring
        print("\nThroughput Benchmark Results:")
        for batch_size, results in throughput_results.items():
            print(f"Batch size {batch_size}: {results['throughput']:.2f} texts/sec, "
                  f"{results['avg_time_per_text']*1000:.2f}ms per text")
    
    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        """Test latency percentiles for performance SLA validation."""
        engine = NFLSentimentEngine()
        text = "Performance test with NFL keywords touchdown quarterback amazing"
        
        # Collect many samples
        latencies = []
        for _ in range(100):
            start_time = time.time()
            await engine.analyze_with_context(text)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)  # milliseconds
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[49]  # 50th percentile (median)
        p95 = latencies[94]  # 95th percentile
        p99 = latencies[98]  # 99th percentile
        
        # Performance SLA assertions
        assert p50 < 500   # 50% of requests under 500ms
        assert p95 < 1000  # 95% of requests under 1 second
        assert p99 < 2000  # 99% of requests under 2 seconds
        
        print(f"\nLatency Percentiles:")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self):
        """Test scalability with increasing load."""
        engine = NFLSentimentEngine()
        
        # Test different load levels
        load_levels = [10, 25, 50, 100]  # Number of concurrent requests
        scalability_results = {}
        
        for load_level in load_levels:
            text = f"Scalability test with load level {load_level}"
            
            # Create concurrent tasks
            start_time = time.time()
            tasks = [engine.analyze_with_context(text) for _ in range(load_level)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            successful_results = [r for r in results if not isinstance(r, Exception)]
            error_rate = (load_level - len(successful_results)) / load_level
            
            scalability_results[load_level] = {
                "total_time": total_time,
                "successful_requests": len(successful_results),
                "error_rate": error_rate,
                "requests_per_second": len(successful_results) / total_time
            }
        
        # Verify system handles increasing load gracefully
        for load_level, results in scalability_results.items():
            assert results["error_rate"] < 0.05  # Less than 5% error rate
            assert results["requests_per_second"] > 5  # At least 5 requests per second
        
        print("\nScalability Benchmark Results:")
        for load_level, results in scalability_results.items():
            print(f"Load {load_level}: {results['requests_per_second']:.2f} req/sec, "
                  f"{results['error_rate']*100:.1f}% error rate")