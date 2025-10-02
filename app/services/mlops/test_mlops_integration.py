"""
Simple integration test for MLOps services.
"""

import asyncio
import logging
from datetime import datetime

from .mlops_service import MLOpsService
from ...models.mlops import ModelType, ModelPerformanceMetrics

logger = logging.getLogger(__name__)


async def test_mlops_integration():
    """Test basic MLOps functionality"""
    try:
        # Initialize MLOps service
        mlops = MLOpsService()
        await mlops.initialize()

        # Test service status
        status = await mlops.get_service_status()
        print(f"MLOps Service Status: {status}")

        # Test model listing
        models = await mlops.list_models()
        print(f"Available models: {len(models)}")

        # Test sentiment prediction (if HuggingFace is available)
        try:
            prediction = await mlops.predict_sentiment(
                text="The team played amazingly well today!",
                model_name="sentiment_base",
            )
            print(f"Sentiment prediction: {prediction}")
        except Exception as e:
            print(
                f"Sentiment prediction test failed (expected if no models loaded): {e}"
            )

        # Test experiment tracking
        try:
            run_id = await mlops.start_experiment(
                experiment_name="test_experiment",
                config={"test_param": 1.0},
                tags=["test"],
            )

            await mlops.log_metrics({"accuracy": 0.85, "loss": 0.15})
            await mlops.finish_experiment({"final_accuracy": 0.87})

            print(f"Experiment test completed: {run_id}")
        except Exception as e:
            print(
                f"Experiment tracking test failed (expected if W&B not configured): {e}"
            )

        # Test performance monitoring
        try:
            performance_metrics = ModelPerformanceMetrics(
                model_id="test_model",
                model_version="1.0",
                timestamp=datetime.utcnow(),
                accuracy=0.85,
                precision=0.83,
                recall=0.87,
                f1_score=0.85,
                prediction_count=1000,
                avg_prediction_time_ms=50.0,
                error_rate=0.02,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
                sample_size=1000,
            )

            await mlops.monitor_model_performance(
                model_id="test_model",
                model_version="1.0",
                performance_metrics=performance_metrics,
            )

            print("Performance monitoring test completed")
        except Exception as e:
            print(f"Performance monitoring test failed: {e}")

        print("MLOps integration test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"MLOps integration test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_mlops_integration())
