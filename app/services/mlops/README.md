# MLOps Pipeline Implementation

This directory contains the complete MLOps pipeline implementation for the NFL Sentiment Analyzer, providing comprehensive model management, deployment, and monitoring capabilities.

## Overview

The MLOps pipeline includes:

- **Model Management**: HuggingFace integration for model storage and versioning
- **Experiment Tracking**: Weights & Biases integration for experiment monitoring
- **Feature Store**: Hopsworks integration for ML feature management
- **Model Deployment**: Automated deployment with A/B testing capabilities
- **Performance Monitoring**: Real-time model performance tracking and alerting
- **Automated Retraining**: Trigger-based model retraining pipeline

## Components

### Core Services

1. **MLOpsService** (`mlops_service.py`)
   - Main orchestration service
   - Unified interface for all MLOps operations
   - Coordinates between different MLOps components

2. **HuggingFaceModelService** (`huggingface_service.py`)
   - Model loading and inference
   - Model registration and versioning
   - Fine-tuning capabilities
   - Batch prediction support

3. **WandBService** (`wandb_service.py`)
   - Experiment tracking and logging
   - Hyperparameter optimization
   - Model comparison and analysis
   - Artifact management

4. **HopsworksService** (`hopsworks_service.py`)
   - Feature store management
   - Feature engineering pipelines
   - Training data preparation
   - Feature versioning and lineage

5. **ModelDeploymentService** (`model_deployment_service.py`)
   - Model deployment strategies (Blue-Green, Canary, A/B Testing)
   - Deployment monitoring and health checks
   - Automated rollback capabilities
   - Performance alerting

6. **ModelRetrainingService** (`model_retraining_service.py`)
   - Automated retraining triggers
   - Performance-based retraining
   - Data drift detection
   - Scheduled retraining jobs

## Configuration

### Environment Variables

```bash
# HuggingFace Configuration
HUGGINGFACE_TOKEN=your_hf_token_here

# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=nfl-sentiment-analyzer
WANDB_ENTITY=your_wandb_entity

# Hopsworks Configuration
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=nfl-sentiment-analyzer
HOPSWORKS_HOST=https://c.app.hopsworks.ai

# Model Storage
MODELS_DIR=./models
DEPLOYMENTS_DIR=./deployments
```

### Dependencies

The MLOps pipeline requires the following additional dependencies:

```
wandb==0.16.0
hopsworks==3.4.4
huggingface-hub==0.19.4
```

## Usage

### Initialize MLOps Services

```python
from app.services.mlops.mlops_service import mlops_service

# Initialize all services
await mlops_service.initialize()

# Check service status
status = await mlops_service.get_service_status()
```

### Model Management

```python
# Register a new model
model_metadata = await mlops_service.register_model(
    model_name="nfl_sentiment_v2",
    model_path="./models/nfl_sentiment_v2",
    model_type=ModelType.SENTIMENT_ANALYSIS,
    metrics={"accuracy": 0.87, "f1_score": 0.85},
    description="Enhanced NFL sentiment analysis model"
)

# Deploy model
deployment = await mlops_service.deploy_model(
    model_id=model_metadata.model_id,
    model_version=model_metadata.version,
    environment="production",
    strategy=DeploymentStrategy.BLUE_GREEN
)
```

### Experiment Tracking

```python
# Start experiment
run_id = await mlops_service.start_experiment(
    experiment_name="sentiment_model_training",
    config={"learning_rate": 2e-5, "batch_size": 16},
    tags=["sentiment", "nfl", "training"]
)

# Log metrics during training
await mlops_service.log_metrics({
    "train_loss": 0.15,
    "val_accuracy": 0.87,
    "epoch": 1
})

# Finish experiment
await mlops_service.finish_experiment({
    "final_accuracy": 0.89,
    "final_f1": 0.87
})
```

### A/B Testing

```python
# Create A/B test
test_id = await mlops_service.create_ab_test(
    test_name="sentiment_model_comparison",
    model_a_id="nfl_sentiment_v1",
    model_a_version="1.0",
    model_b_id="nfl_sentiment_v2",
    model_b_version="1.0",
    traffic_split=50.0,
    duration_days=7
)
```

### Model Monitoring

```python
# Monitor model performance
performance_metrics = ModelPerformanceMetrics(
    model_id="nfl_sentiment_v2",
    model_version="1.0",
    timestamp=datetime.utcnow(),
    accuracy=0.85,
    f1_score=0.83,
    error_rate=0.02,
    avg_prediction_time_ms=45.0,
    # ... other metrics
)

await mlops_service.monitor_model_performance(
    model_id="nfl_sentiment_v2",
    model_version="1.0",
    performance_metrics=performance_metrics
)
```

### Automated Retraining

```python
# Trigger manual retraining
run_id = await mlops_service.trigger_retraining(
    model_name="nfl_sentiment_v2",
    reason="performance_degradation",
    config={"epochs": 5, "learning_rate": 1e-5}
)

# Check retraining triggers
trigger_status = await mlops_service.retraining_service.check_retraining_triggers(
    "nfl_sentiment_v2"
)
```

## API Endpoints

The MLOps functionality is exposed through REST API endpoints at `/mlops/`:

### Model Management
- `POST /mlops/models/register` - Register a new model
- `GET /mlops/models` - List available models
- `GET /mlops/models/{model_id}` - Get model information

### Deployment
- `POST /mlops/deployments` - Deploy a model
- `GET /mlops/deployments` - List deployments
- `POST /mlops/deployments/{deployment_id}/rollback` - Rollback deployment

### Experiments
- `POST /mlops/experiments` - Start experiment
- `POST /mlops/experiments/metrics` - Log metrics
- `GET /mlops/experiments` - List experiments

### Retraining
- `POST /mlops/retraining/trigger` - Trigger retraining
- `GET /mlops/retraining/jobs` - List retraining jobs
- `GET /mlops/retraining/triggers/{model_name}` - Check triggers

### Predictions
- `POST /mlops/predict/sentiment` - Single prediction
- `POST /mlops/predict/sentiment/batch` - Batch predictions

### Monitoring
- `GET /mlops/status` - Service status
- `GET /mlops/health` - Health check

## Testing

Run the integration test to verify the MLOps setup:

```python
python -m app.services.mlops.test_mlops_integration
```

## Architecture

The MLOps pipeline follows a microservices architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer    │    │  MLOps Service   │    │  External APIs  │
│                │    │                  │    │                 │
│ FastAPI Routes │◄──►│ Orchestration    │◄──►│ HuggingFace Hub │
│                │    │                  │    │ Weights & Biases│
└─────────────────┘    └──────────────────┘    │ Hopsworks       │
                                               └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Core Services   │
                    │                  │
                    │ • Model Deploy   │
                    │ • Retraining     │
                    │ • Monitoring     │
                    │ • Feature Store  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Data Layer     │
                    │                  │
                    │ • MongoDB        │
                    │ • Redis Cache    │
                    │ • File Storage   │
                    └──────────────────┘
```

## Best Practices

1. **Model Versioning**: Always version your models and track metadata
2. **Experiment Tracking**: Log all experiments with proper tags and notes
3. **Performance Monitoring**: Set up alerts for model performance degradation
4. **Automated Testing**: Test models before deployment
5. **Gradual Rollouts**: Use canary or blue-green deployments for safety
6. **Data Quality**: Monitor for data drift and quality issues
7. **Resource Management**: Monitor resource usage and costs

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Ensure API keys are properly configured
2. **Model Loading Failures**: Check model paths and permissions
3. **Memory Issues**: Monitor memory usage during model loading
4. **Network Timeouts**: Configure appropriate timeouts for external APIs

### Logs

Check application logs for detailed error information:

```bash
# View MLOps service logs
grep "mlops" /var/log/nfl-analyzer/app.log

# View specific service logs
grep "HuggingFace\|WandB\|Hopsworks" /var/log/nfl-analyzer/app.log
```

## Future Enhancements

- [ ] Multi-cloud deployment support
- [ ] Advanced model explainability features
- [ ] Automated hyperparameter optimization
- [ ] Model performance benchmarking
- [ ] Cost optimization recommendations
- [ ] Advanced data drift detection algorithms