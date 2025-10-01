"""
MLOps data models for model management, versioning, and monitoring.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    STAGING = "staging"
    RETIRED = "retired"
    FAILED = "failed"


class DeploymentStrategy(str, Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TEST = "a_b_test"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


class ModelType(str, Enum):
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    EMBEDDING = "embedding"


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelMetadata(BaseModel):
    """Model metadata for versioning and tracking"""
    model_id: str
    model_name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    # Model details
    framework: str = "transformers"  # transformers, pytorch, sklearn, etc.
    model_path: str
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = {}
    
    # Training details
    training_dataset: Optional[str] = None
    validation_dataset: Optional[str] = None
    training_duration_minutes: Optional[float] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Feature information
    feature_importance: Dict[str, float] = {}
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    
    # Deployment info
    deployment_config: Dict[str, Any] = {}
    resource_requirements: Dict[str, Any] = {}
    
    # Monitoring
    drift_threshold: float = 0.1
    performance_threshold: float = 0.8
    
    # Tags and metadata
    tags: List[str] = []
    description: Optional[str] = None
    notes: Optional[str] = None


class ModelDeployment(BaseModel):
    """Model deployment configuration and status"""
    deployment_id: str
    model_id: str
    model_version: str
    environment: str  # staging, production, canary
    strategy: DeploymentStrategy
    status: ModelStatus
    
    # Deployment configuration
    traffic_percentage: float = 100.0  # For A/B testing and canary deployments
    endpoint_url: Optional[str] = None
    replicas: int = 1
    resource_limits: Dict[str, str] = {}
    
    # Deployment timeline
    deployed_at: Optional[datetime] = None
    rollback_version: Optional[str] = None
    rollback_reason: Optional[str] = None
    
    # Health and monitoring
    health_check_url: Optional[str] = None
    monitoring_enabled: bool = True
    alerts_enabled: bool = True
    
    # Performance tracking
    request_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    
    created_at: datetime
    updated_at: datetime


class ExperimentRun(BaseModel):
    """Experiment tracking for model training and evaluation"""
    experiment_id: str
    run_id: str
    experiment_name: str
    run_name: Optional[str] = None
    status: ExperimentStatus
    
    # Experiment configuration
    model_type: ModelType
    hyperparameters: Dict[str, Any] = {}
    dataset_config: Dict[str, Any] = {}
    
    # Results
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, str] = {}  # artifact_name -> path/url
    
    # Tracking
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    
    # Environment
    python_version: Optional[str] = None
    dependencies: Dict[str, str] = {}
    git_commit: Optional[str] = None
    
    # Notes and tags
    notes: Optional[str] = None
    tags: List[str] = []
    
    # Parent/child relationships
    parent_run_id: Optional[str] = None
    child_runs: List[str] = []


class ModelPerformanceMetrics(BaseModel):
    """Real-time model performance metrics"""
    model_id: str
    model_version: str
    timestamp: datetime
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Operational metrics
    prediction_count: int = 0
    avg_prediction_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput_per_second: float = 0.0
    
    # Data quality metrics
    data_drift_score: Optional[float] = None
    feature_drift_scores: Dict[str, float] = {}
    prediction_drift_score: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = {}
    
    # Aggregation period
    period_start: datetime
    period_end: datetime
    sample_size: int


class ModelAlert(BaseModel):
    """Model monitoring alerts"""
    alert_id: str
    model_id: str
    model_version: str
    alert_type: str  # performance_degradation, data_drift, error_spike, etc.
    severity: str  # low, medium, high, critical
    
    # Alert details
    message: str
    description: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    
    # Status
    status: str = "active"  # active, acknowledged, resolved
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Actions
    recommended_actions: List[str] = []
    auto_actions_taken: List[str] = []


class FeatureStore(BaseModel):
    """Feature store configuration and metadata"""
    feature_group_name: str
    version: int
    description: Optional[str] = None
    
    # Schema
    features: Dict[str, str] = {}  # feature_name -> data_type
    primary_key: List[str] = []
    event_time_feature: Optional[str] = None
    
    # Configuration
    online_enabled: bool = True
    offline_enabled: bool = True
    
    # Statistics
    feature_statistics: Dict[str, Dict[str, Any]] = {}
    data_validation_rules: Dict[str, Any] = {}
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: List[str] = []


class ModelRegistry(BaseModel):
    """Model registry entry"""
    registry_id: str
    model_name: str
    current_version: str
    latest_version: str
    
    # Versions
    versions: List[ModelMetadata] = []
    
    # Registry metadata
    description: Optional[str] = None
    owner: str
    created_at: datetime
    updated_at: datetime
    
    # Lifecycle management
    retention_policy: Dict[str, Any] = {}
    auto_deployment_rules: Dict[str, Any] = {}


class ABTestConfig(BaseModel):
    """A/B testing configuration for model deployments"""
    test_id: str
    test_name: str
    model_a_id: str
    model_a_version: str
    model_b_id: str
    model_b_version: str
    
    # Traffic split
    traffic_split_percentage: float = 50.0  # Percentage for model B
    
    # Test configuration
    success_metrics: List[str] = []
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    test_duration_days: int = 7
    
    # Status
    status: str = "running"  # running, completed, stopped
    started_at: datetime
    ended_at: Optional[datetime] = None
    
    # Results
    results: Dict[str, Any] = {}
    winner: Optional[str] = None  # "model_a" or "model_b"
    statistical_significance: Optional[float] = None


class ModelRetrainingConfig(BaseModel):
    """Configuration for automated model retraining"""
    config_id: str
    model_name: str
    
    # Trigger conditions
    performance_threshold: float = 0.8
    data_drift_threshold: float = 0.1
    time_based_trigger: Optional[str] = None  # cron expression
    data_volume_trigger: Optional[int] = None  # minimum new samples
    
    # Training configuration
    training_config: Dict[str, Any] = {}
    validation_config: Dict[str, Any] = {}
    
    # Approval workflow
    auto_deploy: bool = False
    approval_required: bool = True
    approvers: List[str] = []
    
    # Notifications
    notification_channels: List[str] = []
    
    # Status
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    next_scheduled: Optional[datetime] = None
    
    created_at: datetime
    updated_at: datetime


class ModelPredictionLog(BaseModel):
    """Log entry for model predictions"""
    prediction_id: str
    model_id: str
    model_version: str
    timestamp: datetime
    
    # Input/Output
    input_data: Dict[str, Any]
    prediction: Any
    confidence: Optional[float] = None
    
    # Metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Performance
    prediction_time_ms: float
    preprocessing_time_ms: Optional[float] = None
    postprocessing_time_ms: Optional[float] = None
    
    # Feedback (if available)
    actual_outcome: Optional[Any] = None
    feedback_timestamp: Optional[datetime] = None
    feedback_source: Optional[str] = None


# Request/Response models for API endpoints

class ModelDeploymentRequest(BaseModel):
    model_id: str
    model_version: str
    environment: str
    strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE
    traffic_percentage: float = 100.0
    resource_config: Dict[str, Any] = {}


class ModelRetrainingRequest(BaseModel):
    model_name: str
    trigger_reason: str
    training_config: Optional[Dict[str, Any]] = None
    auto_deploy: bool = False


class ExperimentCreateRequest(BaseModel):
    experiment_name: str
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    tags: List[str] = []
    notes: Optional[str] = None


class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_type: ModelType
    model_path: str
    version: Optional[str] = None
    metrics: Dict[str, float] = {}
    tags: List[str] = []
    description: Optional[str] = None